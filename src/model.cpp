#include "proofline.hpp"
#include "attention_cache.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace proofline {
std::size_t widthOf(const TensorType& type) {
    return type.shape.back();
}

std::size_t elementCount(const TensorType& type) {
    std::size_t count = 1;
    for (std::size_t dimension : type.shape) {
        if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("tensor shape is empty or too large");
        }
        count *= dimension;
    }
    return count;
}

std::string mathModeName(MathMode mode) {
    return mode == MathMode::Float64 ? "f64" : "bf16";
}

std::uint16_t encodeBFloat16(double value) {
    const float asFloat = static_cast<float>(value);
    if (!std::isfinite(asFloat)) {
        throw std::runtime_error("value is outside the finite float32 range");
    }
    std::uint32_t bits = std::bit_cast<std::uint32_t>(asFloat);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    const float rounded = std::bit_cast<float>(bits & 0xffff0000U);
    if (!std::isfinite(rounded)) {
        throw std::runtime_error("bfloat16 rounding overflowed to infinity");
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

float decodeBFloat16(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

double roundToBFloat16(double value) {
    return static_cast<double>(decodeBFloat16(encodeBFloat16(value)));
}

std::string typeName(ElementType type) {
    switch (type) {
    case ElementType::Float64:
        return "f64";
    }
    throw std::logic_error("unknown element type");
}

std::string tensorDescription(const IRValue& value) {
    std::string description = "tensor<";
    for (std::size_t i = 0; i < value.type.shape.size(); ++i) {
        if (i != 0) {
            description += "x";
        }
        description += std::to_string(value.type.shape[i]);
    }
    return description + "x" + typeName(value.type.elementType) + ">";
}

[[noreturn]] void failAtLine(std::size_t lineNumber, const std::string& message) {
    throw std::runtime_error("line " + std::to_string(lineNumber) + ": " + message);
}

std::size_t readPositiveWidth(std::istringstream& line, std::size_t lineNumber,
                              const std::string& description) {
    long long value = 0;
    if (!(line >> value) || value <= 0
        || static_cast<unsigned long long>(value)
            > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        failAtLine(lineNumber, description + " must be a positive integer");
    }
    return static_cast<std::size_t>(value);
}

ElementType readElementType(std::istringstream& line, std::size_t lineNumber) {
    std::string name;
    if (!(line >> name)) {
        failAtLine(lineNumber, "expected an element type such as f64");
    }
    if (name == "f64") {
        return ElementType::Float64;
    }
    failAtLine(lineNumber, "this version supports f64, not '" + name + "'");
}

double readFiniteNumber(std::istringstream& line, std::size_t lineNumber) {
    double value = 0.0;
    if (!(line >> value) || !std::isfinite(value)) {
        failAtLine(lineNumber, "expected a finite number");
    }
    return value;
}

void rejectExtraTokens(std::istringstream& line, std::size_t lineNumber) {
    std::string extra;
    if (line >> extra) {
        failAtLine(lineNumber, "unexpected extra value '" + extra + "'");
    }
}

ValueId addValue(Network& network, std::string name, ElementType type,
                 std::vector<std::size_t> shape) {
    const ValueId id = network.values.size();
    network.values.push_back(IRValue{
        std::move(name),
        TensorType{type, std::move(shape)},
    });
    return id;
}

ValueId addValue(Network& network, std::string name, ElementType type,
                 std::size_t width) {
    return addValue(network, std::move(name), type, std::vector<std::size_t>{width});
}

Network readNetwork(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open model file: " + path);
    }

    Network network;
    bool sawInput = false;
    std::size_t currentWidth = 0;
    ElementType currentType = ElementType::Float64;
    std::size_t lineNumber = 0;
    std::string rawLine;

    while (std::getline(file, rawLine)) {
        ++lineNumber;
        const std::size_t comment = rawLine.find('#');
        if (comment != std::string::npos) {
            rawLine.erase(comment);
        }

        std::istringstream line(rawLine);
        std::string command;
        if (!(line >> command)) {
            continue;
        }

        if (command == "input" || command == "input2d") {
            if (sawInput || !network.layers.empty()) {
                failAtLine(lineNumber, "input must appear once, before every layer");
            }

            std::vector<std::size_t> inputShape;
            if (command == "input2d") {
                const std::size_t rows = readPositiveWidth(line, lineNumber, "input row count");
                const std::size_t columns = readPositiveWidth(line, lineNumber, "input feature width");
                if (rows > std::numeric_limits<std::size_t>::max() / columns) {
                    failAtLine(lineNumber, "input tensor is too large");
                }
                inputShape = {rows, columns};
            } else {
                inputShape = {readPositiveWidth(line, lineNumber, "input width")};
            }
            currentWidth = inputShape.back();
            currentType = readElementType(line, lineNumber);
            network.input = addValue(network, "input", currentType, std::move(inputShape));
            network.output = network.input;
            rejectExtraTokens(line, lineNumber);
            sawInput = true;
            continue;
        }

        if (!sawInput) {
            failAtLine(lineNumber, "declare the input width before adding a layer");
        }

        if (command == "dense" || command == "dense_from") {
            ValueId sourceId = network.output;
            std::size_t inputWidth = currentWidth;
            if (command == "dense_from") {
                std::string sourceName;
                if (!(line >> sourceName)) {
                    failAtLine(lineNumber, "dense_from expects the name of an earlier tensor");
                }
                const auto source = std::find_if(network.values.begin(), network.values.end(),
                    [&sourceName](const IRValue& value) { return value.name == sourceName; });
                if (source == network.values.end()) {
                    failAtLine(lineNumber, "unknown tensor name '" + sourceName + "'");
                }
                sourceId = static_cast<ValueId>(std::distance(network.values.begin(), source));
                if (source->type.shape.size() != 1) {
                    failAtLine(lineNumber, "dense_from expects a one-dimensional tensor; use linear for matrices");
                }
                inputWidth = widthOf(source->type);
                currentType = source->type.elementType;
            } else if (network.values.at(sourceId).type.shape.size() != 1) {
                failAtLine(lineNumber, "dense expects a vector; use linear for matrices");
            }
            const std::size_t outputWidth =
                readPositiveWidth(line, lineNumber, "dense output width");
            const ElementType outputType = readElementType(line, lineNumber);
            if (outputWidth > std::numeric_limits<std::size_t>::max() / inputWidth) {
                failAtLine(lineNumber, "dense layer is too large");
            }

            Layer layer{LayerKind::Dense, {sourceId}, 0, {}, {}};
            layer.weights.reserve(inputWidth * outputWidth);
            layer.biases.reserve(outputWidth);

            for (std::size_t i = 0; i < inputWidth * outputWidth; ++i) {
                layer.weights.push_back(readFiniteNumber(line, lineNumber));
            }
            for (std::size_t i = 0; i < outputWidth; ++i) {
                layer.biases.push_back(readFiniteNumber(line, lineNumber));
            }
            rejectExtraTokens(line, lineNumber);

            layer.output = addValue(
                network, "v" + std::to_string(network.values.size()), outputType, outputWidth);
            network.layers.push_back(std::move(layer));
            currentWidth = outputWidth;
            currentType = outputType;
        } else if (command == "linear" || command == "linear_from") {
            ValueId sourceId = network.output;
            const TensorType* sourceType = &network.values.at(sourceId).type;
            if (command == "linear_from") {
                std::string sourceName;
                if (!(line >> sourceName)) {
                    failAtLine(lineNumber, "linear_from expects the name of an earlier matrix");
                }
                const auto source = std::find_if(network.values.begin(), network.values.end(),
                    [&sourceName](const IRValue& value) { return value.name == sourceName; });
                if (source == network.values.end()) {
                    failAtLine(lineNumber, "unknown tensor name '" + sourceName + "'");
                }
                sourceId = static_cast<ValueId>(std::distance(network.values.begin(), source));
                sourceType = &source->type;
            }
            if (sourceType->shape.size() != 2) {
                failAtLine(lineNumber, "linear expects a rank-two tensor");
            }
            const std::size_t rows = sourceType->shape[0];
            const std::size_t inputWidth = sourceType->shape[1];
            const std::size_t outputWidth =
                readPositiveWidth(line, lineNumber, "linear output width");
            const ElementType outputType = readElementType(line, lineNumber);
            if (rows > std::numeric_limits<std::size_t>::max() / outputWidth) {
                failAtLine(lineNumber, "linear output tensor is too large");
            }
            if (outputWidth > std::numeric_limits<std::size_t>::max() / inputWidth) {
                failAtLine(lineNumber, "linear layer is too large");
            }
            Layer layer{LayerKind::Linear, {sourceId}, 0, {}, {}};
            layer.weights.reserve(inputWidth * outputWidth);
            layer.biases.reserve(outputWidth);
            for (std::size_t i = 0; i < inputWidth * outputWidth; ++i) {
                layer.weights.push_back(readFiniteNumber(line, lineNumber));
            }
            for (std::size_t i = 0; i < outputWidth; ++i) {
                layer.biases.push_back(readFiniteNumber(line, lineNumber));
            }
            rejectExtraTokens(line, lineNumber);
            layer.output = addValue(network,
                "v" + std::to_string(network.values.size()), outputType,
                std::vector<std::size_t>{rows, outputWidth});
            network.layers.push_back(std::move(layer));
            currentWidth = outputWidth;
            currentType = outputType;
        } else if (command == "relu") {
            rejectExtraTokens(line, lineNumber);
            Layer layer{LayerKind::Relu, {network.output}, 0, {}, {}};
            layer.output = addValue(
                network, "v" + std::to_string(network.values.size()), currentType,
                network.values.at(network.output).type.shape);
            network.layers.push_back(std::move(layer));
        } else if (command == "rmsnorm") {
            const double epsilon = readFiniteNumber(line, lineNumber);
            if (epsilon <= 0.0) {
                failAtLine(lineNumber, "rmsnorm epsilon must be greater than zero");
            }
            Layer layer{LayerKind::RmsNorm, {network.output}, 0, {}, {}, epsilon};
            layer.weights.reserve(currentWidth);
            for (std::size_t i = 0; i < currentWidth; ++i) {
                layer.weights.push_back(readFiniteNumber(line, lineNumber));
            }
            rejectExtraTokens(line, lineNumber);
            // Keep normalization in f64 until a later projection explicitly chooses BF16.
            currentType = ElementType::Float64;
            layer.output = addValue(
                network, "v" + std::to_string(network.values.size()), currentType,
                network.values.at(network.output).type.shape);
            network.layers.push_back(std::move(layer));
        } else if (command == "softmax") {
            if (network.values.at(network.output).type.shape.size() != 1) {
                failAtLine(lineNumber, "softmax currently expects a vector");
            }
            rejectExtraTokens(line, lineNumber);
            Layer layer{LayerKind::Softmax, {network.output}, 0, {}, {}};
            currentType = ElementType::Float64;
            layer.output = addValue(
                network, "v" + std::to_string(network.values.size()), currentType,
                network.values.at(network.output).type.shape);
            network.layers.push_back(std::move(layer));
        } else if (command == "rope") {
            long long position = 0;
            if (!(line >> position) || position < 0) {
                failAtLine(lineNumber, "rope position must be a nonnegative integer");
            }
            const double base = readFiniteNumber(line, lineNumber);
            if (base <= 1.0) {
                failAtLine(lineNumber, "rope base must be greater than one");
            }
            rejectExtraTokens(line, lineNumber);
            if (currentWidth < 2 || currentWidth % 2 != 0) {
                failAtLine(lineNumber, "rope requires an even vector width of at least two");
            }
            const TensorType& ropeInputType = network.values.at(network.output).type;
            const std::size_t tokenCount = ropeInputType.shape.size() == 2
                ? ropeInputType.shape.at(0) : 1;
            const std::size_t lastPosition = std::numeric_limits<std::size_t>::max()
                - (tokenCount - 1);
            if (static_cast<unsigned long long>(position)
                > static_cast<unsigned long long>(lastPosition)) {
                failAtLine(lineNumber, "rope position range is too large for this tensor");
            }
            Layer layer{LayerKind::Rope, {network.output}, 0, {}, {}};
            layer.position = static_cast<std::size_t>(position);
            layer.ropeBase = base;
            currentType = ElementType::Float64;
            layer.output = addValue(
                network, "v" + std::to_string(network.values.size()), currentType,
                network.values.at(network.output).type.shape);
            network.layers.push_back(std::move(layer));
        } else if (command == "add") {
            std::string otherName;
            if (!(line >> otherName)) {
                failAtLine(lineNumber, "add expects the name of an earlier tensor");
            }
            rejectExtraTokens(line, lineNumber);
            const auto other = std::find_if(network.values.begin(), network.values.end(),
                [&otherName](const IRValue& value) { return value.name == otherName; });
            if (other == network.values.end()) {
                failAtLine(lineNumber, "unknown tensor name '" + otherName + "'");
            }
            const TensorType& currentTypeInfo = network.values.at(network.output).type;
            if (other->type.shape != currentTypeInfo.shape) {
                failAtLine(lineNumber, "add inputs must have the same shape");
            }
            if (other->type.elementType != currentType) {
                failAtLine(lineNumber, "add inputs must have the same element type");
            }
            Layer layer{LayerKind::Add, {network.output,
                static_cast<ValueId>(std::distance(network.values.begin(), other))}, 0, {}, {}};
            currentType = ElementType::Float64;
            layer.output = addValue(
                network, "v" + std::to_string(network.values.size()), currentType,
                currentTypeInfo.shape);
            network.layers.push_back(std::move(layer));
        } else if (command == "attention") {
            std::string queryName;
            std::string keyName;
            std::string valueName;
            std::string attentionOption;
            if (!(line >> queryName >> keyName >> valueName >> attentionOption)) {
                failAtLine(lineNumber,
                    "attention expects query, key, value tensor names, and a mask");
            }
            std::size_t headCount = 1;
            std::string maskName = attentionOption;
            if (attentionOption != "causal" && attentionOption != "full") {
                std::istringstream headToken(attentionOption);
                long long parsedHeads = 0;
                std::string extra;
                if (!(headToken >> parsedHeads) || parsedHeads <= 0 || (headToken >> extra)
                    || static_cast<unsigned long long>(parsedHeads)
                        > static_cast<unsigned long long>(
                            std::numeric_limits<std::size_t>::max())) {
                    failAtLine(lineNumber, "attention head count must be a positive integer");
                }
                headCount = static_cast<std::size_t>(parsedHeads);
                if (!(line >> maskName)) {
                    failAtLine(lineNumber, "attention head count must be followed by causal or full");
                }
            }
            rejectExtraTokens(line, lineNumber);
            const auto findValue = [&](const std::string& name) {
                const auto found = std::find_if(network.values.begin(), network.values.end(),
                    [&name](const IRValue& value) { return value.name == name; });
                if (found == network.values.end()) {
                    failAtLine(lineNumber, "unknown attention tensor '" + name + "'");
                }
                return static_cast<ValueId>(std::distance(network.values.begin(), found));
            };
            const ValueId queryId = findValue(queryName);
            const ValueId keyId = findValue(keyName);
            const ValueId valueId = findValue(valueName);
            const TensorType& queryType = network.values.at(queryId).type;
            const TensorType& keyType = network.values.at(keyId).type;
            const TensorType& valueType = network.values.at(valueId).type;
            if (queryType.shape.size() != 2 || keyType.shape.size() != 2
                || valueType.shape.size() != 2) {
                failAtLine(lineNumber, "attention expects rank-two token-by-feature tensors");
            }
            if (queryType.shape[0] != keyType.shape[0]
                || queryType.shape[0] != valueType.shape[0]
                || queryType.shape[1] != keyType.shape[1]) {
                failAtLine(lineNumber,
                    "self-attention requires matching token counts and query/key feature widths");
            }
            if (queryType.shape[1] % headCount != 0
                || valueType.shape[1] % headCount != 0) {
                failAtLine(lineNumber,
                    "query, key, and value feature widths must be divisible by the head count");
            }
            if (queryType.shape[0]
                > std::numeric_limits<std::size_t>::max() / valueType.shape[1]) {
                failAtLine(lineNumber, "attention output tensor is too large");
            }
            if (maskName != "causal" && maskName != "full") {
                failAtLine(lineNumber, "attention mask must be 'causal' or 'full'");
            }
            Layer layer{LayerKind::Attention, {queryId, keyId, valueId}, 0, {}, {}};
            layer.causal = maskName == "causal";
            layer.heads = headCount;
            currentType = ElementType::Float64;
            currentWidth = valueType.shape[1];
            layer.output = addValue(network,
                "v" + std::to_string(network.values.size()), currentType,
                std::vector<std::size_t>{queryType.shape[0], currentWidth});
            network.layers.push_back(std::move(layer));
        } else {
            failAtLine(lineNumber, "unknown operation '" + command + "'");
        }

        network.output = network.layers.back().output;
    }

    if (!sawInput) {
        throw std::runtime_error("model does not declare an input width");
    }
    if (network.layers.empty()) {
        throw std::runtime_error("model must contain at least one layer");
    }
    return network;
}

std::size_t denseLayerCount(const Network& network) {
    return static_cast<std::size_t>(std::count_if(
        network.layers.begin(), network.layers.end(),
        [](const Layer& layer) {
            return layer.kind == LayerKind::Dense || layer.kind == LayerKind::Linear;
        }));
}

std::vector<double> evaluate(const Network& network, std::vector<double> input,
                             const PrecisionPlan& plan) {
    const std::size_t inputCount = elementCount(network.values.at(network.input).type);
    if (input.size() != inputCount) {
        throw std::runtime_error("expected " + std::to_string(inputCount)
            + " input values, received " + std::to_string(input.size()));
    }

    std::vector<std::vector<double>> values(network.values.size());
    if (plan.denseModes.size() != denseLayerCount(network)) {
        throw std::runtime_error("precision plan does not match the number of projections");
    }
    values.at(network.input) = std::move(input);

    std::size_t denseIndex = 0;
    for (const Layer& layer : network.layers) {
        const std::vector<double>& source = values.at(layer.inputs.at(0));
        const IRValue& resultType = network.values.at(layer.output);
        const std::size_t outputWidth = widthOf(resultType.type);
        const std::size_t outputCount = elementCount(resultType.type);
        const std::size_t inputWidth = widthOf(network.values.at(layer.inputs.at(0)).type);
        std::vector<double>& result = values.at(layer.output);
        result.assign(outputCount, 0.0);

        if (layer.kind == LayerKind::Dense) {
            const MathMode mode = plan.denseModes.at(denseIndex++);
            for (std::size_t row = 0; row < outputWidth; ++row) {
                if (mode == MathMode::Float64) {
                    double sum = layer.biases.at(row);
                    for (std::size_t column = 0; column < inputWidth; ++column) {
                        sum += layer.weights.at(row * inputWidth + column)
                            * source.at(column);
                    }
                    result[row] = sum;
                } else {
                    float sum = static_cast<float>(
                        roundToBFloat16(layer.biases.at(row)));
                    for (std::size_t column = 0; column < inputWidth; ++column) {
                        const float weight = static_cast<float>(roundToBFloat16(
                            layer.weights.at(row * inputWidth + column)));
                        const float activation = static_cast<float>(
                            roundToBFloat16(source.at(column)));
                        const float product = weight * activation;
                        sum = sum + product;
                    }
                    result[row] = roundToBFloat16(sum);
                }
            }
        } else if (layer.kind == LayerKind::Linear) {
            const MathMode mode = plan.denseModes.at(denseIndex++);
            const TensorType& sourceType = network.values.at(layer.inputs.at(0)).type;
            const std::size_t inputWidth = widthOf(sourceType);
            const std::size_t rows = sourceType.shape.at(0);
            for (std::size_t token = 0; token < rows; ++token) {
                for (std::size_t row = 0; row < outputWidth; ++row) {
                    double sum = layer.biases.at(row);
                    if (mode == MathMode::Float64) {
                        for (std::size_t column = 0; column < inputWidth; ++column) {
                            sum += layer.weights.at(row * inputWidth + column)
                                * source.at(token * inputWidth + column);
                        }
                        result[token * outputWidth + row] = sum;
                    } else {
                        float lowPrecisionSum = static_cast<float>(
                            roundToBFloat16(layer.biases.at(row)));
                        for (std::size_t column = 0; column < inputWidth; ++column) {
                            const float weight = static_cast<float>(roundToBFloat16(
                                layer.weights.at(row * inputWidth + column)));
                            const float activation = static_cast<float>(roundToBFloat16(
                                source.at(token * inputWidth + column)));
                            lowPrecisionSum = lowPrecisionSum + weight * activation;
                        }
                        result[token * outputWidth + row] = roundToBFloat16(lowPrecisionSum);
                    }
                }
            }
        } else if (layer.kind == LayerKind::Relu) {
            for (std::size_t i = 0; i < outputCount; ++i) {
                result[i] = std::max(0.0, source.at(i));
            }
        } else if (layer.kind == LayerKind::RmsNorm) {
            const std::size_t rows = source.size() / inputWidth;
            for (std::size_t token = 0; token < rows; ++token) {
                const std::size_t offset = token * inputWidth;
                double normalizationScale = std::sqrt(layer.epsilon);
                for (std::size_t i = 0; i < inputWidth; ++i) {
                    normalizationScale = std::max(normalizationScale,
                        std::abs(source.at(offset + i)));
                }
                double squareSum = 0.0;
                for (std::size_t i = 0; i < inputWidth; ++i) {
                    const double scaled = source.at(offset + i) / normalizationScale;
                    squareSum += scaled * scaled;
                }
                const double epsilonRatio = std::sqrt(layer.epsilon) / normalizationScale;
                const double rms = normalizationScale * std::sqrt(
                    squareSum / static_cast<double>(inputWidth)
                    + epsilonRatio * epsilonRatio);
                for (std::size_t i = 0; i < inputWidth; ++i) {
                    result[offset + i] = (source.at(offset + i) / rms) * layer.weights.at(i);
                }
            }
        } else if (layer.kind == LayerKind::Softmax) {
            const std::size_t rows = source.size() / inputWidth;
            for (std::size_t token = 0; token < rows; ++token) {
                const std::size_t offset = token * inputWidth;
                const auto first = source.begin() + static_cast<std::ptrdiff_t>(offset);
                const double maximum = *std::max_element(first, first + inputWidth);
                double exponentialSum = 0.0;
                for (std::size_t i = 0; i < inputWidth; ++i) {
                    result[offset + i] = std::exp(source.at(offset + i) - maximum);
                    exponentialSum += result[offset + i];
                }
                for (std::size_t i = 0; i < inputWidth; ++i) {
                    result[offset + i] /= exponentialSum;
                }
            }
        } else if (layer.kind == LayerKind::Rope) {
            const std::size_t rows = source.size() / inputWidth;
            for (std::size_t token = 0; token < rows; ++token) {
                const std::size_t offset = token * inputWidth;
                const double position = static_cast<double>(layer.position + token);
                for (std::size_t pair = 0; pair < inputWidth / 2; ++pair) {
                    const double exponent = -2.0 * static_cast<double>(pair)
                        / static_cast<double>(inputWidth);
                    const double inverseFrequency = std::pow(layer.ropeBase, exponent);
                    const double angle = position * inverseFrequency;
                    const double cosine = std::cos(angle);
                    const double sine = std::sin(angle);
                    const double even = source.at(offset + 2 * pair);
                    const double odd = source.at(offset + 2 * pair + 1);
                    result[offset + 2 * pair] = even * cosine - odd * sine;
                    result[offset + 2 * pair + 1] = even * sine + odd * cosine;
                }
            }
        } else if (layer.kind == LayerKind::Attention) {
            const std::vector<double>& keys = values.at(layer.inputs.at(1));
            const std::vector<double>& projectedValues = values.at(layer.inputs.at(2));
            const TensorType& queryType = network.values.at(layer.inputs.at(0)).type;
            const TensorType& keyType = network.values.at(layer.inputs.at(1)).type;
            const TensorType& valueType = network.values.at(layer.inputs.at(2)).type;
            const std::size_t tokenCount = queryType.shape.at(0);
            const std::size_t keyWidth = keyType.shape.at(1);
            const std::size_t valueWidth = valueType.shape.at(1);
            if (layer.causal) {
                AttentionKvCache cache(tokenCount, keyWidth, valueWidth, layer.heads);
                for (std::size_t queryToken = 0; queryToken < tokenCount; ++queryToken) {
                    const std::size_t rowOffset = queryToken * keyWidth;
                    const std::size_t valueOffset = queryToken * valueWidth;
                    cache.append(
                        std::span<const double>(keys).subspan(rowOffset, keyWidth),
                        std::span<const double>(projectedValues).subspan(valueOffset, valueWidth));
                    const std::vector<double> rowOutput = cache.attend(
                        std::span<const double>(source).subspan(rowOffset, keyWidth));
                    std::copy(rowOutput.begin(), rowOutput.end(),
                              result.begin() + static_cast<std::ptrdiff_t>(valueOffset));
                }
                continue;
            }
            const std::size_t headWidth = keyWidth / layer.heads;
            const std::size_t valueHeadWidth = valueWidth / layer.heads;
            const double scale = 1.0 / std::sqrt(static_cast<double>(headWidth));
            std::vector<double> scores(tokenCount);
            for (std::size_t queryToken = 0; queryToken < tokenCount; ++queryToken) {
                for (std::size_t head = 0; head < layer.heads; ++head) {
                    double maximum = -std::numeric_limits<double>::infinity();
                    for (std::size_t keyToken = 0; keyToken < tokenCount; ++keyToken) {
                        if (layer.causal && keyToken > queryToken) {
                            scores[keyToken] = -std::numeric_limits<double>::infinity();
                            continue;
                        }
                        double dot = 0.0;
                        for (std::size_t feature = 0; feature < headWidth; ++feature) {
                            const std::size_t queryIndex = queryToken * keyWidth
                                + head * headWidth + feature;
                            const std::size_t keyIndex = keyToken * keyWidth
                                + head * headWidth + feature;
                            dot += source.at(queryIndex) * keys.at(keyIndex);
                        }
                        scores[keyToken] = dot * scale;
                        maximum = std::max(maximum, scores[keyToken]);
                    }
                    double exponentialSum = 0.0;
                    for (std::size_t keyToken = 0; keyToken < tokenCount; ++keyToken) {
                        scores[keyToken] = std::exp(scores[keyToken] - maximum);
                        exponentialSum += scores[keyToken];
                    }
                    for (std::size_t feature = 0; feature < valueHeadWidth; ++feature) {
                        double weightedSum = 0.0;
                        for (std::size_t keyToken = 0; keyToken < tokenCount; ++keyToken) {
                            const double probability = scores[keyToken] / exponentialSum;
                            const std::size_t valueIndex = keyToken * valueWidth
                                + head * valueHeadWidth + feature;
                            weightedSum += probability * projectedValues.at(valueIndex);
                        }
                        const std::size_t outputIndex = queryToken * valueWidth
                            + head * valueHeadWidth + feature;
                        result[outputIndex] = weightedSum;
                    }
                }
            }
        } else {
            const std::vector<double>& residual = values.at(layer.inputs.at(1));
            for (std::size_t i = 0; i < outputCount; ++i) {
                result[i] = source.at(i) + residual.at(i);
            }
        }
    }
    return values.at(network.output);
}

std::vector<double> evaluate(const Network& network, std::vector<double> input,
                             MathMode uniformMode) {
    return evaluate(network, std::move(input),
                    PrecisionPlan{std::vector<MathMode>(denseLayerCount(network), uniformMode)});
}

void reportTensorStorage(const Network& network) {
    std::size_t parameterCount = 0;
    for (const Layer& layer : network.layers) {
        parameterCount += layer.weights.size() + layer.biases.size();
    }
    std::size_t activationCount = 0;
    for (const IRValue& value : network.values) {
        activationCount += elementCount(value.type);
    }
    const std::size_t f64Bytes = (parameterCount + activationCount) * sizeof(double);
    const std::size_t bf16Bytes = (parameterCount + activationCount) * sizeof(std::uint16_t);
    std::cout << "tensor_payload_bytes: f64=" << f64Bytes
              << " bf16=" << bf16Bytes
              << " ratio=" << static_cast<double>(f64Bytes) / bf16Bytes
              << " (parameters plus graph values; excludes executable/runtime overhead)\n";
}

double parseInput(const std::string& token) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid input value '" + token + "'");
    }
    if (consumed != token.size() || !std::isfinite(value)) {
        throw std::runtime_error("input value must be finite: '" + token + "'");
    }
    return value;
}

std::vector<std::vector<double>> readCalibration(const std::string& path,
                                                  std::size_t inputWidth) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open calibration file: " + path);
    }

    std::vector<std::vector<double>> samples;
    std::string rawLine;
    std::size_t lineNumber = 0;
    while (std::getline(file, rawLine)) {
        ++lineNumber;
        const std::size_t comment = rawLine.find('#');
        if (comment != std::string::npos) {
            rawLine.erase(comment);
        }
        std::istringstream line(rawLine);
        std::vector<double> sample;
        std::string token;
        while (line >> token) {
            try {
                sample.push_back(parseInput(token));
            } catch (const std::exception& error) {
                throw std::runtime_error("calibration line " + std::to_string(lineNumber)
                    + ": " + error.what());
            }
        }
        if (sample.empty()) {
            continue;
        }
        if (sample.size() != inputWidth) {
            throw std::runtime_error("calibration line " + std::to_string(lineNumber)
                + " has " + std::to_string(sample.size()) + " values; expected "
                + std::to_string(inputWidth));
        }
        samples.push_back(std::move(sample));
    }
    if (samples.empty()) {
        throw std::runtime_error("calibration file contains no input vectors");
    }
    return samples;
}

std::size_t parameterPayloadBytes(const Network& network,
                                  const PrecisionPlan& plan) {
    if (plan.denseModes.size() != denseLayerCount(network)) {
        throw std::runtime_error("precision plan does not match the number of projections");
    }
    std::size_t bytes = 0;
    std::size_t denseIndex = 0;
    for (const Layer& layer : network.layers) {
        if (layer.kind != LayerKind::Dense && layer.kind != LayerKind::Linear) {
            if (layer.kind == LayerKind::RmsNorm) {
                bytes += layer.weights.size() * sizeof(double);
            }
            continue;
        }
        const std::size_t elementBytes = plan.denseModes.at(denseIndex++)
                == MathMode::Float64
            ? sizeof(double) : sizeof(std::uint16_t);
        bytes += (layer.weights.size() + layer.biases.size()) * elementBytes;
    }
    return bytes;
}

double maximumCalibrationError(const Network& network,
                               const std::vector<std::vector<double>>& samples,
                               const PrecisionPlan& plan) {
    double maximumError = 0.0;
    for (const std::vector<double>& sample : samples) {
        const std::vector<double> reference = evaluate(network, sample,
            PrecisionPlan{std::vector<MathMode>(denseLayerCount(network), MathMode::Float64)});
        const std::vector<double> candidate = evaluate(network, sample, plan);
        for (std::size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
                return std::numeric_limits<double>::infinity();
            }
            maximumError = std::max(maximumError,
                std::abs(reference[i] - candidate[i]));
        }
    }
    return maximumError;
}

PlanSearchResult findPlan(const Network& network,
                          const std::vector<std::vector<double>>& samples,
                          double absoluteErrorLimit) {
    if (!std::isfinite(absoluteErrorLimit) || absoluteErrorLimit < 0.0) {
        throw std::runtime_error("absolute error limit must be a finite nonnegative number");
    }
    const std::size_t denseCount = denseLayerCount(network);
    if (denseCount > 16) {
        throw std::runtime_error("exhaustive plan search supports at most 16 projections");
    }

    const std::size_t planCount = std::size_t{1} << denseCount;
    bool found = false;
    PlanSearchResult best{{}, 0.0, 0, planCount};
    for (std::size_t mask = 0; mask < planCount; ++mask) {
        PrecisionPlan candidate;
        candidate.denseModes.reserve(denseCount);
        for (std::size_t denseIndex = 0; denseIndex < denseCount; ++denseIndex) {
            candidate.denseModes.push_back((mask & (std::size_t{1} << denseIndex)) != 0
                ? MathMode::BFloat16 : MathMode::Float64);
        }
        const double error = maximumCalibrationError(network, samples, candidate);
        if (error > absoluteErrorLimit) {
            continue;
        }
        const std::size_t bytes = parameterPayloadBytes(network, candidate);
        if (!found || bytes < best.parameterBytes
            || (bytes == best.parameterBytes && error < best.maximumAbsoluteError)) {
            best = PlanSearchResult{std::move(candidate), error, bytes, planCount};
            found = true;
        }
    }
    if (!found) {
        throw std::runtime_error("no precision plan meets the calibration error limit");
    }
    return best;
}

void writePlan(const std::string& path, const PlanSearchResult& result,
               double absoluteErrorLimit) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not create plan file: " + path);
    }
    output << "# Proofline precision plan v1\n"
           << "# calibration_max_absolute_error " << std::setprecision(17)
           << result.maximumAbsoluteError << "\n"
           << "# requested_absolute_error_limit " << absoluteErrorLimit << "\n"
           << "# parameter_payload_bytes " << result.parameterBytes << "\n";
    for (MathMode mode : result.plan.denseModes) {
        output << "projection " << mathModeName(mode) << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed while writing plan file: " + path);
    }
}

PrecisionPlan readPlan(const Network& network, const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open precision plan: " + path);
    }
    PrecisionPlan plan;
    std::string rawLine;
    std::size_t lineNumber = 0;
    while (std::getline(file, rawLine)) {
        ++lineNumber;
        const std::size_t comment = rawLine.find('#');
        if (comment != std::string::npos) {
            rawLine.erase(comment);
        }
        std::istringstream line(rawLine);
        std::string command;
        if (!(line >> command)) {
            continue;
        }
        std::string modeName;
        if ((command != "dense" && command != "projection") || !(line >> modeName)) {
            throw std::runtime_error("plan line " + std::to_string(lineNumber)
                + ": expected 'projection f64' or 'projection bf16'");
        }
        rejectExtraTokens(line, lineNumber);
        if (modeName == "f64") {
            plan.denseModes.push_back(MathMode::Float64);
        } else if (modeName == "bf16") {
            plan.denseModes.push_back(MathMode::BFloat16);
        } else {
            throw std::runtime_error("plan line " + std::to_string(lineNumber)
                + ": unsupported precision '" + modeName + "'");
        }
    }
    if (plan.denseModes.size() != denseLayerCount(network)) {
        throw std::runtime_error("plan contains " + std::to_string(plan.denseModes.size())
            + " projection assignments; model requires "
            + std::to_string(denseLayerCount(network)));
    }
    return plan;
}

} // namespace proofline
