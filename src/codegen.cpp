#include "codegen.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace proofline {

void writeValues(std::ostream& output, const std::vector<double>& values) {
    output << std::setprecision(17);
    for (std::size_t i = 0; i < values.size(); ++i) {
        output << (i == 0 ? "" : ", ") << values[i];
    }
}

void writeBFloat16Values(std::ostream& output, const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        output << (i == 0 ? "" : ", ") << encodeBFloat16(values[i]);
    }
}

void writeCpp(const Network& network, const std::string& path,
              const PrecisionPlan& plan) {
    if (plan.denseModes.size() != denseLayerCount(network)) {
        throw std::runtime_error("precision plan does not match the number of projections");
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not create generated source: " + path);
    }

    const std::size_t inputWidth = elementCount(network.values.at(network.input).type);
    std::vector<MathMode> valueModes(network.values.size(), MathMode::Float64);
    std::vector<MathMode> operationModes(network.layers.size(), MathMode::Float64);
    std::size_t denseIndex = 0;
    for (std::size_t i = 0; i < network.layers.size(); ++i) {
        const Layer& layer = network.layers[i];
        if (layer.kind == LayerKind::Dense || layer.kind == LayerKind::Linear) {
            operationModes[i] = plan.denseModes.at(denseIndex++);
            valueModes.at(layer.output) = operationModes[i];
        } else if (layer.kind == LayerKind::RmsNorm) {
            valueModes.at(layer.output) = MathMode::Float64;
        } else if (layer.kind == LayerKind::Softmax) {
            valueModes.at(layer.output) = MathMode::Float64;
        } else if (layer.kind == LayerKind::Rope) {
            valueModes.at(layer.output) = MathMode::Float64;
        } else if (layer.kind == LayerKind::Attention) {
            valueModes.at(layer.output) = MathMode::Float64;
        } else if (layer.kind == LayerKind::Add) {
            valueModes.at(layer.output) = MathMode::Float64;
        } else {
            valueModes.at(layer.output) = valueModes.at(layer.inputs.at(0));
        }
    }
    output << "#include <algorithm>\n"
              "#include <array>\n"
              "#include <bit>\n"
              "#include <chrono>\n"
              "#include <cmath>\n"
              "#include <cstddef>\n"
              "#include <cstdint>\n"
              "#include <iomanip>\n"
              "#include <iostream>\n"
              "#include <limits>\n"
              "#include <stdexcept>\n"
              "#include <string>\n"
              "#include <vector>\n\n"
              "inline std::uint16_t encodeBfloat16(float value) noexcept {\n"
              "    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);\n"
              "    bits += 0x7fffU + ((bits >> 16U) & 1U);\n"
              "    return static_cast<std::uint16_t>(bits >> 16U);\n"
              "}\n"
              "inline float decodeBfloat16(std::uint16_t bits) noexcept {\n"
              "    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);\n"
              "}\n"
              "std::uint16_t checkedEncodeBfloat16(double value) {\n"
              "    const float asFloat = static_cast<float>(value);\n"
              "    if (!std::isfinite(asFloat)) {\n"
              "        throw std::runtime_error(\"value is outside the finite float32 range\");\n"
              "    }\n"
              "    const std::uint16_t bits = encodeBfloat16(asFloat);\n"
              "    if (!std::isfinite(decodeBfloat16(bits))) {\n"
              "        throw std::runtime_error(\"bfloat16 rounding overflowed to infinity\");\n"
              "    }\n"
              "    return bits;\n"
              "}\n\n"
              ;

    for (std::size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
        const Layer& layer = network.layers[layerIndex];
        if (layer.kind != LayerKind::Attention || !layer.causal) {
            continue;
        }
        const TensorType& queryType = network.values.at(layer.inputs.at(0)).type;
        const TensorType& keyType = network.values.at(layer.inputs.at(1)).type;
        const TensorType& valueType = network.values.at(layer.inputs.at(2)).type;
        const std::size_t capacity = queryType.shape.at(0);
        const std::size_t queryWidth = queryType.shape.at(1);
        const std::size_t keyWidth = keyType.shape.at(1);
        const std::size_t valueWidth = valueType.shape.at(1);
        const std::size_t headWidth = keyWidth / layer.heads;
        const std::size_t valueHeadWidth = valueWidth / layer.heads;
        const double scale = 1.0 / std::sqrt(static_cast<double>(headWidth));
        std::array<const Layer*, 3> projectionLayers{};
        std::array<std::size_t, 3> projectionIndexes{};
        const std::array<ValueId, 3> projectedValues{
            layer.inputs.at(0), layer.inputs.at(1), layer.inputs.at(2)};
        bool canProjectInputRow = queryType.shape.size() == 2
            && network.values.at(network.input).type.shape.size() == 2;
        for (std::size_t projection = 0; projection < projectedValues.size(); ++projection) {
            for (std::size_t candidate = 0; candidate < network.layers.size(); ++candidate) {
                if (network.layers[candidate].output == projectedValues[projection]) {
                    projectionLayers[projection] = &network.layers[candidate];
                    projectionIndexes[projection] = candidate;
                    break;
                }
            }
            if (projectionLayers[projection] == nullptr
                || projectionLayers[projection]->kind != LayerKind::Linear
                || projectionLayers[projection]->inputs.at(0) != network.input) {
                canProjectInputRow = false;
            }
        }
        const std::size_t inputRowWidth = network.values.at(network.input).type.shape.size() == 2
            ? network.values.at(network.input).type.shape.at(1) : 0;

        output << "// Stateful causal attention for layer " << layerIndex
               << "; each step appends K/V and attends to the available prefix.\n"
               << "class AttentionSession_" << layerIndex << " {\n"
               << "public:\n"
               << "    using Query = std::array<double, " << queryWidth << ">;\n"
               << "    using Key = std::array<double, " << keyWidth << ">;\n"
               << "    using Value = std::array<double, " << valueWidth << ">;\n"
               << "    using Output = std::array<double, " << valueWidth << ">;\n\n"
               ;
        if (canProjectInputRow) {
            output << "    using InputRow = std::array<double, " << inputRowWidth << ">;\n\n";
        }
        output
               << "    AttentionSession_" << layerIndex << "() {\n"
               << "        key_cache_.reserve(" << capacity * keyWidth << ");\n"
               << "        value_cache_.reserve(" << capacity * valueWidth << ");\n"
               << "    }\n\n"
               << "    Output step(const Query& query, const Key& key, const Value& value) {\n"
               << "        if (cached_tokens_ == " << capacity << ") {\n"
               << "            throw std::length_error(\"attention session capacity exceeded\");\n"
               << "        }\n"
               << "        const auto finite = [](double item) { return std::isfinite(item); };\n"
               << "        if (!std::all_of(query.begin(), query.end(), finite)\n"
               << "            || !std::all_of(key.begin(), key.end(), finite)\n"
               << "            || !std::all_of(value.begin(), value.end(), finite)) {\n"
               << "            throw std::invalid_argument(\"attention session rows must be finite\");\n"
               << "        }\n"
               << "        key_cache_.insert(key_cache_.end(), key.begin(), key.end());\n"
               << "        value_cache_.insert(value_cache_.end(), value.begin(), value.end());\n"
               << "        ++cached_tokens_;\n"
               << "        Output output{};\n"
               << "        std::array<double, " << capacity << "> scores{};\n"
               << "        for (std::size_t head = 0; head < " << layer.heads << "; ++head) {\n"
               << "            double maximum = -std::numeric_limits<double>::infinity();\n"
               << "            for (std::size_t token = 0; token < cached_tokens_; ++token) {\n"
               << "                double dot = 0.0;\n"
               << "                for (std::size_t feature = 0; feature < " << headWidth << "; ++feature) {\n"
               << "                    const std::size_t offset = head * " << headWidth << " + feature;\n"
               << "                    dot += query[offset] * key_cache_[token * " << keyWidth << " + offset];\n"
               << "                }\n"
               << "                scores[token] = dot * " << std::setprecision(17) << scale << ";\n"
               << "                maximum = std::max(maximum, scores[token]);\n"
               << "            }\n"
               << "            double exponential_sum = 0.0;\n"
               << "            for (std::size_t token = 0; token < cached_tokens_; ++token) {\n"
               << "                scores[token] = std::exp(scores[token] - maximum);\n"
               << "                exponential_sum += scores[token];\n"
               << "            }\n"
               << "            for (std::size_t feature = 0; feature < " << valueHeadWidth << "; ++feature) {\n"
               << "                double weighted_sum = 0.0;\n"
               << "                for (std::size_t token = 0; token < cached_tokens_; ++token) {\n"
               << "                    const std::size_t offset = head * " << valueHeadWidth << " + feature;\n"
               << "                    weighted_sum += (scores[token] / exponential_sum)\n"
               << "                        * value_cache_[token * " << valueWidth << " + offset];\n"
               << "                }\n"
               << "                output[head * " << valueHeadWidth << " + feature] = weighted_sum;\n"
               << "            }\n"
               << "        }\n"
               << "        return output;\n"
               << "    }\n\n";

        if (canProjectInputRow) {
            output << "    Output stepFromInput(const InputRow& token) {\n"
                   << "        Query query{};\n"
                   << "        Key key{};\n"
                   << "        Value value{};\n";
            const std::array<std::string, 3> projectionNames{"query", "key", "value"};
            for (std::size_t projection = 0; projection < projectionLayers.size(); ++projection) {
                const Layer& projectionLayer = *projectionLayers[projection];
                const std::size_t projectionIndex = projectionIndexes[projection];
                const MathMode mode = operationModes[projectionIndex];
                const std::string elementType = mode == MathMode::Float64
                    ? "double" : "std::uint16_t";
                const std::string suffix = projectionNames[projection];
                output << "        static constexpr std::array<" << elementType << ", "
                       << projectionLayer.weights.size() << "> " << suffix << "_weights{";
                if (mode == MathMode::Float64) {
                    writeValues(output, projectionLayer.weights);
                } else {
                    writeBFloat16Values(output, projectionLayer.weights);
                }
                output << "};\n"
                       << "        static constexpr std::array<" << elementType << ", "
                       << projectionLayer.biases.size() << "> " << suffix << "_biases{";
                if (mode == MathMode::Float64) {
                    writeValues(output, projectionLayer.biases);
                } else {
                    writeBFloat16Values(output, projectionLayer.biases);
                }
                output << "};\n"
                       << "        for (std::size_t row = 0; row < "
                       << projectionLayer.biases.size() << "; ++row) {\n";
                const std::string target = suffix == "query" ? "query" : suffix;
                if (mode == MathMode::Float64) {
                    output << "            double sum = " << suffix << "_biases[row];\n"
                           << "            for (std::size_t column = 0; column < " << inputRowWidth << "; ++column) {\n"
                           << "                sum += " << suffix << "_weights[row * " << inputRowWidth
                           << " + column] * token[column];\n"
                           << "            }\n"
                           << "            " << target << "[row] = sum;\n";
                } else {
                    output << "            float sum = decodeBfloat16(" << suffix << "_biases[row]);\n"
                           << "            for (std::size_t column = 0; column < " << inputRowWidth << "; ++column) {\n"
                           << "                const float weight = decodeBfloat16(" << suffix << "_weights[row * "
                           << inputRowWidth << " + column]);\n"
                           << "                const float activation = decodeBfloat16(checkedEncodeBfloat16(token[column]));\n"
                           << "                sum = sum + weight * activation;\n"
                           << "            }\n"
                           << "            " << target << "[row] = decodeBfloat16(encodeBfloat16(sum));\n";
                }
                output << "        }\n";
            }
            output << "        return step(query, key, value);\n"
                   << "    }\n\n";
        }

        output << "    void reset() noexcept {\n"
               << "        cached_tokens_ = 0;\n"
               << "        key_cache_.clear();\n"
               << "        value_cache_.clear();\n"
               << "    }\n"
               << "    std::size_t size() const noexcept { return cached_tokens_; }\n"
               << "    static constexpr std::size_t capacity() noexcept { return " << capacity << "; }\n"
               << "\nprivate:\n"
               << "    std::size_t cached_tokens_ = 0;\n"
               << "    std::vector<double> key_cache_;\n"
               << "    std::vector<double> value_cache_;\n"
               << "};\n\n";
    }

    output << "#ifndef PROOFLINE_LIBRARY\n"
              "int main(int argc, char** argv) {\n"
           << "    constexpr std::size_t input_width = " << inputWidth << ";\n"
              "    bool benchmark = false;\n"
              "    std::size_t iterations = 1;\n"
              "    std::size_t first_input_argument = 1;\n"
              "    if (argc > 1 && std::string(argv[1]) == \"--bench\") {\n"
              "        if (argc < 3) {\n"
              "            std::cerr << \"usage: program --bench iterations inputs...\\n\";\n"
              "            return 2;\n"
              "        }\n"
              "        try {\n"
              "            std::size_t consumed = 0;\n"
              "            iterations = std::stoull(argv[2], &consumed);\n"
              "            if (consumed != std::string(argv[2]).size() || iterations == 0) {\n"
              "                throw std::runtime_error(\"iterations must be a positive integer\");\n"
              "            }\n"
              "        } catch (const std::exception& error) {\n"
              "            std::cerr << error.what() << '\\n';\n"
              "            return 2;\n"
              "        }\n"
              "        benchmark = true;\n"
              "        first_input_argument = 3;\n"
              "    }\n"
           << "    if (argc != static_cast<int>(first_input_argument + input_width)) {\n"
              "        std::cerr << \"expected \" << input_width << \" input values\\n\";\n"
              "        return 2;\n"
              "    }\n"
           << "    std::array<double, " << inputWidth << "> "
           << network.values.at(network.input).name << "{};\n"
              "    try {\n"
              "        for (std::size_t i = 0; i < input_width; ++i) {\n"
              "            std::size_t consumed = 0;\n"
              "            const std::string token = argv[first_input_argument + i];\n"
              "            "
              "            const double parsed = std::stod(token, &consumed);\n"
              "            if (consumed != token.size() || !std::isfinite(parsed)) {\n"
              "                throw std::runtime_error(\"input must be a finite number\");\n"
              "            }\n"
           << "            " << network.values.at(network.input).name << "[i] = parsed;\n";
    output << "        }\n"
              "    } catch (const std::exception& error) {\n"
              "        std::cerr << error.what() << '\\n';\n"
              "        return 1;\n"
              "    }\n"
           << "    const std::array<double, " << inputWidth << "> original_input = "
           << network.values.at(network.input).name << ";\n";

    for (std::size_t valueId = 0; valueId < network.values.size(); ++valueId) {
        if (valueId == network.input) {
            continue;
        }
        const IRValue& value = network.values[valueId];
        const std::string storageType = valueModes[valueId] == MathMode::Float64
            ? "double" : "std::uint16_t";
        output << "    std::array<" << storageType << ", "
               << elementCount(value.type) << "> " << value.name << "{};\n";
    }

    for (std::size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
        const Layer& layer = network.layers[layerIndex];
        if (layer.kind == LayerKind::Attention) {
            const std::size_t tokenCount = network.values.at(layer.inputs.at(0)).type.shape.at(0);
            output << "    std::array<double, " << tokenCount << "> attention_scores_"
                   << layerIndex << "{};\n";
        }
    }

    for (std::size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
        const Layer& layer = network.layers[layerIndex];
        if ((layer.kind == LayerKind::Dense || layer.kind == LayerKind::Linear)
            && operationModes[layerIndex] == MathMode::BFloat16
            && valueModes.at(layer.inputs.at(0)) == MathMode::Float64) {
            output << "    std::array<std::uint16_t, "
                   << elementCount(network.values.at(layer.inputs.at(0)).type)
                   << "> quantized_input_" << layerIndex << "{};\n";
        }
    }

    for (std::size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
        const Layer& layer = network.layers[layerIndex];
        const MathMode mode = operationModes[layerIndex];
        if (layer.kind == LayerKind::Dense || layer.kind == LayerKind::Linear) {
            const std::string storageType = mode == MathMode::Float64
                ? "double" : "std::uint16_t";
            output << "    const std::array<" << storageType << ", " << layer.weights.size()
                   << "> weights_"
                   << layerIndex << "{";
            if (mode == MathMode::Float64) {
                writeValues(output, layer.weights);
            } else {
                writeBFloat16Values(output, layer.weights);
            }
            output << "};\n"
                   << "    const std::array<" << storageType << ", " << layer.biases.size()
                   << "> biases_"
                   << layerIndex << "{";
            if (mode == MathMode::Float64) {
                writeValues(output, layer.biases);
            } else {
                writeBFloat16Values(output, layer.biases);
            }
            output << "};\n";
        } else if (layer.kind == LayerKind::RmsNorm) {
            output << "    const std::array<double, " << layer.weights.size()
                   << "> scale_" << layerIndex << "{";
            writeValues(output, layer.weights);
            output << "};\n";
        }
    }

    output << "    volatile double benchmark_checksum = 0.0;\n"
              "    const auto benchmark_start = std::chrono::steady_clock::now();\n"
              "    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {\n"
              "        " << network.values.at(network.input).name << " = original_input;\n"
              "        if (benchmark) {\n"
              "            " << network.values.at(network.input).name
           << "[0] += static_cast<double>(iteration % 17) * 1.0e-12;\n"
              "        }\n";

    for (std::size_t layerIndex = 0; layerIndex < network.layers.size(); ++layerIndex) {
        const Layer& layer = network.layers[layerIndex];
        const IRValue& input = network.values.at(layer.inputs.at(0));
        const IRValue& result = network.values.at(layer.output);
        const std::size_t outputWidth = widthOf(result.type);
        const std::size_t outputCount = elementCount(result.type);
        const std::size_t layerInputWidth = widthOf(input.type);
        const std::size_t inputCount = elementCount(input.type);
        const std::size_t rows = input.type.shape.size() == 2 ? input.type.shape[0] : 1;
        const MathMode mode = operationModes[layerIndex];
        const MathMode inputMode = valueModes.at(layer.inputs.at(0));
        const MathMode outputMode = valueModes.at(layer.output);

        if (layer.kind == LayerKind::Dense || layer.kind == LayerKind::Linear) {
            if (mode == MathMode::BFloat16 && inputMode == MathMode::Float64) {
                output << "        for (std::size_t i = 0; i < " << inputCount << "; ++i) {\n"
                       << "            quantized_input_" << layerIndex << "[i] = checkedEncodeBfloat16("
                       << input.name << "[i]);\n"
                          "        }\n";
            }
            output << "        for (std::size_t token = 0; token < " << rows << "; ++token) {\n"
                   << "            for (std::size_t row = 0; row < " << outputWidth << "; ++row) {\n";
            if (mode == MathMode::Float64) {
                const std::string activation = inputMode == MathMode::BFloat16
                    ? "static_cast<double>(decodeBfloat16(" + input.name + "[token * "
                        + std::to_string(layerInputWidth) + " + column]))"
                    : input.name + "[token * " + std::to_string(layerInputWidth) + " + column]";
                output << "                double sum = biases_" << layerIndex << "[row];\n"
                       << "                for (std::size_t column = 0; column < "
                       << layerInputWidth << "; ++column) {\n"
                       << "                    sum += weights_" << layerIndex << "[row * "
                       << layerInputWidth << " + column] * " << activation << ";\n"
                          "                }\n"
                       << "                " << result.name << "[token * " << outputWidth
                       << " + row] = sum;\n";
            } else {
                output << "#if defined(__clang__)\n"
                          "#pragma clang fp reassociate(on)\n"
                          "#endif\n";
                const std::string activation = inputMode == MathMode::BFloat16
                    ? "decodeBfloat16(" + input.name + "[token * "
                        + std::to_string(layerInputWidth) + " + column])"
                    : "decodeBfloat16(quantized_input_" + std::to_string(layerIndex)
                        + "[token * " + std::to_string(layerInputWidth) + " + column])";
                output << "                float sum = decodeBfloat16(biases_"
                       << layerIndex << "[row]);\n"
                       << "                for (std::size_t column = 0; column < "
                       << layerInputWidth << "; ++column) {\n"
                       << "                    const float weight = decodeBfloat16(weights_"
                       << layerIndex << "[row * " << layerInputWidth << " + column]);\n"
                       << "                    const float activation = " << activation << ";\n"
                          "                    const float product = weight * activation;\n"
                          "                    sum = sum + product;\n"
                          "                }\n"
                       << "                " << result.name << "[token * " << outputWidth
                       << " + row] = encodeBfloat16(sum);\n";
            }
            output << "            }\n        }\n";
            if (mode == MathMode::BFloat16) {
                output << "        for (std::size_t i = 0; i < " << outputCount << "; ++i) {\n"
                       << "            if (!std::isfinite(decodeBfloat16(" << result.name << "[i]))) {\n"
                          "                throw std::runtime_error(\"bfloat16 layer output is not finite\");\n"
                          "            }\n"
                          "        }\n";
            }
        } else if (layer.kind == LayerKind::Relu) {
            output << "        for (std::size_t i = 0; i < " << outputCount << "; ++i) {\n"
                   << "            " << result.name << "[i] = ";
            if (outputMode == MathMode::BFloat16) {
                output << "encodeBfloat16(static_cast<float>(std::max(0.0, "
                       << "static_cast<double>(decodeBfloat16(" << input.name << "[i])))))";
            } else {
                output << "std::max(0.0, " << input.name << "[i])";
            }
            output << ";\n"
                      "        }\n";
        } else if (layer.kind == LayerKind::RmsNorm) {
            const std::string activationAtOffset = inputMode == MathMode::BFloat16
                ? "static_cast<double>(decodeBfloat16(" + input.name + "[offset + i]))"
                : input.name + "[offset + i]";
            output << "        for (std::size_t token = 0; token < " << rows << "; ++token) {\n"
                   << "            const std::size_t offset = token * " << layerInputWidth << ";\n"
                   << "            double normalization_scale = std::sqrt("
                   << std::setprecision(17) << layer.epsilon << ");\n"
                   << "            for (std::size_t i = 0; i < " << layerInputWidth << "; ++i) {\n"
                   << "                const double value = " << activationAtOffset << ";\n"
                   << "                normalization_scale = std::max(normalization_scale, std::abs(value));\n"
                   << "            }\n"
                   << "            double scaled_square_sum = 0.0;\n"
                   << "            for (std::size_t i = 0; i < " << layerInputWidth << "; ++i) {\n"
                   << "                const double value = " << activationAtOffset << ";\n"
                   << "                const double scaled = value / normalization_scale;\n"
                   << "                scaled_square_sum += scaled * scaled;\n"
                   << "            }\n"
                   << "            const double epsilon_ratio = std::sqrt("
                   << std::setprecision(17) << layer.epsilon
                   << ") / normalization_scale;\n"
                   << "            const double rms = normalization_scale * std::sqrt("
                   << "scaled_square_sum / " << layerInputWidth
                   << " + epsilon_ratio * epsilon_ratio);\n"
                   << "            for (std::size_t i = 0; i < " << outputWidth << "; ++i) {\n"
                   << "                const double value = " << activationAtOffset << ";\n"
                   << "                " << result.name
                   << "[offset + i] = (value / rms) * scale_" << layerIndex << "[i];\n"
                   << "            }\n"
                   << "        }\n";
        } else if (layer.kind == LayerKind::Softmax) {
            output << "        double maximum_" << layerIndex << " = ";
            if (inputMode == MathMode::BFloat16) {
                output << "static_cast<double>(decodeBfloat16(" << input.name << "[0]))";
            } else {
                output << input.name << "[0]";
            }
            output << ";\n"
                   << "        for (std::size_t i = 1; i < " << layerInputWidth << "; ++i) {\n"
                   << "            const double value = ";
            if (inputMode == MathMode::BFloat16) {
                output << "static_cast<double>(decodeBfloat16(" << input.name << "[i]))";
            } else {
                output << input.name << "[i]";
            }
            output << ";\n"
                   << "            maximum_" << layerIndex
                   << " = std::max(maximum_" << layerIndex << ", value);\n"
                      "        }\n"
                   << "        double exponential_sum_" << layerIndex << " = 0.0;\n"
                   << "        for (std::size_t i = 0; i < " << outputWidth << "; ++i) {\n"
                   << "            const double value = ";
            if (inputMode == MathMode::BFloat16) {
                output << "static_cast<double>(decodeBfloat16(" << input.name << "[i]))";
            } else {
                output << input.name << "[i]";
            }
            output << ";\n"
                   << "            " << result.name << "[i] = std::exp(value - maximum_"
                   << layerIndex << ");\n"
                   << "            exponential_sum_" << layerIndex << " += "
                   << result.name << "[i];\n"
                      "        }\n"
                   << "        for (std::size_t i = 0; i < " << outputWidth << "; ++i) {\n"
                   << "            " << result.name << "[i] /= exponential_sum_"
                   << layerIndex << ";\n"
                      "        }\n";
        } else if (layer.kind == LayerKind::Rope) {
            const std::string evenValue = inputMode == MathMode::BFloat16
                ? "static_cast<double>(decodeBfloat16(" + input.name
                    + "[offset + 2 * pair]))"
                : input.name + "[offset + 2 * pair]";
            const std::string oddValue = inputMode == MathMode::BFloat16
                ? "static_cast<double>(decodeBfloat16(" + input.name
                    + "[offset + 2 * pair + 1]))"
                : input.name + "[offset + 2 * pair + 1]";
            output << "        for (std::size_t token = 0; token < " << rows << "; ++token) {\n"
                   << "            const std::size_t offset = token * " << layerInputWidth << ";\n"
                   << "            const double position = static_cast<double>("
                   << layer.position << " + token);\n"
                   << "            for (std::size_t pair = 0; pair < " << layerInputWidth / 2
                   << "; ++pair) {\n"
                   << "                const double exponent = -2.0 * static_cast<double>(pair) / "
                   << layerInputWidth << ";\n"
                   << "                const double inverse_frequency = std::pow("
                   << std::setprecision(17) << layer.ropeBase << ", exponent);\n"
                   << "                const double angle = position * inverse_frequency;\n"
                   << "                const double cosine = std::cos(angle);\n"
                   << "                const double sine = std::sin(angle);\n"
                   << "                const double even = " << evenValue << ";\n"
                   << "                const double odd = " << oddValue << ";\n"
                   << "                " << result.name
                   << "[offset + 2 * pair] = even * cosine - odd * sine;\n"
                   << "                " << result.name
                   << "[offset + 2 * pair + 1] = even * sine + odd * cosine;\n"
                   << "            }\n"
                   << "        }\n";
        } else if (layer.kind == LayerKind::Attention) {
            const IRValue& key = network.values.at(layer.inputs.at(1));
            const IRValue& projectedValue = network.values.at(layer.inputs.at(2));
            const MathMode keyMode = valueModes.at(layer.inputs.at(1));
            const MathMode projectedValueMode = valueModes.at(layer.inputs.at(2));
            const std::size_t tokenCount = input.type.shape.at(0);
            const std::size_t keyWidth = input.type.shape.at(1);
            const std::size_t valueWidth = widthOf(projectedValue.type);
            const std::size_t headWidth = keyWidth / layer.heads;
            const std::size_t valueHeadWidth = valueWidth / layer.heads;
            const double scale = 1.0 / std::sqrt(static_cast<double>(headWidth));
            const auto readValue = [](const std::string& name, MathMode valueMode,
                                      const std::string& index) {
                return valueMode == MathMode::BFloat16
                    ? "static_cast<double>(decodeBfloat16(" + name + "[" + index + "]))"
                    : name + "[" + index + "]";
            };
            output << "        for (std::size_t query_token = 0; query_token < "
                   << tokenCount << "; ++query_token) {\n"
                   << "            for (std::size_t head = 0; head < " << layer.heads
                   << "; ++head) {\n"
                   << "                double maximum_score = -std::numeric_limits<double>::infinity();\n"
                   << "                for (std::size_t key_token = 0; key_token < " << tokenCount
                   << "; ++key_token) {\n";
            if (layer.causal) {
                output << "                    if (key_token > query_token) {\n"
                       << "                        attention_scores_" << layerIndex
                       << "[key_token] = -std::numeric_limits<double>::infinity();\n"
                       << "                        continue;\n"
                       << "                    }\n";
            }
            output << "                    double score = 0.0;\n"
                   << "                    for (std::size_t feature = 0; feature < " << headWidth
                   << "; ++feature) {\n"
                   << "                        score += "
                   << readValue(input.name, inputMode,
                        "query_token * " + std::to_string(keyWidth) + " + head * "
                            + std::to_string(headWidth) + " + feature")
                   << " * "
                   << readValue(key.name, keyMode,
                        "key_token * " + std::to_string(keyWidth) + " + head * "
                            + std::to_string(headWidth) + " + feature")
                   << ";\n"
                   << "                    }\n"
                   << "                    attention_scores_" << layerIndex << "[key_token] = score * "
                   << std::setprecision(17) << scale << ";\n"
                   << "                    maximum_score = std::max(maximum_score, attention_scores_"
                   << layerIndex << "[key_token]);\n"
                   << "                }\n"
                   << "                double exponential_sum = 0.0;\n"
                   << "                for (std::size_t key_token = 0; key_token < " << tokenCount
                   << "; ++key_token) {\n"
                   << "                    attention_scores_" << layerIndex
                   << "[key_token] = std::exp(attention_scores_" << layerIndex
                   << "[key_token] - maximum_score);\n"
                   << "                    exponential_sum += attention_scores_" << layerIndex
                   << "[key_token];\n"
                   << "                }\n"
                   << "                for (std::size_t feature = 0; feature < " << valueHeadWidth
                   << "; ++feature) {\n"
                   << "                    double weighted_sum = 0.0;\n"
                   << "                    for (std::size_t key_token = 0; key_token < " << tokenCount
                   << "; ++key_token) {\n"
                   << "                        const double probability = attention_scores_"
                   << layerIndex << "[key_token] / exponential_sum;\n"
                   << "                        weighted_sum += probability * "
                   << readValue(projectedValue.name, projectedValueMode,
                        "key_token * " + std::to_string(valueWidth) + " + head * "
                            + std::to_string(valueHeadWidth) + " + feature")
                   << ";\n"
                   << "                    }\n"
                   << "                    " << result.name << "[query_token * " << valueWidth
                   << " + head * " << valueHeadWidth << " + feature] = weighted_sum;\n"
                   << "                }\n"
                   << "            }\n"
                   << "        }\n";
        } else {
            const IRValue& residual = network.values.at(layer.inputs.at(1));
            const MathMode residualMode = valueModes.at(layer.inputs.at(1));
            output << "        for (std::size_t i = 0; i < " << outputCount << "; ++i) {\n"
                   << "            " << result.name << "[i] = ";
            if (inputMode == MathMode::BFloat16) {
                output << "static_cast<double>(decodeBfloat16(" << input.name << "[i]))";
            } else {
                output << input.name << "[i]";
            }
            output << " + ";
            if (residualMode == MathMode::BFloat16) {
                output << "static_cast<double>(decodeBfloat16(" << residual.name << "[i]))";
            } else {
                output << residual.name << "[i]";
            }
            output << ";\n"
                      "        }\n";
        }
    }

    const IRValue& finalValue = network.values.at(network.output);
    const std::size_t outputWidth = elementCount(finalValue.type);
    output << "        double final_output_sum = 0.0;\n"
              "        for (std::size_t i = 0; i < " << outputWidth << "; ++i) {\n"
              "            final_output_sum += ";
    if (valueModes.at(network.output) == MathMode::BFloat16) {
        output << "decodeBfloat16(" << finalValue.name << "[i])";
    } else {
        output << finalValue.name << "[i]";
    }
    output << ";\n"
              "        }\n"
              "        const double previous_checksum = benchmark_checksum;\n"
              "        benchmark_checksum = previous_checksum + final_output_sum;\n"
              "    }\n"
              "    const auto benchmark_end = std::chrono::steady_clock::now();\n"
              "    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
              "        benchmark_end - benchmark_start).count();\n"
              "    if (benchmark) {\n"
              "        std::cout << std::setprecision(17)\n"
              "                  << \"iterations=\" << iterations\n"
              "                  << \" elapsed_ns=\" << elapsed_ns\n"
              "                  << \" ns_per_inference=\"\n"
              "                  << static_cast<double>(elapsed_ns) / iterations\n"
              "                  << \" checksum=\" << static_cast<double>(benchmark_checksum) << '\\n';\n"
              "        return 0;\n"
              "    }\n"
              "    std::cout << std::setprecision(17);\n"
              "    for (std::size_t i = 0; i < " << outputWidth << "; ++i) {\n"
              "        std::cout << (i == 0 ? \"\" : \" \") << ";
    if (valueModes.at(network.output) == MathMode::BFloat16) {
        output << "decodeBfloat16(" << finalValue.name << "[i])";
    } else {
        output << finalValue.name << "[i]";
    }
    output << ";\n"
              "    }\n"
              "    std::cout << '\\n';\n"
              "    return 0;\n"
              "}\n"
              "#endif // PROOFLINE_LIBRARY\n";
    if (!output) {
        throw std::runtime_error("failed while writing generated source: " + path);
    }
}

} // namespace proofline
