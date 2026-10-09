#include "certificate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace proofline {

double roundedUp(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("certificate arithmetic overflowed");
    }
    const double result = std::nextafter(value,
        std::numeric_limits<double>::infinity());
    if (!std::isfinite(result)) {
        throw std::runtime_error("certificate bound is too large to represent");
    }
    return result;
}

double roundedDown(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("certificate arithmetic overflowed");
    }
    const double result = std::nextafter(value,
        -std::numeric_limits<double>::infinity());
    if (!std::isfinite(result)) {
        throw std::runtime_error("certificate interval is too large to represent");
    }
    return result;
}

double addUp(double left, double right) {
    if (left == 0.0) return right;
    if (right == 0.0) return left;
    return roundedUp(left + right);
}

double multiplyUp(double left, double right) {
    if (left < 0.0 || right < 0.0) {
        throw std::runtime_error("internal certificate error: expected nonnegative values");
    }
    if (left == 0.0 || right == 0.0) return 0.0;
    if (left == 1.0) return right;
    if (right == 1.0) return left;
    return roundedUp(left * right);
}

double divideUp(double numerator, double denominator) {
    if (numerator < 0.0 || denominator <= 0.0) {
        throw std::runtime_error("internal certificate error: invalid division");
    }
    return roundedUp(numerator / denominator);
}

double maximumMagnitude(const Interval& range) {
    return std::max(std::abs(range.low), std::abs(range.high));
}

Interval multiplyInterval(double coefficient, const Interval& range) {
    const double first = coefficient * range.low;
    const double second = coefficient * range.high;
    return Interval{roundedDown(std::min(first, second)),
                     roundedUp(std::max(first, second))};
}

Interval denseRange(const Layer& layer, const std::vector<Interval>& inputRanges,
                    std::size_t inputWidth, std::size_t row) {
    Interval sum{layer.biases.at(row), layer.biases.at(row)};
    for (std::size_t column = 0; column < inputWidth; ++column) {
        const Interval term = multiplyInterval(
            layer.weights.at(row * inputWidth + column), inputRanges.at(column));
        sum.low = roundedDown(sum.low + term.low);
        sum.high = roundedUp(sum.high + term.high);
    }
    return sum;
}

double gammaBound(std::size_t operationCount, double unitRoundoff,
                  double absoluteSum, double underflowAllowance) {
    const double count = static_cast<double>(operationCount);
    const double ku = multiplyUp(count, unitRoundoff);
    if (ku >= 1.0) {
        throw std::runtime_error("layer is too wide for the selected roundoff bound");
    }
    const double denominator = roundedDown(1.0 - ku);
    const double gamma = divideUp(ku, denominator);
    const double accumulatedUnderflow = divideUp(
        multiplyUp(count, underflowAllowance), denominator);
    return addUp(multiplyUp(gamma, absoluteSum), accumulatedUnderflow);
}

double maximumBfloatRoundingError(double magnitude) {
    if (magnitude == 0.0) return 0.0;
    constexpr double minimumNormal = 0x1p-126;
    constexpr double halfSubnormalStep = 0x1p-134;
    if (magnitude <= minimumNormal) return halfSubnormalStep;

    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent);
    // At an exact power of two, values immediately below it use the previous
    // binade's smaller spacing. Otherwise the current binade sets the maximum.
    const int halfUlpExponent = exponent - (fraction == 0.5 ? 10 : 9);
    return std::ldexp(1.0, halfUlpExponent);
}

double roundedFloatMagnitudeUpper(double magnitude) {
    if (magnitude == 0.0) return 0.0;
    if (magnitude > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::runtime_error("bfloat16 certificate cannot rule out float32 overflow");
    }
    const float rounded = static_cast<float>(magnitude);
    if (!std::isfinite(rounded)) {
        throw std::runtime_error("bfloat16 certificate cannot rule out float32 overflow");
    }
    return static_cast<double>(rounded);
}

double bfloatInputError(double magnitude) {
    // A runtime activation is rounded to float32 before BF16. The float32
    // conversion uses a relative bound; BF16 rounding uses the largest half-ULP
    // in the binades actually reachable by the rounded float32 magnitude.
    if (magnitude == 0.0) return 0.0;
    const double floatError = addUp(
        multiplyUp(std::ldexp(1.0, -24), magnitude), std::ldexp(1.0, -149));
    const double floatMagnitude = roundedFloatMagnitudeUpper(magnitude);
    return addUp(floatError, maximumBfloatRoundingError(floatMagnitude));
}

double checkedBfloatMagnitude(double magnitude) {
    const double floatMagnitude = roundedFloatMagnitudeUpper(magnitude);
    const double quantizedMagnitude = addUp(floatMagnitude,
        maximumBfloatRoundingError(floatMagnitude));
    if (quantizedMagnitude > 3.3895313892515355e38) {
        throw std::runtime_error("bfloat16 certificate cannot rule out a non-finite conversion");
    }
    return quantizedMagnitude;
}

double checkedBfloatConstant(double value) {
    const double encoded = roundToBFloat16(value);
    if (!std::isfinite(encoded)) {
        throw std::runtime_error("bfloat16 certificate cannot encode a model constant finitely");
    }
    return encoded;
}

double absoluteDifferenceUp(double left, double right) {
    return roundedUp(std::abs(left - right));
}

double bfloatOutputError(double magnitude) {
    return maximumBfloatRoundingError(magnitude);
}

Certificate certifyPlan(const Network& network,
                        const std::vector<Interval>& inputDomain,
                        const PrecisionPlan& plan) {
    if (network.values.at(network.input).type.shape.size() != 1) {
        throw std::runtime_error(
            "bounded-domain certification currently supports vector inputs only");
    }
    if (inputDomain.size() != elementCount(network.values.at(network.input).type)) {
        throw std::runtime_error("input domain width does not match the model");
    }
    if (std::any_of(network.layers.begin(), network.layers.end(),
            [](const Layer& layer) { return layer.kind == LayerKind::Linear; })) {
        throw std::runtime_error(
            "bounded-domain certification does not yet support token-wise Linear");
    }
    std::vector<std::vector<Interval>> exactValues(network.values.size());
    std::vector<std::vector<double>> referenceErrors(network.values.size());
    std::vector<std::vector<double>> candidateErrors(network.values.size());
    exactValues.at(network.input) = inputDomain;
    referenceErrors.at(network.input).assign(inputDomain.size(), 0.0);
    candidateErrors.at(network.input).assign(inputDomain.size(), 0.0);

    std::size_t denseIndex = 0;
    constexpr double float64Unit = 0x1p-53;
    constexpr double float32Unit = 0x1p-24;
    constexpr double float64Tiny = 0x1p-1074;
    constexpr double float32Tiny = 0x1p-149;
    constexpr double bfloatMaximum = 3.3895313892515355e38;
    for (const Layer& layer : network.layers) {
        const std::vector<Interval>& sourceRange = exactValues.at(layer.inputs.at(0));
        const std::vector<double>& referenceInputError = referenceErrors.at(layer.inputs.at(0));
        const std::vector<double>& candidateInputError = candidateErrors.at(layer.inputs.at(0));
        const std::size_t inputWidth = sourceRange.size();
        const std::size_t outputWidth = widthOf(network.values.at(layer.output).type);
        std::vector<Interval>& outputRange = exactValues.at(layer.output);
        std::vector<double>& referenceOutputError = referenceErrors.at(layer.output);
        std::vector<double>& candidateOutputError = candidateErrors.at(layer.output);
        outputRange.resize(outputWidth);
        referenceOutputError.resize(outputWidth);
        candidateOutputError.resize(outputWidth);

        if (layer.kind == LayerKind::Relu) {
            for (std::size_t i = 0; i < outputWidth; ++i) {
                outputRange[i] = Interval{std::max(0.0, sourceRange[i].low),
                                          std::max(0.0, sourceRange[i].high)};
                // ReLU is 1-Lipschitz; a BF16 value remains exactly representable.
                referenceOutputError[i] = referenceInputError[i];
                candidateOutputError[i] = candidateInputError[i];
            }
            continue;
        }
        if (layer.kind == LayerKind::RmsNorm) {
            throw std::runtime_error(
                "bounded-domain certification does not yet support RMSNorm");
        }
        if (layer.kind == LayerKind::Softmax) {
            throw std::runtime_error(
                "bounded-domain certification does not yet support Softmax");
        }
        if (layer.kind == LayerKind::Rope) {
            throw std::runtime_error(
                "bounded-domain certification does not yet support RoPE");
        }
        if (layer.kind == LayerKind::Add) {
            const std::vector<Interval>& residualRange =
                exactValues.at(layer.inputs.at(1));
            const std::vector<double>& referenceResidualError =
                referenceErrors.at(layer.inputs.at(1));
            const std::vector<double>& candidateResidualError =
                candidateErrors.at(layer.inputs.at(1));
            for (std::size_t i = 0; i < outputWidth; ++i) {
                outputRange[i] = Interval{
                    roundedDown(sourceRange[i].low + residualRange[i].low),
                    roundedUp(sourceRange[i].high + residualRange[i].high),
                };
                const double referenceMagnitude = addUp(
                    addUp(maximumMagnitude(sourceRange[i]), referenceInputError[i]),
                    addUp(maximumMagnitude(residualRange[i]), referenceResidualError[i]));
                const double candidateMagnitude = addUp(
                    addUp(maximumMagnitude(sourceRange[i]), candidateInputError[i]),
                    addUp(maximumMagnitude(residualRange[i]), candidateResidualError[i]));
                const double referenceRoundoff = gammaBound(1, float64Unit,
                    referenceMagnitude, float64Tiny);
                const double candidateRoundoff = gammaBound(1, float64Unit,
                    candidateMagnitude, float64Tiny);
                referenceOutputError[i] = addUp(
                    addUp(referenceInputError[i], referenceResidualError[i]),
                    referenceRoundoff);
                candidateOutputError[i] = addUp(
                    addUp(candidateInputError[i], candidateResidualError[i]),
                    candidateRoundoff);
            }
            continue;
        }

        const MathMode mode = plan.denseModes.at(denseIndex++);
        for (std::size_t row = 0; row < outputWidth; ++row) {
            const std::size_t operations = inputWidth * 2;
            double referenceAbsoluteSum = std::abs(layer.biases.at(row));
            double referenceInputContribution = 0.0;
            double candidateInputContribution = 0.0;
            double candidateAbsoluteSum = std::abs(layer.biases.at(row));
            double quantizedAbsoluteSum = 0.0;
            double bfloatPerturbation = 0.0;
            const bool usesBfloat = mode == MathMode::BFloat16;

            for (std::size_t column = 0; column < inputWidth; ++column) {
                const double weight = layer.weights.at(row * inputWidth + column);
                const double weightMagnitude = std::abs(weight);
                const double rangeMagnitude = maximumMagnitude(sourceRange[column]);
                const double referenceMagnitude = addUp(rangeMagnitude,
                    referenceInputError[column]);
                const double candidateMagnitude = addUp(rangeMagnitude,
                    candidateInputError[column]);
                const double candidateTerm = multiplyUp(weightMagnitude,
                    candidateMagnitude);
                referenceAbsoluteSum = addUp(referenceAbsoluteSum,
                    multiplyUp(weightMagnitude, referenceMagnitude));
                referenceInputContribution = addUp(referenceInputContribution,
                    multiplyUp(weightMagnitude, referenceInputError[column]));
                candidateInputContribution = addUp(candidateInputContribution,
                    multiplyUp(weightMagnitude, candidateInputError[column]));
                candidateAbsoluteSum = addUp(candidateAbsoluteSum, candidateTerm);

                if (usesBfloat) {
                    const double inputQuantizationError = addUp(
                        candidateInputError[column],
                        bfloatInputError(candidateMagnitude));
                    const double quantizedInputMagnitude = checkedBfloatMagnitude(
                        candidateMagnitude);
                    const double quantizedWeight = checkedBfloatConstant(weight);
                    const double weightQuantizationError = absoluteDifferenceUp(
                        quantizedWeight, weight);
                    const double quantizedWeightMagnitude = roundedUp(
                        std::abs(quantizedWeight));
                    bfloatPerturbation = addUp(bfloatPerturbation,
                        addUp(multiplyUp(weightMagnitude, inputQuantizationError),
                              multiplyUp(weightQuantizationError,
                                         quantizedInputMagnitude)));
                    quantizedAbsoluteSum = addUp(quantizedAbsoluteSum,
                        multiplyUp(quantizedWeightMagnitude,
                                   quantizedInputMagnitude));
                }
            }

            outputRange[row] = denseRange(layer, sourceRange, inputWidth, row);
            const double referenceRoundoff = gammaBound(operations, float64Unit,
                referenceAbsoluteSum, float64Tiny);
            referenceOutputError[row] = addUp(referenceInputContribution,
                                               referenceRoundoff);

            if (mode == MathMode::Float64) {
                const double candidateRoundoff = gammaBound(operations, float64Unit,
                    candidateAbsoluteSum, float64Tiny);
                candidateOutputError[row] = addUp(candidateInputContribution,
                                                   candidateRoundoff);
            } else {
                const double bias = layer.biases.at(row);
                const double quantizedBias = checkedBfloatConstant(bias);
                const double quantizedBiasError = absoluteDifferenceUp(
                    quantizedBias, bias);
                const double quantizedBiasMagnitude = roundedUp(
                    std::abs(quantizedBias));
                quantizedAbsoluteSum = addUp(quantizedAbsoluteSum,
                                              quantizedBiasMagnitude);
                bfloatPerturbation = addUp(bfloatPerturbation,
                                            quantizedBiasError);
                const double floatRoundoff = gammaBound(operations, float32Unit,
                    quantizedAbsoluteSum, float32Tiny);
                const double accumulatorMagnitude = addUp(quantizedAbsoluteSum,
                                                            floatRoundoff);
                if (accumulatorMagnitude > bfloatMaximum
                    || addUp(accumulatorMagnitude,
                             bfloatOutputError(accumulatorMagnitude)) > bfloatMaximum) {
                    throw std::runtime_error("bfloat16 certificate cannot rule out a non-finite layer result");
                }
                candidateOutputError[row] = addUp(
                    addUp(bfloatPerturbation, floatRoundoff),
                    bfloatOutputError(accumulatorMagnitude));
            }
        }
    }

    const std::vector<Interval>& finalRange = exactValues.at(network.output);
    const std::vector<double>& finalReferenceError = referenceErrors.at(network.output);
    const std::vector<double>& finalCandidateError = candidateErrors.at(network.output);
    Certificate result{finalRange, {}, 0.0};
    result.outputErrorBounds.reserve(finalRange.size());
    for (std::size_t i = 0; i < finalRange.size(); ++i) {
        const double bound = addUp(finalReferenceError[i], finalCandidateError[i]);
        result.outputErrorBounds.push_back(bound);
        result.maximumAbsoluteError = std::max(result.maximumAbsoluteError, bound);
    }
    return result;
}

std::size_t parseGridCount(const std::string& token) {
    std::size_t consumed = 0;
    unsigned long long count = 0;
    try {
        count = std::stoull(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("grid point count must be a positive integer");
    }
    if (consumed != token.size() || count < 2
        || count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("grid point count must be between 2 and the platform limit");
    }
    return static_cast<std::size_t>(count);
}

GridCheckResult checkCertificateGrid(const Network& network,
                                     const PrecisionPlan& plan,
                                     const std::vector<Interval>& domain,
                                     const Certificate& certificate,
                                     std::size_t pointsPerDimension) {
    constexpr std::size_t maximumGridPoints = 1000000;
    std::size_t totalPoints = 1;
    for (std::size_t dimension = 0; dimension < domain.size(); ++dimension) {
        if (totalPoints > maximumGridPoints / pointsPerDimension) {
            throw std::runtime_error("grid exceeds one million points; reduce points per dimension");
        }
        totalPoints *= pointsPerDimension;
    }

    PrecisionPlan referencePlan;
    referencePlan.denseModes.assign(denseLayerCount(network), MathMode::Float64);
    std::vector<double> input(domain.size());
    GridCheckResult result{0, 0.0, true};
    const auto visitDimension = [&](const auto& self, std::size_t dimension) -> void {
        if (dimension == domain.size()) {
            const std::vector<double> reference = evaluate(network, input, referencePlan);
            const std::vector<double> candidate = evaluate(network, input, plan);
            ++result.pointsVisited;
            for (std::size_t output = 0; output < reference.size(); ++output) {
                const double observed = std::abs(reference[output] - candidate[output]);
                result.maximumObservedError = std::max(result.maximumObservedError, observed);
                if (!std::isfinite(observed)
                    || observed > certificate.outputErrorBounds.at(output)) {
                    result.withinBound = false;
                }
            }
            return;
        }
        const Interval range = domain[dimension];
        for (std::size_t point = 0; point < pointsPerDimension; ++point) {
            const double fraction = static_cast<double>(point)
                / static_cast<double>(pointsPerDimension - 1);
            const double value = point == 0 ? range.low
                : point + 1 == pointsPerDimension ? range.high
                : range.low + (range.high - range.low) * fraction;
            input[dimension] = std::clamp(value, range.low, range.high);
            self(self, dimension + 1);
        }
    };
    visitDimension(visitDimension, 0);
    return result;
}

std::vector<Interval> readInputDomain(const std::string& path,
                                      std::size_t inputWidth) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open input domain file: " + path);
    }
    std::vector<Interval> domain;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream values(line);
        double low = 0.0;
        if (!(values >> low)) continue;
        double high = 0.0;
        if (!(values >> high) || !std::isfinite(low) || !std::isfinite(high)
            || low > high) {
            throw std::runtime_error("domain line " + std::to_string(lineNumber)
                + " must contain finite lower and upper bounds in order");
        }
        std::string extra;
        if (values >> extra) {
            throw std::runtime_error("domain line " + std::to_string(lineNumber)
                + " contains extra values");
        }
        domain.push_back(Interval{low, high});
    }
    if (domain.size() != inputWidth) {
        throw std::runtime_error("domain file must contain exactly "
            + std::to_string(inputWidth) + " lower/upper pairs");
    }
    return domain;
}

void writeCertificate(const std::string& path, const Certificate& certificate,
                      double requestedLimit, const PrecisionPlan& plan) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("could not create certificate report: " + path);
    output << "Proofline bounded-domain precision certificate v1\n"
           << "Arithmetic model: IEEE binary floating point, round-to-nearest, gradual underflow;"
              " BF16 round-to-nearest-even; compiler reassociation bounded by gamma_n roundoff\n"
           << "Requested maximum absolute error: " << std::setprecision(17)
           << requestedLimit << '\n'
           << "Maximum certified f64-reference difference: "
           << certificate.maximumAbsoluteError << '\n'
           << "Verdict: "
           << (certificate.maximumAbsoluteError <= requestedLimit ? "PASS" : "FAIL")
           << '\n'
           << "Projection plan:\n";
    for (std::size_t i = 0; i < plan.denseModes.size(); ++i) {
        output << "  Projection " << i << ": " << mathModeName(plan.denseModes[i]) << '\n';
    }
    output << "Output coordinate bounds (real reference interval, absolute error):\n";
    for (std::size_t i = 0; i < certificate.exactOutputRanges.size(); ++i) {
        output << "  " << i << ": [" << certificate.exactOutputRanges[i].low << ", "
               << certificate.exactOutputRanges[i].high << "], "
               << certificate.outputErrorBounds[i] << '\n';
    }
    if (!output) throw std::runtime_error("failed while writing certificate report: " + path);
}

} // namespace proofline
