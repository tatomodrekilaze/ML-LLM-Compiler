#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace proofline {

enum class ElementType {
    Float64,
};

enum class LayerKind {
    Dense,
    Linear,
    Relu,
    RmsNorm,
    Softmax,
    Rope,
    Add,
    Attention,
};

enum class MathMode {
    Float64,
    BFloat16,
};

using ValueId = std::size_t;

struct TensorType {
    ElementType elementType;
    std::vector<std::size_t> shape;
};

struct IRValue {
    std::string name;
    TensorType type;
};

struct Layer {
    LayerKind kind;
    std::vector<ValueId> inputs;
    ValueId output;
    std::vector<double> weights;
    std::vector<double> biases;
    double epsilon = 1.0e-5;
    std::size_t position = 0;
    double ropeBase = 10000.0;
    bool causal = true;
    std::size_t heads = 1;
};

struct Network {
    std::vector<IRValue> values;
    std::vector<Layer> layers;
    ValueId input = 0;
    ValueId output = 0;
};

struct PrecisionPlan {
    std::vector<MathMode> denseModes;
};

struct PlanSearchResult {
    PrecisionPlan plan;
    double maximumAbsoluteError;
    std::size_t parameterBytes;
    std::size_t plansExamined;
};

std::size_t widthOf(const TensorType& type);
std::size_t elementCount(const TensorType& type);
std::string mathModeName(MathMode mode);
std::uint16_t encodeBFloat16(double value);
float decodeBFloat16(std::uint16_t bits);
double roundToBFloat16(double value);
std::string typeName(ElementType type);
std::string tensorDescription(const IRValue& value);

Network readNetwork(const std::string& path);
std::size_t denseLayerCount(const Network& network);
std::vector<double> evaluate(const Network& network, std::vector<double> input,
                             const PrecisionPlan& plan);
std::vector<double> evaluate(const Network& network, std::vector<double> input,
                             MathMode uniformMode);
void reportTensorStorage(const Network& network);
double parseInput(const std::string& token);
std::vector<std::vector<double>> readCalibration(const std::string& path,
                                                  std::size_t inputWidth);
std::size_t parameterPayloadBytes(const Network& network,
                                  const PrecisionPlan& plan);
PlanSearchResult findPlan(const Network& network,
                          const std::vector<std::vector<double>>& samples,
                          double absoluteErrorLimit);
void writePlan(const std::string& path, const PlanSearchResult& result,
               double absoluteErrorLimit);
PrecisionPlan readPlan(const Network& network, const std::string& path);

} // namespace proofline
