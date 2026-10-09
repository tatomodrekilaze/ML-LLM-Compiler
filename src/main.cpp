#include "certificate.hpp"
#include "codegen.hpp"
#include "proofline.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace proofline;

namespace {
void writeValues(std::ostream& output, const std::vector<double>& values) {
    output << std::setprecision(17);
    for (std::size_t i = 0; i < values.size(); ++i) {
        output << (i == 0 ? "" : ", ") << values[i];
    }
}

void printUsage() {
    std::cout << "Usage:\n"
              << "  proofline inspect <model.proof>\n"
              << "  proofline run <model.proof> [--bf16 | --plan plan.txt] <input0> ...\n"
              << "  proofline compare <model.proof> <input0> <input1> ...\n"
              << "  proofline plan <model.proof> <calibration.txt> <abs-error-limit> <plan.txt>\n"
              << "  proofline certify <model.proof> <plan.txt> <input-domain.txt> <abs-error-limit> <report.txt>\n"
              << "  proofline check-grid <model.proof> <plan.txt> <input-domain.txt> <points-per-dimension>\n"
              << "  proofline compile <model.proof> <output.cpp> [--bf16 | --plan plan.txt]\n";
}

void inspect(const Network& network) {
    const IRValue& input = network.values.at(network.input);
    std::cout << "Input " << input.name << ": " << tensorDescription(input) << '\n';
    for (std::size_t i = 0; i < network.layers.size(); ++i) {
        const Layer& layer = network.layers[i];
        const IRValue& source = network.values.at(layer.inputs.at(0));
        const IRValue& result = network.values.at(layer.output);
        std::string extraInputs;
        if (layer.kind == LayerKind::Add) {
            extraInputs = " + " + network.values.at(layer.inputs.at(1)).name;
        } else if (layer.kind == LayerKind::Attention) {
            extraInputs = "(" + network.values.at(layer.inputs.at(1)).name + ", "
                + network.values.at(layer.inputs.at(2)).name
                + "; " + std::to_string(layer.heads) + " heads, "
                + (layer.causal ? "causal)" : "full)");
        }
        const std::string operation = layer.kind == LayerKind::Dense ? "Dense"
            : layer.kind == LayerKind::Linear ? "Linear"
            : layer.kind == LayerKind::Relu ? "ReLU"
            : layer.kind == LayerKind::RmsNorm ? "RMSNorm"
            : layer.kind == LayerKind::Softmax ? "Softmax"
            : layer.kind == LayerKind::Rope ? "RoPE"
            : layer.kind == LayerKind::Attention ? "Attention" : "Add";
        std::cout << "Operation " << i << ": " << operation << ' ';
        std::cout << source.name << extraInputs << " -> " << result.name << " : "
                  << tensorDescription(result) << '\n';
    }
    const IRValue& output = network.values.at(network.output);
    std::cout << "Output " << output.name << ": " << tensorDescription(output) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            printUsage();
            return 2;
        }

        const std::string action = argv[1];
        const Network network = readNetwork(argv[2]);

        if (action == "inspect") {
            if (argc != 3) {
                throw std::runtime_error("inspect does not accept input values");
            }
            inspect(network);
            return 0;
        }

        if (action == "plan") {
            if (argc != 6) {
                throw std::runtime_error("plan requires a calibration file, error limit, and output plan path");
            }
            const double errorLimit = parseInput(argv[4]);
            const std::size_t inputWidth = elementCount(network.values.at(network.input).type);
            const std::vector<std::vector<double>> samples =
                readCalibration(argv[3], inputWidth);
            const PlanSearchResult result = findPlan(network, samples, errorLimit);
            writePlan(argv[5], result, errorLimit);
            std::cout << "Selected plan " << argv[5] << '\n'
                      << "Calibration vectors: " << samples.size() << '\n'
                      << "Maximum observed absolute error: "
                      << std::setprecision(17) << result.maximumAbsoluteError << '\n'
                      << "Parameter payload: " << result.parameterBytes << " bytes\n"
                      << "Assignments checked: " << result.plansExamined << '\n';
            for (std::size_t i = 0; i < result.plan.denseModes.size(); ++i) {
                std::cout << "Projection " << i << ": "
                          << mathModeName(result.plan.denseModes[i]) << '\n';
            }
            return 0;
        }

        if (action == "certify") {
            if (argc != 7) {
                throw std::runtime_error("certify requires a plan, input domain, error limit, and report path");
            }
            const PrecisionPlan plan = readPlan(network, argv[3]);
            const std::vector<Interval> domain = readInputDomain(argv[4],
                elementCount(network.values.at(network.input).type));
            const double errorLimit = parseInput(argv[5]);
            if (errorLimit < 0.0) {
                throw std::runtime_error("absolute error limit must be nonnegative");
            }
            const Certificate certificate = certifyPlan(network, domain, plan);
            writeCertificate(argv[6], certificate, errorLimit, plan);
            std::cout << "Certified maximum absolute error: "
                      << std::setprecision(17) << certificate.maximumAbsoluteError
                      << '\n' << "Requested limit: " << errorLimit << '\n'
                      << "Verdict: "
                      << (certificate.maximumAbsoluteError <= errorLimit ? "PASS" : "FAIL")
                      << " (report: " << argv[6] << ")\n";
            return certificate.maximumAbsoluteError <= errorLimit ? 0 : 1;
        }

        if (action == "check-grid") {
            if (argc != 6) {
                throw std::runtime_error("check-grid requires a plan, input domain, and grid size");
            }
            const PrecisionPlan plan = readPlan(network, argv[3]);
            const std::vector<Interval> domain = readInputDomain(argv[4],
                elementCount(network.values.at(network.input).type));
            const std::size_t pointsPerDimension = parseGridCount(argv[5]);
            const Certificate certificate = certifyPlan(network, domain, plan);
            const GridCheckResult result = checkCertificateGrid(network, plan, domain,
                certificate, pointsPerDimension);
            std::cout << std::setprecision(17)
                      << "Grid points checked: " << result.pointsVisited << '\n'
                      << "Maximum sampled absolute error: " << result.maximumObservedError << '\n'
                      << "Certified maximum absolute error: "
                      << certificate.maximumAbsoluteError << '\n'
                      << "Sanity check: " << (result.withinBound ? "PASS" : "FAIL")
                      << " (finite grid sampling is not a proof)\n";
            return result.withinBound ? 0 : 1;
        }

        if (action == "run") {
            PrecisionPlan plan;
            plan.denseModes.assign(denseLayerCount(network), MathMode::Float64);
            int inputStart = 3;
            if (argc > 3 && std::string(argv[3]) == "--bf16") {
                plan.denseModes.assign(denseLayerCount(network), MathMode::BFloat16);
                inputStart = 4;
            } else if (argc > 3 && std::string(argv[3]) == "--plan") {
                if (argc < 5) {
                    throw std::runtime_error("run --plan requires a plan file");
                }
                plan = readPlan(network, argv[4]);
                inputStart = 5;
            }
            std::vector<double> input;
            for (int i = inputStart; i < argc; ++i) {
                input.push_back(parseInput(argv[i]));
            }
            const std::vector<double> result = evaluate(network, std::move(input), plan);
            std::cout << "Output (precision plan): ";
            writeValues(std::cout, result);
            std::cout << '\n';
            return 0;
        }

        if (action == "compare") {
            std::vector<double> input;
            for (int i = 3; i < argc; ++i) {
                input.push_back(parseInput(argv[i]));
            }
            const std::vector<double> reference = evaluate(network, input, MathMode::Float64);
            const std::vector<double> approximate = evaluate(network, std::move(input),
                                                              MathMode::BFloat16);
            std::cout << std::setprecision(17);
            for (std::size_t i = 0; i < reference.size(); ++i) {
                const double absoluteError = std::abs(reference[i] - approximate[i]);
                std::cout << "output[" << i << "]: f64=" << reference[i]
                          << " bf16=" << approximate[i]
                          << " absolute_error=" << absoluteError;
                if (reference[i] != 0.0) {
                    std::cout << " relative_error="
                              << absoluteError / std::abs(reference[i]);
                } else {
                    std::cout << " relative_error=undefined (reference is zero)";
                }
                std::cout << '\n';
            }
            reportTensorStorage(network);
            return 0;
        }

        if (action == "compile") {
            if (argc != 4 && argc != 5 && argc != 6) {
                throw std::runtime_error("compile requires an output path and optional precision selection");
            }
            PrecisionPlan plan;
            plan.denseModes.assign(denseLayerCount(network), MathMode::Float64);
            if (argc == 5) {
                if (std::string(argv[4]) != "--bf16") {
                    throw std::runtime_error("the only precision option currently available is --bf16");
                }
                plan.denseModes.assign(denseLayerCount(network), MathMode::BFloat16);
            } else if (argc == 6) {
                if (std::string(argv[4]) != "--plan") {
                    throw std::runtime_error("expected --plan followed by a plan file");
                }
                plan = readPlan(network, argv[5]);
            }
            writeCpp(network, argv[3], plan);
            std::cout << "Generated " << argv[3] << '\n';
            return 0;
        }

        printUsage();
        throw std::runtime_error("unknown command '" + action + "'");
    } catch (const std::exception& error) {
        std::cerr << "proofline: " << error.what() << '\n';
        return 1;
    }
}

