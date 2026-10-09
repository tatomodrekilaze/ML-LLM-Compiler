#include "attention_cache.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void requireNear(double actual, double expected, const std::string& label) {
    if (std::abs(actual - expected) > 1.0e-14) {
        throw std::runtime_error(label + " differs from its expected value");
    }
}

template <typename Function>
void requireThrows(Function&& function, const std::string& label) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label + " did not reject invalid input");
}

} // namespace

int main() {
    try {
        proofline::AttentionKvCache cache(2, 4, 4, 2);
        const std::array<double, 4> key0{1.0, 0.0, 0.0, 1.0};
        const std::array<double, 4> value0 = key0;
        const std::array<double, 4> query0 = key0;
        cache.append(key0, value0);
        if (cache.size() != 1 || cache.capacity() != 2) {
            throw std::runtime_error("cache length or capacity is incorrect after the first token");
        }
        const auto firstOutput = cache.attend(query0);
        for (std::size_t i = 0; i < firstOutput.size(); ++i) {
            requireNear(firstOutput[i], value0[i], "first-token attention output");
        }

        const std::array<double, 4> key1{0.0, 1.0, 1.0, 0.0};
        const std::array<double, 4> value1 = key1;
        const std::array<double, 4> query1 = key1;
        cache.append(key1, value1);
        const auto secondOutput = cache.attend(query1);
        const double laterWeight = 1.0 / (1.0 + std::exp(-1.0 / std::sqrt(2.0)));
        const std::array<double, 4> expected{
            1.0 - laterWeight, laterWeight, laterWeight, 1.0 - laterWeight};
        for (std::size_t i = 0; i < expected.size(); ++i) {
            requireNear(secondOutput[i], expected[i], "second-token attention output");
        }

        requireThrows([&] { cache.append(key0, value0); }, "full cache");
        const std::array<double, 3> wrongWidth{};
        requireThrows([&] { cache.attend(wrongWidth); }, "wrong query width");
        std::cout << "incremental two-head KV cache matched hand calculations and rejected invalid capacity/shape\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attention cache check: " << error.what() << '\n';
        return 1;
    }
}
