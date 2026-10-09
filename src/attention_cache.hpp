#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace proofline {

// A bounded cache for one sequence and one attention layer. Keys and values
// are stored in token-major order, with heads laid out next to one another.
class AttentionKvCache {
public:
    AttentionKvCache(std::size_t capacity, std::size_t keyWidth,
                     std::size_t valueWidth, std::size_t heads);

    void append(std::span<const double> key, std::span<const double> value);
    std::vector<double> attend(std::span<const double> query) const;

    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;

private:
    std::size_t capacity_;
    std::size_t keyWidth_;
    std::size_t valueWidth_;
    std::size_t heads_;
    std::size_t tokens_ = 0;
    std::vector<double> keys_;
    std::vector<double> values_;
};

} // namespace proofline
