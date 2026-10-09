#include "attention_cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace proofline {
namespace {

std::size_t checkedStorageSize(std::size_t tokens, std::size_t width) {
    if (width == 0 || tokens > std::numeric_limits<std::size_t>::max() / width) {
        throw std::invalid_argument("attention cache shape is empty or too large");
    }
    return tokens * width;
}

void requireFinite(std::span<const double> values, const char* name) {
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
}

} // namespace

AttentionKvCache::AttentionKvCache(std::size_t capacity, std::size_t keyWidth,
                                   std::size_t valueWidth, std::size_t heads)
    : capacity_(capacity), keyWidth_(keyWidth), valueWidth_(valueWidth), heads_(heads) {
    if (capacity_ == 0 || heads_ == 0 || keyWidth_ == 0 || valueWidth_ == 0
        || keyWidth_ % heads_ != 0 || valueWidth_ % heads_ != 0) {
        throw std::invalid_argument("attention cache dimensions must be positive and divisible by heads");
    }
    keys_.reserve(checkedStorageSize(capacity_, keyWidth_));
    values_.reserve(checkedStorageSize(capacity_, valueWidth_));
}

void AttentionKvCache::append(std::span<const double> key,
                              std::span<const double> value) {
    if (key.size() != keyWidth_ || value.size() != valueWidth_) {
        throw std::invalid_argument("cached key or value width does not match the cache");
    }
    if (tokens_ == capacity_) {
        throw std::length_error("attention key/value cache capacity exceeded");
    }
    requireFinite(key, "key");
    requireFinite(value, "value");
    keys_.insert(keys_.end(), key.begin(), key.end());
    values_.insert(values_.end(), value.begin(), value.end());
    ++tokens_;
}

std::vector<double> AttentionKvCache::attend(std::span<const double> query) const {
    if (query.size() != keyWidth_) {
        throw std::invalid_argument("query width does not match the attention cache");
    }
    if (tokens_ == 0) {
        throw std::logic_error("cannot attend before adding a key/value pair");
    }
    requireFinite(query, "query");

    const std::size_t keyHeadWidth = keyWidth_ / heads_;
    const std::size_t valueHeadWidth = valueWidth_ / heads_;
    const double scale = 1.0 / std::sqrt(static_cast<double>(keyHeadWidth));
    std::vector<double> output(valueWidth_, 0.0);
    std::vector<double> scores(tokens_);

    for (std::size_t head = 0; head < heads_; ++head) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t token = 0; token < tokens_; ++token) {
            double dot = 0.0;
            for (std::size_t feature = 0; feature < keyHeadWidth; ++feature) {
                const std::size_t headOffset = head * keyHeadWidth + feature;
                dot += query[headOffset] * keys_[token * keyWidth_ + headOffset];
            }
            scores[token] = dot * scale;
            maximum = std::max(maximum, scores[token]);
        }

        double exponentialSum = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            exponentialSum += score;
        }

        for (std::size_t feature = 0; feature < valueHeadWidth; ++feature) {
            double weightedSum = 0.0;
            for (std::size_t token = 0; token < tokens_; ++token) {
                const double probability = scores[token] / exponentialSum;
                const std::size_t valueOffset = head * valueHeadWidth + feature;
                weightedSum += probability * values_[token * valueWidth_ + valueOffset];
            }
            output[head * valueHeadWidth + feature] = weightedSum;
        }
    }
    return output;
}

std::size_t AttentionKvCache::size() const noexcept {
    return tokens_;
}

std::size_t AttentionKvCache::capacity() const noexcept {
    return capacity_;
}

} // namespace proofline
