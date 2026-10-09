#pragma once

#include "proofline.hpp"

namespace proofline {

struct Interval {
    double low;
    double high;
};

struct Certificate {
    std::vector<Interval> exactOutputRanges;
    std::vector<double> outputErrorBounds;
    double maximumAbsoluteError;
};

struct GridCheckResult {
    std::size_t pointsVisited;
    double maximumObservedError;
    bool withinBound;
};

Certificate certifyPlan(const Network& network,
                        const std::vector<Interval>& inputDomain,
                        const PrecisionPlan& plan);
std::size_t parseGridCount(const std::string& token);
GridCheckResult checkCertificateGrid(const Network& network,
                                     const PrecisionPlan& plan,
                                     const std::vector<Interval>& domain,
                                     const Certificate& certificate,
                                     std::size_t pointsPerDimension);
std::vector<Interval> readInputDomain(const std::string& path,
                                      std::size_t inputWidth);
void writeCertificate(const std::string& path, const Certificate& certificate,
                      double requestedLimit, const PrecisionPlan& plan);

} // namespace proofline
