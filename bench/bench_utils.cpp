#include "edgert/bench_utils.h"

#include <algorithm>
#include <cstdio>

namespace edgert::bench {

double min_value(const std::vector<double>& values) {
    return *std::min_element(values.begin(), values.end());
}

double median_value(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

void print_result(const std::string& name, const BenchStats& stats, double gflops_per_run) {
    const double min_gflops = gflops_per_run / (stats.min_ms / 1000.0);
    const double median_gflops = gflops_per_run / (stats.median_ms / 1000.0);
    std::printf("%-32s min: %8.3f ms (%7.2f GFLOP/s)   median: %8.3f ms (%7.2f GFLOP/s)\n",
                name.c_str(), stats.min_ms, min_gflops, stats.median_ms, median_gflops);
}

}  // namespace edgert::bench