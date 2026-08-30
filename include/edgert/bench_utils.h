#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace edgert::bench {

// Runs `fn` `warmup_iters` times (discarded, lets caches/branch predictors
// settle) followed by `timed_iters` timed runs, and returns each timed
// run's wall-clock duration in milliseconds, in the order they ran. The
// caller decides how to summarize (min, median, ...) — different
// workloads care about different statistics.
template <typename Fn>
std::vector<double> time_ms(Fn&& fn, int warmup_iters, int timed_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn();
    }
    std::vector<double> durations;
    durations.reserve(static_cast<std::size_t>(timed_iters));
    for (int i = 0; i < timed_iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        fn();
        auto end = std::chrono::steady_clock::now();
        durations.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return durations;
}

// Smallest value in `values`. This is the conventional statistic for
// microbenchmarks: system noise (scheduler preemption, other processes,
// thermal effects) only ever makes a run slower than the workload's true
// cost, never faster — so the fastest observed run is the closest
// estimate of the workload's actual cost.
double min_value(const std::vector<double>& values);

// Middle value of `values` (sorts a copy). Useful alongside min_value()
// to see typical-case behavior, not just best-case.
double median_value(std::vector<double> values);

struct BenchStats {
    double min_ms;
    double median_ms;
};

inline BenchStats summarize(const std::vector<double>& durations_ms) {
    return BenchStats{min_value(durations_ms), median_value(durations_ms)};
}

// Prints one formatted benchmark result line. `gflops_per_run` is the
// number of billion floating-point operations one call of the benchmarked
// workload performs (computed by the caller from its problem size), used
// to convert timings into throughput.
void print_result(const std::string& name, const BenchStats& stats, double gflops_per_run);

}  // namespace edgert::bench