#include <cstdint>
#include <cstdio>

#include "edgert/bench_utils.h"
#include "edgert/ops/linear.h"
#include "edgert/ops/matmul.h"
#include "edgert/tensor.h"

using edgert::Tensor;
using edgert::bench::print_result;
using edgert::bench::summarize;
using edgert::bench::time_ms;
using edgert::ops::LinearOp;
using edgert::ops::MatMulOp;

namespace {

// Reference-only implementation using the original i -> j -> k loop order.
// Kept here purely so the benchmark has an honest "before" number to show
// next to MatMulOp's current i -> k -> j implementation. It is NOT
// registered as an Operator — MatMulOp itself now uses the faster order,
// so there's exactly one MatMul implementation in production, not two
// competing ones.
void naive_ijk_matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    const int64_t m = a.shape()[0];
    const int64_t k = a.shape()[1];
    const int64_t n = b.shape()[1];
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = out.data();
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0F;
            for (int64_t p = 0; p < k; ++p) {
                sum += pa[i * k + p] * pb[p * n + j];
            }
            po[i * n + j] = sum;
        }
    }
}

void bench_matmul(int64_t m, int64_t k, int64_t n, int warmup, int iters) {
    Tensor a({m, k});
    Tensor b({k, n});
    a.fill(1.0F);
    b.fill(1.0F);

    Tensor naive_out({m, n});
    auto naive_durations =
        time_ms([&] { naive_ijk_matmul(a, b, naive_out); }, warmup, iters);

    MatMulOp op;
    auto ikj_durations = time_ms([&] { Tensor c = op.compute({&a, &b}); }, warmup, iters);

    const double flops_per_run =
        2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);

    char name[64];
    std::snprintf(name, sizeof(name), "MatMul ijk (old) [%ld,%ld,%ld]", static_cast<long>(m),
                  static_cast<long>(k), static_cast<long>(n));
    const auto naive_stats = summarize(naive_durations);
    print_result(name, naive_stats, flops_per_run / 1e9);

    std::snprintf(name, sizeof(name), "MatMul ikj (current) [%ld,%ld,%ld]", static_cast<long>(m),
                  static_cast<long>(k), static_cast<long>(n));
    const auto ikj_stats = summarize(ikj_durations);
    print_result(name, ikj_stats, flops_per_run / 1e9);

    std::printf("  -> %.2fx faster from loop reordering alone (same math, same FLOPs)\n\n",
                naive_stats.min_ms / ikj_stats.min_ms);
}

void bench_linear(int64_t batch, int64_t in_features, int64_t out_features, int warmup, int iters) {
    Tensor x({batch, in_features});
    Tensor w({out_features, in_features});
    Tensor b({out_features});
    x.fill(1.0F);
    w.fill(1.0F);
    b.fill(0.0F);
    LinearOp op;

    auto durations = time_ms([&] { Tensor y = op.compute({&x, &w, &b}); }, warmup, iters);

    const double flops_per_run = 2.0 * static_cast<double>(batch) *
                                  static_cast<double>(in_features) * static_cast<double>(out_features);

    char name[64];
    std::snprintf(name, sizeof(name), "Linear batch=%ld %ld->%ld", static_cast<long>(batch),
                  static_cast<long>(in_features), static_cast<long>(out_features));
    print_result(name, summarize(durations), flops_per_run / 1e9);
}

}  // namespace

int main() {
    std::printf("EdgeRT operator benchmarks (naive, single-threaded, no SIMD)\n\n");

    constexpr int kWarmup = 3;
    constexpr int kIters = 10;

    bench_matmul(64, 64, 64, kWarmup, kIters);
    bench_matmul(128, 128, 128, kWarmup, kIters);
    bench_matmul(256, 256, 256, kWarmup, kIters);

    bench_linear(1, 256, 256, kWarmup, kIters);
    bench_linear(32, 256, 256, kWarmup, kIters);

    return 0;
}