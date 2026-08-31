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
// Kept here purely so the benchmark has an honest "before" number. Not
// registered as an Operator.
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

// Reference-only scalar i -> k -> j order, with no SIMD. A snapshot of
// what MatMulOp used before AVX2 dispatch was added, kept here so the
// benchmark can isolate "cache-friendly order alone" from "cache-friendly
// order plus vectorization" (what MatMulOp actually runs now). Not
// registered as an Operator — production has exactly one MatMul
// implementation, which picks its own fast path internally.
void scalar_ikj_matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    const int64_t m = a.shape()[0];
    const int64_t k = a.shape()[1];
    const int64_t n = b.shape()[1];
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = out.data();
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const float a_val = pa[i * k + p];
            for (int64_t j = 0; j < n; ++j) {
                po[i * n + j] += a_val * pb[p * n + j];
            }
        }
    }
}

void bench_matmul(int64_t m, int64_t k, int64_t n, int warmup, int iters) {
    Tensor a({m, k});
    Tensor b({k, n});
    a.fill(1.0F);
    b.fill(1.0F);

    Tensor ijk_out({m, n});
    auto ijk_durations = time_ms([&] { naive_ijk_matmul(a, b, ijk_out); }, warmup, iters);

    Tensor ikj_out({m, n});
    auto ikj_durations = time_ms(
        [&] {
            ikj_out.fill(0.0F);
            scalar_ikj_matmul(a, b, ikj_out);
        },
        warmup, iters);

    MatMulOp op;
    auto simd_durations = time_ms([&] { Tensor c = op.compute({&a, &b}); }, warmup, iters);

    const double flops_per_run =
        2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);

    char name[96];

    std::snprintf(name, sizeof(name), "MatMul ijk scalar [%ld,%ld,%ld]", static_cast<long>(m),
                  static_cast<long>(k), static_cast<long>(n));
    const auto ijk_stats = summarize(ijk_durations);
    print_result(name, ijk_stats, flops_per_run / 1e9);

    std::snprintf(name, sizeof(name), "MatMul ikj scalar [%ld,%ld,%ld]", static_cast<long>(m),
                  static_cast<long>(k), static_cast<long>(n));
    const auto ikj_stats = summarize(ikj_durations);
    print_result(name, ikj_stats, flops_per_run / 1e9);

    std::snprintf(name, sizeof(name), "MatMul ikj AVX2 (current) [%ld,%ld,%ld]", static_cast<long>(m),
                  static_cast<long>(k), static_cast<long>(n));
    const auto simd_stats = summarize(simd_durations);
    print_result(name, simd_stats, flops_per_run / 1e9);

    std::printf("  -> %.2fx faster: ijk -> ikj (cache order alone)\n",
                ijk_stats.min_ms / ikj_stats.min_ms);
    std::printf("  -> %.2fx faster: ikj scalar -> ikj AVX2 (vectorization alone)\n",
                ikj_stats.min_ms / simd_stats.min_ms);
    std::printf("  -> %.2fx faster overall: ijk -> current MatMulOp\n\n",
                ijk_stats.min_ms / simd_stats.min_ms);
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
    std::printf("EdgeRT operator benchmarks (single-threaded; MatMul auto-dispatches to AVX2+FMA "
                 "when available, Linear is scalar)\n\n");

    constexpr int kWarmup = 3;
    constexpr int kIters = 10;

    bench_matmul(64, 64, 64, kWarmup, kIters);
    bench_matmul(128, 128, 128, kWarmup, kIters);
    bench_matmul(256, 256, 256, kWarmup, kIters);

    bench_linear(1, 256, 256, kWarmup, kIters);
    bench_linear(32, 256, 256, kWarmup, kIters);

    return 0;
}