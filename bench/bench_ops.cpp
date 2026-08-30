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

void bench_matmul(int64_t m, int64_t k, int64_t n, int warmup, int iters) {
    Tensor a({m, k});
    Tensor b({k, n});
    a.fill(1.0F);
    b.fill(1.0F);
    MatMulOp op;

    auto durations = time_ms([&] { Tensor c = op.compute({&a, &b}); }, warmup, iters);

    const double flops_per_run =
        2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);

    char name[64];
    std::snprintf(name, sizeof(name), "MatMul [%ld,%ld]x[%ld,%ld]", static_cast<long>(m),
                  static_cast<long>(k), static_cast<long>(k), static_cast<long>(n));
    print_result(name, summarize(durations), flops_per_run / 1e9);
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