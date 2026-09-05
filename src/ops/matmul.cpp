#include "edgert/ops/matmul.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define EDGERT_X86 1
#else
#define EDGERT_X86 0
#endif

namespace edgert::ops {

namespace {

struct MatMulShape {
    int64_t m;
    int64_t k;
    int64_t n;
};

MatMulShape validate_matmul(const Tensor& a, const Tensor& b) {
    if (a.dtype() != DataType::Float32 || b.dtype() != DataType::Float32) {
        throw std::invalid_argument("MatMul: only Float32 tensors are supported");
    }
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("MatMul: inputs must be rank 2 (matrices)");
    }
    const int64_t m = a.shape()[0];
    const int64_t k = a.shape()[1];
    const int64_t k2 = b.shape()[0];
    const int64_t n = b.shape()[1];
    if (k != k2) {
        throw std::invalid_argument(
            "MatMul: inner dimensions must match: A is [" + std::to_string(m) + ", " +
            std::to_string(k) + "], B is [" + std::to_string(k2) + ", " + std::to_string(n) + "]");
    }
    return {m, k, n};
}

void matmul_ikj_scalar(int64_t row_start, int64_t row_end, int64_t k, int64_t n, const float* pa,
                        const float* pb, float* po) {
    for (int64_t i = row_start; i < row_end; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const float a_val = pa[i * k + p];
            for (int64_t j = 0; j < n; ++j) {
                po[i * n + j] += a_val * pb[p * n + j];
            }
        }
    }
}

#if EDGERT_X86
__attribute__((target("avx2,fma"))) void matmul_ikj_avx2(int64_t row_start, int64_t row_end,
                                                           int64_t k, int64_t n, const float* pa,
                                                           const float* pb, float* po) {
    for (int64_t i = row_start; i < row_end; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const float a_val = pa[i * k + p];
            const __m256 a_vec = _mm256_set1_ps(a_val);

            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&pb[p * n + j]);
                __m256 out_vec = _mm256_loadu_ps(&po[i * n + j]);
                out_vec = _mm256_fmadd_ps(a_vec, b_vec, out_vec);
                _mm256_storeu_ps(&po[i * n + j], out_vec);
            }
            for (; j < n; ++j) {
                po[i * n + j] += a_val * pb[p * n + j];
            }
        }
    }
}
#endif  // EDGERT_X86

void matmul_dispatch(int64_t row_start, int64_t row_end, int64_t k, int64_t n, const float* pa,
                      const float* pb, float* po) {
#if EDGERT_X86
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        matmul_ikj_avx2(row_start, row_end, k, n, pa, pb, po);
        return;
    }
#endif
    matmul_ikj_scalar(row_start, row_end, k, n, pa, pb, po);
}

}  // namespace

Tensor MatMulOp::compute(const std::vector<const Tensor*>& inputs) const {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatMul: expected 2 inputs, got " + std::to_string(inputs.size()));
    }
    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];
    const auto [m, k, n] = validate_matmul(a, b);

    Tensor output({m, n}, DataType::Float32);
    output.fill(0.0F);
    matmul_dispatch(0, m, k, n, a.data(), b.data(), output.data());
    return output;
}

Tensor matmul_parallel(const Tensor& a, const Tensor& b, ThreadPool& pool) {
    const auto [m, k, n] = validate_matmul(a, b);

    Tensor output({m, n}, DataType::Float32);
    output.fill(0.0F);
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = output.data();

    if (m == 0) {
        return output;
    }

    const int64_t num_workers = static_cast<int64_t>(pool.num_threads());
    const int64_t rows_per_task = (m + num_workers - 1) / num_workers;

    for (int64_t row_start = 0; row_start < m; row_start += rows_per_task) {
        const int64_t row_end = std::min(row_start + rows_per_task, m);
        pool.submit([row_start, row_end, k, n, pa, pb, po] {
            matmul_dispatch(row_start, row_end, k, n, pa, pb, po);
        });
    }
    pool.wait_all();
    return output;
}

}  // namespace edgert::ops