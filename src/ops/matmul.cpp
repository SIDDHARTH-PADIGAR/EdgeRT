#include "edgert/ops/matmul.h"

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

// Scalar i -> k -> j loop: correct on every platform, and the fallback
// used whenever the CPU doesn't support AVX2/FMA.
//
// Loop order is i -> k -> j, not the "obvious" i -> j -> k. Both B and the
// output are stored row-major, so walking along a row (fixed row index,
// increasing column index) touches memory that's physically next to
// itself — a stride-1 access, cheap for the cache. Walking down a column
// instead jumps N floats every step, a cache miss almost every time for
// large N. With i -> j -> k, the innermost loop over k would read B
// column-by-column; with i -> k -> j, the innermost loop over j reads B
// and the output row-by-row instead, and a[i,k] is loaded once per k and
// reused for the whole row.
void matmul_ikj_scalar(int64_t m, int64_t k, int64_t n, const float* pa, const float* pb, float* po) {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const float a_val = pa[i * k + p];
            for (int64_t j = 0; j < n; ++j) {
                po[i * n + j] += a_val * pb[p * n + j];
            }
        }
    }
}

#if EDGERT_X86
// Same i -> k -> j loop, but the innermost loop over j processes 8 floats
// at a time using AVX2 (256-bit registers = 8 x float32) with a fused
// multiply-add: out[j..j+8] += a_val * b[j..j+8] in one instruction instead
// of eight separate multiply-then-add pairs. Whatever's left over after
// the last full group of 8 (n % 8 elements) falls through to the same
// scalar loop as the fallback, so this is correct for every n, not just
// multiples of 8.
//
// __attribute__((target("avx2,fma"))) tells GCC/Clang to compile *only
// this function* using AVX2/FMA instructions, while the rest of this file
// stays at the compiler's default baseline instruction set. That's what
// lets us check __builtin_cpu_supports("avx2") at runtime and safely fall
// back to the scalar version on a CPU that doesn't have it, instead of
// the whole program requiring AVX2 just to start up.
__attribute__((target("avx2,fma"))) void matmul_ikj_avx2(int64_t m, int64_t k, int64_t n,
                                                           const float* pa, const float* pb,
                                                           float* po) {
    for (int64_t i = 0; i < m; ++i) {
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
            // Remainder: fewer than 8 elements left in this row.
            for (; j < n; ++j) {
                po[i * n + j] += a_val * pb[p * n + j];
            }
        }
    }
}
#endif  // EDGERT_X86

}  // namespace

Tensor MatMulOp::compute(const std::vector<const Tensor*>& inputs) const {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatMul: expected 2 inputs, got " + std::to_string(inputs.size()));
    }
    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

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

    Tensor output({m, n}, DataType::Float32);
    output.fill(0.0F);
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = output.data();

#if EDGERT_X86
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        matmul_ikj_avx2(m, k, n, pa, pb, po);
    } else {
        matmul_ikj_scalar(m, k, n, pa, pb, po);
    }
#else
    matmul_ikj_scalar(m, k, n, pa, pb, po);
#endif

    return output;
}

}  // namespace edgert::ops