#include "edgert/ops/linear.h"

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

// Scalar loop: correct on every platform, and the fallback used whenever
// the CPU doesn't support AVX2/FMA. Each output element is a dot product
// of one x row and one W row, both already contiguous in memory — unlike
// MatMul, there's no column-striding problem to fix here, so this loop
// order was already cache-friendly from the start.
void linear_scalar(int64_t batch, int64_t in_features, int64_t out_features, const float* px,
                    const float* pw, const float* pb, float* po) {
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_features; ++o) {
            float sum = pb[o];
            for (int64_t i = 0; i < in_features; ++i) {
                sum += px[n * in_features + i] * pw[o * in_features + i];
            }
            po[n * out_features + o] = sum;
        }
    }
}

#if EDGERT_X86
// Same dot product, but computed 8 floats at a time with AVX2+FMA. Unlike
// MatMul's kernel (which accumulates into 8 separate output positions at
// once), here we're accumulating a SINGLE output scalar from 8 products at
// a time, so after the vectorized loop we need a "horizontal sum": add the
// 8 lanes of the accumulator together into one number. Simplest correct
// way to do that is to store the 8 lanes to a small array and sum them
// with a normal loop — cheap, since it only happens once per output
// element, not once per input element. Whatever's left over after the
// last full group of 8 (in_features % 8) is handled by the same scalar
// accumulation as the fallback.
__attribute__((target("avx2,fma"))) void linear_avx2(int64_t batch, int64_t in_features,
                                                       int64_t out_features, const float* px,
                                                       const float* pw, const float* pb,
                                                       float* po) {
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_features; ++o) {
            __m256 acc = _mm256_setzero_ps();

            int64_t i = 0;
            for (; i + 8 <= in_features; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(&px[n * in_features + i]);
                __m256 w_vec = _mm256_loadu_ps(&pw[o * in_features + i]);
                acc = _mm256_fmadd_ps(x_vec, w_vec, acc);
            }

            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            float sum = pb[o];
            for (float lane : lanes) {
                sum += lane;
            }

            // Remainder: fewer than 8 elements left in this dot product.
            for (; i < in_features; ++i) {
                sum += px[n * in_features + i] * pw[o * in_features + i];
            }
            po[n * out_features + o] = sum;
        }
    }
}
#endif  // EDGERT_X86

}  // namespace

Tensor LinearOp::compute(const std::vector<const Tensor*>& inputs) const {
    if (inputs.size() != 3) {
        throw std::invalid_argument("Linear: expected 3 inputs (x, W, b), got " +
                                     std::to_string(inputs.size()));
    }
    const Tensor& x = *inputs[0];
    const Tensor& w = *inputs[1];
    const Tensor& b = *inputs[2];

    if (x.dtype() != DataType::Float32 || w.dtype() != DataType::Float32 ||
        b.dtype() != DataType::Float32) {
        throw std::invalid_argument("Linear: only Float32 tensors are supported");
    }
    if (x.ndim() != 2) {
        throw std::invalid_argument("Linear: x must be rank 2 [batch, in_features]");
    }
    if (w.ndim() != 2) {
        throw std::invalid_argument("Linear: W must be rank 2 [out_features, in_features]");
    }
    if (b.ndim() != 1) {
        throw std::invalid_argument("Linear: b must be rank 1 [out_features]");
    }

    const int64_t batch = x.shape()[0];
    const int64_t in_features = x.shape()[1];
    const int64_t out_features = w.shape()[0];
    const int64_t w_in_features = w.shape()[1];

    if (in_features != w_in_features) {
        throw std::invalid_argument(
            "Linear: x's in_features (" + std::to_string(in_features) +
            ") does not match W's in_features (" + std::to_string(w_in_features) + ")");
    }
    if (b.shape()[0] != out_features) {
        throw std::invalid_argument(
            "Linear: b's size (" + std::to_string(b.shape()[0]) +
            ") does not match W's out_features (" + std::to_string(out_features) + ")");
    }

    Tensor output({batch, out_features}, DataType::Float32);
    const float* px = x.data();
    const float* pw = w.data();
    const float* pb = b.data();
    float* po = output.data();

#if EDGERT_X86
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        linear_avx2(batch, in_features, out_features, px, pw, pb, po);
    } else {
        linear_scalar(batch, in_features, out_features, px, pw, pb, po);
    }
#else
    linear_scalar(batch, in_features, out_features, px, pw, pb, po);
#endif

    return output;
}

}  // namespace edgert::ops