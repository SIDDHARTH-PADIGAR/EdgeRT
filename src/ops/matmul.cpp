#include "edgert/ops/matmul.h"

#include <stdexcept>
#include <string>

namespace edgert::ops {

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

    // Loop order is i -> k -> j, not the "obvious" i -> j -> k.
    //
    // Both B and the output are stored row-major, so walking along a row
    // (fixed row index, increasing column index) touches memory that's
    // physically next to itself — that's a stride-1 access, cheap for the
    // cache. Walking down a column instead jumps N floats every step,
    // which is a cache miss almost every time for large N.
    //
    // With i -> j -> k, the innermost loop over k reads B column-by-column
    // (pb[p * n + j], stride N) — a cache miss on nearly every access.
    // With i -> k -> j, the innermost loop over j reads B and the output
    // row-by-row (stride 1) instead, and a[i,k] is loaded once per k and
    // reused for the whole row. Same arithmetic, same FLOP count, just a
    // memory access pattern the cache actually likes.
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const float a_val = pa[i * k + p];
            for (int64_t j = 0; j < n; ++j) {
                po[i * n + j] += a_val * pb[p * n + j];
            }
        }
    }
    return output;
}

}  // namespace edgert::ops