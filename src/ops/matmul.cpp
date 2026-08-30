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
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = output.data();

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0F;
            for (int64_t p = 0; p < k; ++p) {
                sum += pa[i * k + p] * pb[p * n + j];
            }
            po[i * n + j] = sum;
        }
    }
    return output;
}

}  // namespace edgert::ops