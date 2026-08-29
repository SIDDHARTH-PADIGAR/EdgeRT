#include "edgert/ops/relu.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace edgert::ops {

Tensor ReluOp::compute(const std::vector<const Tensor*>& inputs) const {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Relu: expected 1 input, got " + std::to_string(inputs.size()));
    }
    const Tensor& x = *inputs[0];
    if (x.dtype() != DataType::Float32) {
        throw std::invalid_argument("Relu: only Float32 tensors are supported");
    }

    Tensor output(x.shape(), DataType::Float32);
    const float* px = x.data();
    float* po = output.data();
    for (int64_t i = 0; i < x.numel(); ++i) {
        po[i] = std::max(0.0F, px[i]);
    }
    return output;
}

}  // namespace edgert::ops