#include "edgert/ops/add.h"

#include <stdexcept>
#include <string>

namespace edgert::ops {

Tensor AddOp::compute(const std::vector<const Tensor*>& inputs) const {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Add: expected 2 inputs, got " + std::to_string(inputs.size()));
    }
    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

    if (a.dtype() != DataType::Float32 || b.dtype() != DataType::Float32) {
        throw std::invalid_argument("Add: only Float32 tensors are supported");
    }
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Add: input shapes must match (broadcasting not supported)");
    }

    Tensor output(a.shape(), DataType::Float32);
    const float* pa = a.data();
    const float* pb = b.data();
    float* po = output.data();
    for (int64_t i = 0; i < a.numel(); ++i) {
        po[i] = pa[i] + pb[i];
    }
    return output;
}

}  // namespace edgert::ops