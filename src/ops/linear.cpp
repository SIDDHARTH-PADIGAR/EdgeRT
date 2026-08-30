#include "edgert/ops/linear.h"

#include <stdexcept>
#include <string>

namespace edgert::ops {

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

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_features; ++o) {
            float sum = pb[o];
            for (int64_t i = 0; i < in_features; ++i) {
                sum += px[n * in_features + i] * pw[o * in_features + i];
            }
            po[n * out_features + o] = sum;
        }
    }
    return output;
}

}  // namespace edgert::ops