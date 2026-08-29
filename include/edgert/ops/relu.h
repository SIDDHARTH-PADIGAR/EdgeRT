#pragma once

#include "edgert/operator.h"

namespace edgert::ops {

// Elementwise ReLU: output[i] = max(0, input[i]).
// Requires exactly one Float32 input.
class ReluOp : public Operator {
public:
    std::string name() const override { return "Relu"; }
    Tensor compute(const std::vector<const Tensor*>& inputs) const override;
};

}  // namespace edgert::ops