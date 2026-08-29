#pragma once

#include "edgert/operator.h"

namespace edgert::ops {

// Elementwise addition of two tensors: output[i] = a[i] + b[i].
// Requires both inputs to have identical shape and DataType::Float32.
// Broadcasting is not yet supported.
class AddOp : public Operator {
public:
    std::string name() const override { return "Add"; }
    Tensor compute(const std::vector<const Tensor*>& inputs) const override;
};

}  // namespace edgert::ops