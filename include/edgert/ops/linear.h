#pragma once

#include "edgert/operator.h"

namespace edgert::ops {

// Fully-connected / dense layer: y = x @ W^T + b.
//
// Shapes (PyTorch nn.Linear convention):
//   x: [batch, in_features]
//   W: [out_features, in_features]
//   b: [out_features]
//   y: [batch, out_features]
//
// W is stored [out_features, in_features] (not [in_features, out_features])
// so no separate Transpose step is needed; this op reads W as if
// transposed internally.
class LinearOp : public Operator {
public:
    std::string name() const override { return "Linear"; }
    Tensor compute(const std::vector<const Tensor*>& inputs) const override;
};

}  // namespace edgert::ops