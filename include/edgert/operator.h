#pragma once

#include <string>
#include <vector>

#include "edgert/tensor.h"

namespace edgert {

// Operator is the interface every CPU kernel implements. An Operator is
// stateless with respect to tensor data: compute() takes input tensors and
// returns a freshly allocated output tensor. Shape and dtype validation is
// the operator's responsibility and must happen before any computation.
class Operator {
public:
    virtual ~Operator() = default;

    // Human-readable operator name, e.g. "Add", "Relu". Must match the
    // name it is registered under in the OperatorRegistry.
    virtual std::string name() const = 0;

    // Executes the operator. Throws std::invalid_argument if the number
    // of inputs, their shapes, or their dtypes are not valid for this
    // operator.
    virtual Tensor compute(const std::vector<const Tensor*>& inputs) const = 0;
};

}  // namespace edgert