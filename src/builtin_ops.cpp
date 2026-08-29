#include "edgert/builtin_ops.h"

#include <memory>

#include "edgert/ops/add.h"
#include "edgert/ops/relu.h"

namespace edgert {

void register_builtin_operators(OperatorRegistry& registry) {
    registry.register_op("Add", [] { return std::make_unique<ops::AddOp>(); });
    registry.register_op("Relu", [] { return std::make_unique<ops::ReluOp>(); });
}

}  // namespace edgert