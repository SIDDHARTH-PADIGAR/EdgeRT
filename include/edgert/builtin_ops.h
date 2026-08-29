#pragma once

#include "edgert/operator_registry.h"

namespace edgert {

// Registers every built-in CPU operator (Add, Relu, ...) into `registry`.
// Call this once before resolving operators by name (e.g. during model
// loading). Kept as an explicit registration function rather than
// self-registering static globals: self-registration in separate
// translation units gets silently dropped by the linker when those
// translation units live in a static library and nothing else references
// their symbols, which is a real footgun. Explicit registration has no
// such surprise.
void register_builtin_operators(OperatorRegistry& registry);

}  // namespace edgert