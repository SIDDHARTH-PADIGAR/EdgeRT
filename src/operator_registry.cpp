#include "edgert/operator_registry.h"

#include <stdexcept>
#include <utility>

namespace edgert {

void OperatorRegistry::register_op(const std::string& name, Factory factory) {
    if (factories_.count(name) != 0) {
        throw std::invalid_argument("OperatorRegistry: operator already registered: " + name);
    }
    factories_[name] = std::move(factory);
}

bool OperatorRegistry::contains(const std::string& name) const {
    return factories_.count(name) != 0;
}

std::unique_ptr<Operator> OperatorRegistry::create(const std::string& name) const {
    auto it = factories_.find(name);
    if (it == factories_.end()) {
        throw std::invalid_argument("OperatorRegistry: unknown operator: " + name);
    }
    return it->second();
}

}  // namespace edgert