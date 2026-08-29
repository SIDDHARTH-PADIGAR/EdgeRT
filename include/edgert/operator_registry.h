#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "edgert/operator.h"

namespace edgert {

// OperatorRegistry maps operator names (as they appear in a model file) to
// factory functions that construct the corresponding Operator. This is the
// indirection the model loader will use later: an op type string like
// "Conv2D" read out of a model file resolves to a registered ConvOp
// instance without the loader knowing about concrete operator classes.
class OperatorRegistry {
public:
    using Factory = std::function<std::unique_ptr<Operator>()>;

    OperatorRegistry() = default;
    OperatorRegistry(const OperatorRegistry&) = delete;
    OperatorRegistry& operator=(const OperatorRegistry&) = delete;

    // Registers a factory under `name`. Throws std::invalid_argument if
    // `name` is already registered.
    void register_op(const std::string& name, Factory factory);

    bool contains(const std::string& name) const;

    // Constructs a new Operator instance for `name`. Throws
    // std::invalid_argument if `name` is not registered.
    std::unique_ptr<Operator> create(const std::string& name) const;

private:
    std::unordered_map<std::string, Factory> factories_;
};

}  // namespace edgert