#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "edgert/operator_registry.h"
#include "edgert/tensor.h"

namespace edgert {

// A single computation step in a Graph: run the operator named `op_type`
// (looked up in the OperatorRegistry at execution time) on the tensors
// named `inputs`, and store the result under the name `output`.
struct GraphNode {
    std::string op_type;
    std::vector<std::string> inputs;
    std::string output;
};

// Graph is an ordered list of GraphNodes plus a declared set of graph-level
// input names. Nodes are executed in the order they were added, so callers
// must add them in a valid dependency order (a node's inputs must already
// be graph inputs or the output of an earlier node). Automatic
// dependency-based ordering is a later milestone.
class Graph {
public:
    // Declares an input name the graph expects to be provided at run()
    // time. Order of add_input() calls does not matter.
    void add_input(const std::string& name);

    // Appends a node to the end of the execution order.
    void add_node(GraphNode node);

    // Runs every node in insertion order. `inputs` must contain a Tensor
    // for every name registered with add_input(). Returns the tensor bound
    // to `output_name` after execution.
    //
    // Throws std::invalid_argument for: a missing graph input, a node
    // referencing a tensor name that has not been produced yet, or an
    // unregistered operator name. Throws std::out_of_range if
    // `output_name` was never produced by any node.
    Tensor run(const OperatorRegistry& registry,
               const std::unordered_map<std::string, Tensor>& inputs,
               const std::string& output_name) const;

private:
    std::vector<std::string> input_names_;
    std::vector<GraphNode> nodes_;
};

}  // namespace edgert