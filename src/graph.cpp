#include "edgert/graph.h"

#include <stdexcept>
#include <utility>

namespace edgert {

void Graph::add_input(const std::string& name) {
    input_names_.push_back(name);
}

void Graph::add_node(GraphNode node) {
    nodes_.push_back(std::move(node));
}

Tensor Graph::run(const OperatorRegistry& registry,
                   const std::unordered_map<std::string, Tensor>& inputs,
                   const std::string& output_name) const {
    // `values` is the running "notebook" of every tensor computed (or
    // provided) so far, looked up by name.
    std::unordered_map<std::string, Tensor> values;

    for (const auto& name : input_names_) {
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            throw std::invalid_argument("Graph::run: missing required input: " + name);
        }
        values.emplace(name, it->second);
    }

    for (const GraphNode& node : nodes_) {
        std::vector<const Tensor*> node_inputs;
        node_inputs.reserve(node.inputs.size());
        for (const auto& input_name : node.inputs) {
            auto it = values.find(input_name);
            if (it == values.end()) {
                throw std::invalid_argument(
                    "Graph::run: node '" + node.output + "' (" + node.op_type +
                    ") references input '" + input_name + "' that has not been produced yet");
            }
            node_inputs.push_back(&it->second);
        }

        auto op = registry.create(node.op_type);
        Tensor result = op->compute(node_inputs);
        values.insert_or_assign(node.output, std::move(result));
    }

    auto it = values.find(output_name);
    if (it == values.end()) {
        throw std::out_of_range("Graph::run: output '" + output_name + "' was never produced");
    }
    return std::move(it->second);
}

}  // namespace edgert