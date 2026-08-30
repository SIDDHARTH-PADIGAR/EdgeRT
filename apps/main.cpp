#include <iostream>
#include <unordered_map>

#include "edgert/builtin_ops.h"
#include "edgert/graph.h"
#include "edgert/operator_registry.h"
#include "edgert/tensor.h"

int main() {
    edgert::Tensor t({2, 3}, edgert::DataType::Float32);
    t.fill(1.0F);
    t.at({0, 0}) = 42.0F;

    std::cout << "EdgeRT demo\n";
    std::cout << "shape: [";
    for (std::size_t i = 0; i < t.shape().size(); ++i) {
        std::cout << t.shape()[i];
        if (i + 1 < t.shape().size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "numel: " << t.numel() << "\n";
    std::cout << "nbytes: " << t.nbytes() << "\n";
    std::cout << "t[0,0] = " << t.at({0, 0}) << "\n";
    std::cout << "t[1,2] = " << t.at({1, 2}) << "\n";

    std::cout << "\nOperator registry demo\n";
    edgert::OperatorRegistry registry;
    edgert::register_builtin_operators(registry);

    edgert::Tensor a({4});
    edgert::Tensor b({4});
    a.fill(2.0F);
    b.fill(-5.0F);

    auto add_op = registry.create("Add");
    edgert::Tensor sum = add_op->compute({&a, &b});

    auto relu_op = registry.create("Relu");
    edgert::Tensor relu_out = relu_op->compute({&sum});

    std::cout << "a = 2, b = -5, Add(a,b) then Relu -> [";
    for (int64_t i = 0; i < relu_out.numel(); ++i) {
        std::cout << relu_out.data()[i];
        if (i + 1 < relu_out.numel()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "\nGraph demo\n";
    edgert::Graph graph;
    graph.add_input("gx");
    graph.add_input("gy");
    graph.add_node(edgert::GraphNode{"Add", {"gx", "gy"}, "gsum"});
    graph.add_node(edgert::GraphNode{"Relu", {"gsum"}, "gout"});

    edgert::Tensor gx({4});
    edgert::Tensor gy({4});
    gx.fill(2.0F);
    gy.fill(-5.0F);

    std::unordered_map<std::string, edgert::Tensor> graph_inputs;
    graph_inputs.emplace("gx", std::move(gx));
    graph_inputs.emplace("gy", std::move(gy));

    edgert::Tensor graph_out = graph.run(registry, graph_inputs, "gout");
    std::cout << "Graph(Add -> Relu) with gx=2, gy=-5 -> [";
    for (int64_t i = 0; i < graph_out.numel(); ++i) {
        std::cout << graph_out.data()[i];
        if (i + 1 < graph_out.numel()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    return 0;
}