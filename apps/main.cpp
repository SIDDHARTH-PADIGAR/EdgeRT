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

    std::cout << "\nMatMul demo\n";
    edgert::Tensor ma({2, 3});
    float ma_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) ma.data()[i] = ma_vals[i];

    edgert::Tensor mb({3, 2});
    float mb_vals[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; ++i) mb.data()[i] = mb_vals[i];

    auto matmul_op = registry.create("MatMul");
    edgert::Tensor mc = matmul_op->compute({&ma, &mb});
    std::cout << "[2,3] x [3,2] -> shape [" << mc.shape()[0] << ", " << mc.shape()[1] << "], values [";
    for (int64_t i = 0; i < mc.numel(); ++i) {
        std::cout << mc.data()[i];
        if (i + 1 < mc.numel()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "\nTiny MLP demo (Linear -> Relu -> Linear via Graph)\n";
    edgert::Graph mlp;
    mlp.add_input("x");
    mlp.add_input("w1");
    mlp.add_input("b1");
    mlp.add_input("w2");
    mlp.add_input("b2");
    mlp.add_node(edgert::GraphNode{"Linear", {"x", "w1", "b1"}, "h1"});
    mlp.add_node(edgert::GraphNode{"Relu", {"h1"}, "h1_relu"});
    mlp.add_node(edgert::GraphNode{"Linear", {"h1_relu", "w2", "b2"}, "out"});

    edgert::Tensor x({1, 3});
    float x_vals[] = {1.0F, -2.0F, 3.0F};
    for (int i = 0; i < 3; ++i) x.data()[i] = x_vals[i];

    edgert::Tensor w1({4, 3});
    w1.fill(0.5F);
    edgert::Tensor b1({4});
    b1.fill(0.1F);
    edgert::Tensor w2({2, 4});
    w2.fill(-0.5F);
    edgert::Tensor b2({2});
    b2.fill(0.2F);

    std::unordered_map<std::string, edgert::Tensor> mlp_inputs;
    mlp_inputs.emplace("x", std::move(x));
    mlp_inputs.emplace("w1", std::move(w1));
    mlp_inputs.emplace("b1", std::move(b1));
    mlp_inputs.emplace("w2", std::move(w2));
    mlp_inputs.emplace("b2", std::move(b2));

    edgert::Tensor mlp_out = mlp.run(registry, mlp_inputs, "out");
    std::cout << "x=[1,-2,3] through Linear(3->4) -> Relu -> Linear(4->2) -> [";
    for (int64_t i = 0; i < mlp_out.numel(); ++i) {
        std::cout << mlp_out.data()[i];
        if (i + 1 < mlp_out.numel()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    return 0;
}