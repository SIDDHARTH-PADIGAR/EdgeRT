#include "edgert/graph.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <unordered_map>

#include "edgert/builtin_ops.h"
#include "edgert/operator_registry.h"
#include "edgert/tensor.h"

using edgert::Graph;
using edgert::GraphNode;
using edgert::OperatorRegistry;
using edgert::Tensor;

namespace {

OperatorRegistry make_registry() {
    OperatorRegistry registry;
    edgert::register_builtin_operators(registry);
    return registry;
}

}  // namespace

TEST(GraphTest, SingleNodeRelu) {
    Graph graph;
    graph.add_input("x");
    graph.add_node(GraphNode{"Relu", {"x"}, "y"});

    Tensor x({3});
    x.data()[0] = -1.0F;
    x.data()[1] = 0.0F;
    x.data()[2] = 5.0F;

    OperatorRegistry registry = make_registry();
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("x", std::move(x));

    Tensor y = graph.run(registry, inputs, "y");
    EXPECT_FLOAT_EQ(y.data()[0], 0.0F);
    EXPECT_FLOAT_EQ(y.data()[1], 0.0F);
    EXPECT_FLOAT_EQ(y.data()[2], 5.0F);
}

TEST(GraphTest, ChainedTwoNodeGraph) {
    // out = Relu(Add(a, b))
    Graph graph;
    graph.add_input("a");
    graph.add_input("b");
    graph.add_node(GraphNode{"Add", {"a", "b"}, "sum"});
    graph.add_node(GraphNode{"Relu", {"sum"}, "out"});

    Tensor a({2});
    Tensor b({2});
    a.fill(2.0F);
    b.fill(-5.0F);

    OperatorRegistry registry = make_registry();
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", std::move(a));
    inputs.emplace("b", std::move(b));

    Tensor out = graph.run(registry, inputs, "out");
    EXPECT_FLOAT_EQ(out.data()[0], 0.0F);
    EXPECT_FLOAT_EQ(out.data()[1], 0.0F);
}

TEST(GraphTest, MissingGraphInputThrows) {
    Graph graph;
    graph.add_input("x");
    graph.add_node(GraphNode{"Relu", {"x"}, "y"});

    OperatorRegistry registry = make_registry();
    std::unordered_map<std::string, Tensor> inputs;  // "x" not provided

    EXPECT_THROW(graph.run(registry, inputs, "y"), std::invalid_argument);
}

TEST(GraphTest, NodeReferencingUnproducedTensorThrows) {
    Graph graph;
    graph.add_input("x");
    // "typo" was never declared as an input nor produced by an earlier node.
    graph.add_node(GraphNode{"Relu", {"typo"}, "y"});

    OperatorRegistry registry = make_registry();
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("x", Tensor({1}));

    EXPECT_THROW(graph.run(registry, inputs, "y"), std::invalid_argument);
}

TEST(GraphTest, UnknownOperatorThrows) {
    Graph graph;
    graph.add_input("x");
    graph.add_node(GraphNode{"NotARealOp", {"x"}, "y"});

    OperatorRegistry registry = make_registry();
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("x", Tensor({1}));

    EXPECT_THROW(graph.run(registry, inputs, "y"), std::invalid_argument);
}

TEST(GraphTest, UnproducedOutputNameThrows) {
    Graph graph;
    graph.add_input("x");
    graph.add_node(GraphNode{"Relu", {"x"}, "y"});

    OperatorRegistry registry = make_registry();
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("x", Tensor({1}));

    EXPECT_THROW(graph.run(registry, inputs, "does_not_exist"), std::out_of_range);
}