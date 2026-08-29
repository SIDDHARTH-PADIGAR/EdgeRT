#include <gtest/gtest.h>

#include <stdexcept>

#include "edgert/builtin_ops.h"
#include "edgert/operator_registry.h"
#include "edgert/ops/add.h"
#include "edgert/ops/relu.h"
#include "edgert/tensor.h"

using edgert::OperatorRegistry;
using edgert::Tensor;
using edgert::ops::AddOp;
using edgert::ops::ReluOp;

TEST(AddOpTest, ElementwiseAddition) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(1.0F);
    b.fill(2.0F);

    AddOp op;
    Tensor out = op.compute({&a, &b});

    EXPECT_EQ(out.shape(), a.shape());
    for (int64_t i = 0; i < out.numel(); ++i) {
        EXPECT_FLOAT_EQ(out.data()[i], 3.0F);
    }
}

TEST(AddOpTest, WrongInputCountThrows) {
    Tensor a({2, 2});
    AddOp op;
    EXPECT_THROW(op.compute({&a}), std::invalid_argument);
}

TEST(AddOpTest, ShapeMismatchThrows) {
    Tensor a({2, 2});
    Tensor b({3});
    AddOp op;
    EXPECT_THROW(op.compute({&a, &b}), std::invalid_argument);
}

TEST(ReluOpTest, ClampsNegativeValues) {
    Tensor x({4});
    x.data()[0] = -1.0F;
    x.data()[1] = 0.0F;
    x.data()[2] = 2.5F;
    x.data()[3] = -0.001F;

    ReluOp op;
    Tensor out = op.compute({&x});

    EXPECT_FLOAT_EQ(out.data()[0], 0.0F);
    EXPECT_FLOAT_EQ(out.data()[1], 0.0F);
    EXPECT_FLOAT_EQ(out.data()[2], 2.5F);
    EXPECT_FLOAT_EQ(out.data()[3], 0.0F);
}

TEST(ReluOpTest, WrongInputCountThrows) {
    Tensor a({2});
    Tensor b({2});
    ReluOp op;
    EXPECT_THROW(op.compute({&a, &b}), std::invalid_argument);
}

TEST(BuiltinOpsTest, RegistersAddAndRelu) {
    OperatorRegistry registry;
    edgert::register_builtin_operators(registry);
    EXPECT_TRUE(registry.contains("Add"));
    EXPECT_TRUE(registry.contains("Relu"));

    auto add = registry.create("Add");
    Tensor a({2});
    Tensor b({2});
    a.fill(1.0F);
    b.fill(1.0F);
    Tensor out = add->compute({&a, &b});
    EXPECT_FLOAT_EQ(out.data()[0], 2.0F);
}