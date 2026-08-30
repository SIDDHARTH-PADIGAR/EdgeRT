#include "edgert/ops/linear.h"

#include <gtest/gtest.h>

#include <stdexcept>

#include "edgert/tensor.h"

using edgert::Tensor;
using edgert::ops::LinearOp;

TEST(LinearOpTest, SingleSampleMatchesHandComputation) {
    // x = [1, 2, 3]                      (batch=1, in_features=3)
    // W = [[1, 0, 1],                    (out_features=2, in_features=3)
    //      [0, 1, 1]]
    // b = [10, 20]
    //
    // y[0] = 1*1 + 2*0 + 3*1 + 10 = 14
    // y[1] = 1*0 + 2*1 + 3*1 + 20 = 25
    Tensor x({1, 3});
    float x_vals[] = {1, 2, 3};
    for (int i = 0; i < 3; ++i) x.data()[i] = x_vals[i];

    Tensor w({2, 3});
    float w_vals[] = {1, 0, 1, 0, 1, 1};
    for (int i = 0; i < 6; ++i) w.data()[i] = w_vals[i];

    Tensor b({2});
    b.data()[0] = 10.0F;
    b.data()[1] = 20.0F;

    LinearOp op;
    Tensor y = op.compute({&x, &w, &b});

    ASSERT_EQ(y.shape(), (std::vector<int64_t>{1, 2}));
    EXPECT_FLOAT_EQ(y.data()[0], 14.0F);
    EXPECT_FLOAT_EQ(y.data()[1], 25.0F);
}

TEST(LinearOpTest, BatchOfTwoSamples) {
    Tensor x({2, 2});
    float x_vals[] = {1, 1, 2, 2};  // two samples, in_features=2
    for (int i = 0; i < 4; ++i) x.data()[i] = x_vals[i];

    Tensor w({1, 2});  // out_features=1
    w.data()[0] = 3.0F;
    w.data()[1] = 4.0F;

    Tensor b({1});
    b.data()[0] = 0.0F;

    LinearOp op;
    Tensor y = op.compute({&x, &w, &b});

    ASSERT_EQ(y.shape(), (std::vector<int64_t>{2, 1}));
    EXPECT_FLOAT_EQ(y.data()[0], 1 * 3 + 1 * 4);  // 7
    EXPECT_FLOAT_EQ(y.data()[1], 2 * 3 + 2 * 4);  // 14
}

TEST(LinearOpTest, WrongInputCountThrows) {
    Tensor x({1, 3});
    Tensor w({2, 3});
    LinearOp op;
    EXPECT_THROW(op.compute({&x, &w}), std::invalid_argument);
}

TEST(LinearOpTest, InFeaturesMismatchThrows) {
    Tensor x({1, 3});
    Tensor w({2, 4});  // W expects in_features=4, x has 3
    Tensor b({2});
    LinearOp op;
    EXPECT_THROW(op.compute({&x, &w, &b}), std::invalid_argument);
}

TEST(LinearOpTest, BiasSizeMismatchThrows) {
    Tensor x({1, 3});
    Tensor w({2, 3});
    Tensor b({5});  // should be 2 (out_features), not 5
    LinearOp op;
    EXPECT_THROW(op.compute({&x, &w, &b}), std::invalid_argument);
}

TEST(LinearOpTest, WrongRankThrows) {
    Tensor x({3});  // rank 1, should be rank 2
    Tensor w({2, 3});
    Tensor b({2});
    LinearOp op;
    EXPECT_THROW(op.compute({&x, &w, &b}), std::invalid_argument);
}