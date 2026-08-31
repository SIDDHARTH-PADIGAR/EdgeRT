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

TEST(LinearOpTest, WideInFeaturesExercisesVectorizedAndRemainderPaths) {
    // in_features=11 is greater than 8 and not a multiple of 8, so this
    // exercises both the width-8 vectorized dot-product loop and the
    // scalar remainder cleanup, on whichever kernel the CPU dispatches to.
    Tensor x({1, 11});
    x.fill(1.0F);
    Tensor w({1, 11});
    w.fill(1.0F);
    Tensor b({1});
    b.data()[0] = 100.0F;

    LinearOp op;
    Tensor y = op.compute({&x, &w, &b});

    ASSERT_EQ(y.shape(), (std::vector<int64_t>{1, 1}));
    // 11 ones dotted with 11 ones = 11, plus bias 100 = 111.
    EXPECT_FLOAT_EQ(y.data()[0], 111.0F);
}

TEST(LinearOpTest, MatchesReferenceOnNonTrivialSizes) {
    // Cross-checks LinearOp's output against a brute-force reference
    // computed independently right here, using varied (non-constant)
    // input values so indexing bugs can't hide behind the symmetry of
    // all-ones test data. in_features=13 is greater than 8 and not a
    // multiple of 8, exercising the same vectorized + remainder split as
    // the test above but with numbers that would expose an off-by-one.
    constexpr int64_t kBatch = 3;
    constexpr int64_t kIn = 13;
    constexpr int64_t kOut = 4;
    Tensor x({kBatch, kIn});
    Tensor w({kOut, kIn});
    Tensor b({kOut});
    for (int64_t i = 0; i < x.numel(); ++i) {
        x.data()[i] = static_cast<float>((i % 7) - 3);
    }
    for (int64_t i = 0; i < w.numel(); ++i) {
        w.data()[i] = static_cast<float>((i % 5) - 2);
    }
    for (int64_t i = 0; i < kOut; ++i) {
        b.data()[i] = static_cast<float>(i);
    }

    LinearOp op;
    Tensor y = op.compute({&x, &w, &b});

    std::vector<float> expected(static_cast<std::size_t>(kBatch * kOut), 0.0F);
    for (int64_t n = 0; n < kBatch; ++n) {
        for (int64_t o = 0; o < kOut; ++o) {
            float sum = b.data()[o];
            for (int64_t i = 0; i < kIn; ++i) {
                sum += x.data()[n * kIn + i] * w.data()[o * kIn + i];
            }
            expected[static_cast<std::size_t>(n * kOut + o)] = sum;
        }
    }

    for (int64_t idx = 0; idx < kBatch * kOut; ++idx) {
        EXPECT_FLOAT_EQ(y.data()[idx], expected[static_cast<std::size_t>(idx)]);
    }
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