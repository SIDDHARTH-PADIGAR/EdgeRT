#include "edgert/ops/matmul.h"

#include <gtest/gtest.h>

#include <stdexcept>

#include "edgert/tensor.h"

using edgert::Tensor;
using edgert::ops::MatMulOp;

TEST(MatMulOpTest, TwoByThreeTimesThreeByTwo) {
    // A = [[1, 2, 3],      B = [[7,  8],
    //      [4, 5, 6]]           [9,  10],
    //                           [11, 12]]
    Tensor a({2, 3});
    float a_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) a.data()[i] = a_vals[i];

    Tensor b({3, 2});
    float b_vals[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; ++i) b.data()[i] = b_vals[i];

    MatMulOp op;
    Tensor c = op.compute({&a, &b});

    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 2}));
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    EXPECT_FLOAT_EQ(c.data()[0], 58.0F);
    EXPECT_FLOAT_EQ(c.data()[1], 64.0F);
    EXPECT_FLOAT_EQ(c.data()[2], 139.0F);
    EXPECT_FLOAT_EQ(c.data()[3], 154.0F);
}

TEST(MatMulOpTest, IdentityMatrixIsNoOp) {
    Tensor a({2, 2});
    a.data()[0] = 3.0F;
    a.data()[1] = 5.0F;
    a.data()[2] = -1.0F;
    a.data()[3] = 2.0F;

    Tensor identity({2, 2});
    identity.data()[0] = 1.0F;
    identity.data()[1] = 0.0F;
    identity.data()[2] = 0.0F;
    identity.data()[3] = 1.0F;

    MatMulOp op;
    Tensor c = op.compute({&a, &identity});

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], a.data()[i]);
    }
}

TEST(MatMulOpTest, InnerDimensionMismatchThrows) {
    Tensor a({2, 3});
    Tensor b({4, 2});  // b's rows (4) don't match a's cols (3)
    MatMulOp op;
    EXPECT_THROW(op.compute({&a, &b}), std::invalid_argument);
}

TEST(MatMulOpTest, WrongRankThrows) {
    Tensor a({6});      // rank 1, not a matrix
    Tensor b({3, 2});
    MatMulOp op;
    EXPECT_THROW(op.compute({&a, &b}), std::invalid_argument);
}

TEST(MatMulOpTest, ZeroInnerDimensionProducesZeroOutput) {
    // A is [2, 0], B is [0, 3]: the reduction dimension is empty, so every
    // output element is a sum over zero terms, i.e. 0.
    Tensor a({2, 0});
    Tensor b({0, 3});
    MatMulOp op;
    Tensor c = op.compute({&a, &b});

    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    for (int64_t i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 0.0F);
    }
}

TEST(MatMulOpTest, WideRowExercisesVectorizedAndRemainderPaths) {
    // A: [1, 4] of all 1s, B: [4, 10] of all 1s -> every output element is
    // sum of 4 ones = 4. N=10 is greater than 8 and not a multiple of 8,
    // so this exercises both the width-8 vectorized loop and the scalar
    // remainder cleanup, on whichever kernel the CPU dispatches to.
    Tensor a({1, 4});
    a.fill(1.0F);
    Tensor b({4, 10});
    b.fill(1.0F);

    MatMulOp op;
    Tensor c = op.compute({&a, &b});

    ASSERT_EQ(c.shape(), (std::vector<int64_t>{1, 10}));
    for (int64_t i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 4.0F);
    }
}

TEST(MatMulOpTest, MatchesReferenceOnNonTrivialSizes) {
    // Cross-checks MatMulOp's output against a brute-force reference
    // computed independently right here, using varied (non-constant)
    // input values so indexing bugs can't hide behind the symmetry of
    // all-ones or all-zeros test data. N=13 is greater than 8 and not a
    // multiple of 8, exercising the same vectorized + remainder split as
    // the test above but with numbers that would expose an off-by-one.
    constexpr int64_t kM = 3;
    constexpr int64_t kK = 5;
    constexpr int64_t kN = 13;
    Tensor a({kM, kK});
    Tensor b({kK, kN});
    for (int64_t i = 0; i < a.numel(); ++i) {
        a.data()[i] = static_cast<float>((i % 7) - 3);
    }
    for (int64_t i = 0; i < b.numel(); ++i) {
        b.data()[i] = static_cast<float>((i % 5) - 2);
    }

    MatMulOp op;
    Tensor c = op.compute({&a, &b});

    std::vector<float> expected(static_cast<std::size_t>(kM * kN), 0.0F);
    for (int64_t i = 0; i < kM; ++i) {
        for (int64_t j = 0; j < kN; ++j) {
            float sum = 0.0F;
            for (int64_t p = 0; p < kK; ++p) {
                sum += a.data()[i * kK + p] * b.data()[p * kN + j];
            }
            expected[static_cast<std::size_t>(i * kN + j)] = sum;
        }
    }

    for (int64_t idx = 0; idx < kM * kN; ++idx) {
        EXPECT_FLOAT_EQ(c.data()[idx], expected[static_cast<std::size_t>(idx)]);
    }
}

TEST(MatMulOpTest, WrongInputCountThrows) {
    Tensor a({2, 2});
    MatMulOp op;
    EXPECT_THROW(op.compute({&a}), std::invalid_argument);
}