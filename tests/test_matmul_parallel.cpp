#include "edgert/ops/matmul.h"

#include <gtest/gtest.h>

#include "edgert/tensor.h"
#include "edgert/thread_pool.h"

using edgert::Tensor;
using edgert::ThreadPool;
using edgert::ops::MatMulOp;
using edgert::ops::matmul_parallel;

TEST(MatMulParallelTest, MatchesHandComputedValues) {
    Tensor a({2, 3});
    float a_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) a.data()[i] = a_vals[i];
    Tensor b({3, 2});
    float b_vals[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; ++i) b.data()[i] = b_vals[i];

    ThreadPool pool(4);
    Tensor c = matmul_parallel(a, b, pool);

    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 58.0F);
    EXPECT_FLOAT_EQ(c.data()[1], 64.0F);
    EXPECT_FLOAT_EQ(c.data()[2], 139.0F);
    EXPECT_FLOAT_EQ(c.data()[3], 154.0F);
}

TEST(MatMulParallelTest, MatchesSingleThreadedOnAwkwardSizes) {
    constexpr int64_t kM = 37;
    constexpr int64_t kK = 21;
    constexpr int64_t kN = 19;
    Tensor a({kM, kK});
    Tensor b({kK, kN});
    for (int64_t i = 0; i < a.numel(); ++i) {
        a.data()[i] = static_cast<float>((i % 11) - 5);
    }
    for (int64_t i = 0; i < b.numel(); ++i) {
        b.data()[i] = static_cast<float>((i % 9) - 4);
    }

    MatMulOp op;
    Tensor expected = op.compute({&a, &b});

    ThreadPool pool(4);
    Tensor actual = matmul_parallel(a, b, pool);

    ASSERT_EQ(actual.shape(), expected.shape());
    for (int64_t i = 0; i < actual.numel(); ++i) {
        EXPECT_FLOAT_EQ(actual.data()[i], expected.data()[i]);
    }
}

TEST(MatMulParallelTest, MoreThreadsThanRowsStillCorrect) {
    Tensor a({2, 4});
    a.fill(1.0F);
    Tensor b({4, 4});
    b.fill(1.0F);

    ThreadPool pool(8);
    Tensor c = matmul_parallel(a, b, pool);

    for (int64_t i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 4.0F);
    }
}