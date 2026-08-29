#include "edgert/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using edgert::DataType;
using edgert::Tensor;

TEST(TensorTest, ConstructsWithCorrectShapeAndNumel) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.dtype(), DataType::Float32);
    EXPECT_EQ(t.nbytes(), 24 * sizeof(float));
}

TEST(TensorTest, DefaultConstructedTensorIsEmpty) {
    Tensor t;
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.nbytes(), 0u);
    EXPECT_EQ(t.data(), nullptr);
}

TEST(TensorTest, ZeroSizedDimensionProducesEmptyBuffer) {
    Tensor t({0, 5});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.data(), nullptr);
}

TEST(TensorTest, NegativeDimensionThrows) {
    EXPECT_THROW(Tensor({-1, 2}), std::invalid_argument);
}

TEST(TensorTest, FillAndAtRoundTrip) {
    Tensor t({2, 2});
    t.fill(3.5F);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 3.5F);
        }
    }
    t.at({1, 1}) = 9.0F;
    EXPECT_FLOAT_EQ(t.at({1, 1}), 9.0F);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 3.5F);
}

TEST(TensorTest, AtRejectsOutOfRangeIndex) {
    Tensor t({2, 2});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0}), std::out_of_range);
}

TEST(TensorTest, BufferIs64ByteAligned) {
    Tensor t({100});
    auto addr = reinterpret_cast<std::uintptr_t>(t.data());
    EXPECT_EQ(addr % 64, 0u);
}

TEST(TensorTest, CopyConstructorDeepCopies) {
    Tensor a({2, 2});
    a.fill(1.0F);
    Tensor b(a);
    b.at({0, 0}) = 99.0F;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 1.0F);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 99.0F);
    EXPECT_NE(a.data(), b.data());
}

TEST(TensorTest, CopyAssignmentDeepCopies) {
    Tensor a({2, 2});
    a.fill(2.0F);
    Tensor b({3});
    b = a;
    EXPECT_EQ(b.shape(), a.shape());
    b.at({0, 0}) = 5.0F;
    EXPECT_FLOAT_EQ(a.at({0, 0}), 2.0F);
}

TEST(TensorTest, MoveConstructorTransfersOwnership) {
    Tensor a({2, 2});
    a.fill(7.0F);
    float* original_ptr = a.data();
    Tensor b(std::move(a));
    EXPECT_EQ(b.data(), original_ptr);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 7.0F);
    EXPECT_EQ(a.numel(), 0);
    EXPECT_EQ(a.data(), nullptr);
}

TEST(TensorTest, MoveAssignmentTransfersOwnership) {
    Tensor a({2, 2});
    a.fill(4.0F);
    Tensor b({5});
    b = std::move(a);
    EXPECT_FLOAT_EQ(b.at({0, 0}), 4.0F);
    EXPECT_EQ(a.numel(), 0);
}

TEST(TensorTest, StridesAreRowMajor) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.strides(), (std::vector<int64_t>{12, 4, 1}));
}