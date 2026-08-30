#include "edgert/bench_utils.h"

#include <gtest/gtest.h>

#include <vector>

using edgert::bench::median_value;
using edgert::bench::min_value;

TEST(BenchUtilsTest, MinValueFindsSmallest) {
    std::vector<double> values{5.0, 1.5, 3.2, 9.9, 1.5};
    EXPECT_DOUBLE_EQ(min_value(values), 1.5);
}

TEST(BenchUtilsTest, MedianValueOddCount) {
    std::vector<double> values{5.0, 1.0, 3.0};
    EXPECT_DOUBLE_EQ(median_value(values), 3.0);
}

TEST(BenchUtilsTest, MedianValueEvenCount) {
    std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(median_value(values), 2.5);
}

TEST(BenchUtilsTest, MedianValueDoesNotMutateCallerVectorOrder) {
    // median_value takes its argument by value, so sorting it internally
    // must not be observable from the caller's copy.
    std::vector<double> values{9.0, 1.0, 5.0};
    std::vector<double> before = values;
    median_value(values);
    EXPECT_EQ(values, before);
}