#pragma once

#include "edgert/operator.h"

namespace edgert::ops {

// 2D matrix multiplication: given A with shape [M, K] and B with shape
// [K, N], produces C with shape [M, N] where C[i,j] = sum_k A[i,k]*B[k,j].
// Both inputs must be rank-2 Float32 tensors with matching inner dimension.
//
// The loop order is i -> k -> j for cache-friendly row-major access (see
// src/ops/matmul.cpp). On top of that, if the CPU supports AVX2+FMA at
// runtime, the innermost loop is vectorized to process 8 floats at a time;
// otherwise it transparently falls back to a scalar loop. Either way this
// produces the same result — the dispatch is an internal implementation
// detail, not something callers need to know about.
class MatMulOp : public Operator {
public:
    std::string name() const override { return "MatMul"; }
    Tensor compute(const std::vector<const Tensor*>& inputs) const override;
};

}  // namespace edgert::ops