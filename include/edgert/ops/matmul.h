#pragma once

#include "edgert/operator.h"

namespace edgert::ops {

// 2D matrix multiplication: given A with shape [M, K] and B with shape
// [K, N], produces C with shape [M, N] where C[i,j] = sum_k A[i,k]*B[k,j].
// Both inputs must be rank-2 Float32 tensors with matching inner dimension.
//
// The inner loops are ordered i -> k -> j (not the more obvious i -> j -> k)
// so that both B and the output are walked row-by-row (stride-1), which is
// what the CPU cache rewards. See src/ops/matmul.cpp for the walkthrough.
// This is still a single scalar loop nest — no blocking/tiling and no
// SIMD yet, both of which are later, separate optimizations.
class MatMulOp : public Operator {
public:
    std::string name() const override { return "MatMul"; }
    Tensor compute(const std::vector<const Tensor*>& inputs) const override;
};

}  // namespace edgert::ops