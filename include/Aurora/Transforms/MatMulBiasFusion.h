//===- MatMulBiasFusion.h - MatMul+Bias fusion passes ------------*- C++ -*-===//
//
// This file contains the declarations of MatMul and Bias fusion passes.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_MATMUL_BIAS_FUSION_H
#define AURORA_TRANSFORMS_MATMUL_BIAS_FUSION_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createMatMulBiasFusionPass();

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_MATMUL_BIAS_FUSION_H
