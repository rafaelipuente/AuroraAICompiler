//===- MatMulBiasFusion.h - MatMul+Bias fusion passes ------------*- C++ -*-===//
//
// This file contains the declarations of MatMul and Bias fusion optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_MATMUL_BIAS_FUSION_H
#define AURORA_TRANSFORMS_MATMUL_BIAS_FUSION_H

// Include MLIR headers first to avoid namespace issues
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createMatMulBiasFusionPass();

class MatMulBiasFusionPass : public PassWrapper<MatMulBiasFusionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulBiasFusionPass)
  
  MatMulBiasFusionPass() = default;
  MatMulBiasFusionPass(const MatMulBiasFusionPass &) = default;
  
  void runOnOperation() override;

  StringRef getName() const override { return "MatMulBiasFusionPass"; }
  StringRef getArgument() const final { return "aurora-matmul-bias-fusion"; }
  StringRef getDescription() const final { return "MatMul and Bias fusion optimization pass"; }
};

// Forward declarations

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_MATMUL_BIAS_FUSION_H
