//===- Fusion.h - Aurora operation fusion passes -----------------*- C++ -*-===//
//
// This file contains the declarations of fusion-related optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_FUSION_H
#define AURORA_TRANSFORMS_FUSION_H

// Include MLIR headers first to avoid namespace issues
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createFusionPass();

class FusionPass : public PassWrapper<FusionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)
  
  FusionPass() = default;
  FusionPass(const FusionPass &) = default;
  
  void runOnOperation() override;
  
  StringRef getName() const override { return "FusionPass"; }
  StringRef getArgument() const final { return "aurora-fusion"; }
  StringRef getDescription() const final { return "Fusion optimization pass"; }
};

// Forward declarations

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_FUSION_H
