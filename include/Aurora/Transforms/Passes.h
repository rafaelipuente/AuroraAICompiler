//===- Passes.h - Aurora Transformation Passes --------------------*- C++ -*-===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_PASSES_H
#define AURORA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createFusionPass();
std::unique_ptr<Pass> createMatMulBiasFusionPass();

#define GEN_PASS_REGISTRATION
#include "Aurora/Transforms/Passes.h.inc"

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_PASSES_H
