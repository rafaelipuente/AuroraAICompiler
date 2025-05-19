//===- Passes.h - Aurora Transformation Passes ------------------------*- C++ -*-===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Aurora 
// transformations.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_PASSES_H
#define AURORA_TRANSFORMS_PASSES_H

#include "Aurora/Transforms/Fusion.h"
#include "Aurora/Transforms/MatMulBiasFusion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace aurora {

// Generate the code for registering passes
#define GEN_PASS_REGISTRATION
#ifndef AURORA_TRANSFORMS_PASSES_H_INC
// The actual implementation will be provided separately
#define AURORA_TRANSFORMS_PASSES_H_INC
#endif // AURORA_TRANSFORMS_PASSES_H_INC

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_PASSES_H
