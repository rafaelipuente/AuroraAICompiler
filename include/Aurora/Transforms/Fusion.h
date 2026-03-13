//===- Fusion.h - Aurora operation fusion passes -----------------*- C++ -*-===//
//
// This file contains the declarations of fusion-related optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_FUSION_H
#define AURORA_TRANSFORMS_FUSION_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createFusionPass();

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_FUSION_H
