//===- Passes.h - Aurora Conversion Passes -----------------------*- C++ -*-===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_CONVERSION_PASSES_H
#define AURORA_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createConvertAuroraToLinalgPass();

#define GEN_PASS_REGISTRATION
#include "Aurora/Conversion/Passes.h.inc"

} // namespace aurora
} // namespace mlir

#endif // AURORA_CONVERSION_PASSES_H
