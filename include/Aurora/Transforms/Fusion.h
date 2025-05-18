#ifndef AURORA_TRANSFORMS_FUSION_H
#define AURORA_TRANSFORMS_FUSION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace aurora {

std::unique_ptr<Pass> createOperationFusionPass();

#define GEN_PASS_REGISTRATION
#include "Aurora/Transforms/Passes.h.inc"

} // namespace aurora
} // namespace mlir

#endif // AURORA_TRANSFORMS_FUSION_H
