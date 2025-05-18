#ifndef AURORA_DIALECT_AURORA_OPS_H
#define AURORA_DIALECT_AURORA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace aurora {

#define GET_OP_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.h.inc"

} // namespace aurora
} // namespace mlir

#endif // AURORA_DIALECT_AURORA_OPS_H
