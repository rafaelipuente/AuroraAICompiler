//===- AuroraOps.h - Aurora dialect operations ----------------*- C++ -*-===//
//
// This file contains the declaration of the Aurora dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_OPS_H
#define AURORA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Aurora/Dialect/Aurora/AuroraDialect.h"

namespace mlir {
namespace aurora {

// Forward declarations for all operations
class AddOp;
class ConvOp;
class FusedAttentionOp;
class LayerNormOp;
class MatMulOp;
class MatMulBiasOp;
class ReluOp;

} // namespace aurora
} // namespace mlir

// Include TableGen-generated declarations
#define GET_OP_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.h.inc"

#endif // AURORA_OPS_H
