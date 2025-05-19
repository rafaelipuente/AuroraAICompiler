//===- AuroraDialect.h - Aurora dialect declaration --------------*- C++ -*-===//
//
// This file contains the declaration of the Aurora dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_DIALECT_H
#define AURORA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace aurora {

// Forward declaration - full class will be defined by TableGen
class AuroraDialect;

} // namespace aurora
} // namespace mlir

// Include the generated dialect declarations
#define GET_DIALECT_DECLS
#include "Aurora/Dialect/Aurora/AuroraDialect.h.inc"

#endif // AURORA_DIALECT_H
