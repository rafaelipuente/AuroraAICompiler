//===----------------------------------------------------------------------===//
// Aurora operations implementation
//===----------------------------------------------------------------------===//

#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace aurora {

// Any custom operation method implementations would go here

} // namespace aurora
} // namespace mlir

// After our namespace implementations, include the TableGen-generated code
#define GET_OP_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.cpp.inc"
