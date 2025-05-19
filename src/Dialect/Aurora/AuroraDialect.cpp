//===----------------------------------------------------------------------===//
// Aurora dialect implementation
//===----------------------------------------------------------------------===//

#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace aurora {

void AuroraDialect::initialize() {
  // Register all operations from the generated file
  addOperations<
#define GET_OP_LIST
#include "Aurora/Dialect/Aurora/AuroraOps.cpp.inc"
  >();
}

// Parse an attribute registered to this dialect
Attribute AuroraDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // Currently, no custom attributes are defined
  parser.emitError(parser.getNameLoc(), "Aurora dialect has no custom attributes");
  return Attribute();
}

// Print an attribute registered to this dialect
void AuroraDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // Currently, no custom attributes are defined
}

// Parse a type registered to this dialect
Type AuroraDialect::parseType(DialectAsmParser &parser) const {
  // Currently, no custom types are defined
  parser.emitError(parser.getNameLoc(), "Aurora dialect has no custom types");
  return Type();
}

// Print a type registered to this dialect
void AuroraDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Currently, no custom types are defined
}

} // namespace aurora
} // namespace mlir

// Include dialect definitions
#define GET_DIALECT_DEFS
#include "Aurora/Dialect/Aurora/AuroraDialect.cpp.inc"
