#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"

using namespace mlir;
using namespace mlir::aurora;

//===----------------------------------------------------------------------===//
// Aurora dialect.
//===----------------------------------------------------------------------===//

void AuroraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Aurora/Dialect/Aurora/AuroraOps.cpp.inc"
  >();
}

AuroraDialect::AuroraDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<AuroraDialect>()) {
  initialize();
}
