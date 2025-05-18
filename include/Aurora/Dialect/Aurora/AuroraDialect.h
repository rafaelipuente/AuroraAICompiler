#ifndef AURORA_DIALECT_AURORA_DIALECT_H
#define AURORA_DIALECT_AURORA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace aurora {

class AuroraDialect : public Dialect {
public:
  explicit AuroraDialect(MLIRContext *context);
  
  static StringRef getDialectNamespace() { return "aurora"; }
  
  void initialize();
};

} // namespace aurora
} // namespace mlir

#endif // AURORA_DIALECT_AURORA_DIALECT_H
