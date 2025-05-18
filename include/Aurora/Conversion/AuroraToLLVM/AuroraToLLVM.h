#ifndef AURORA_CONVERSION_AURORA_TO_LLVM_H
#define AURORA_CONVERSION_AURORA_TO_LLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
namespace aurora {

/// Create a pass to convert Aurora operations to the LLVM dialect.
std::unique_ptr<Pass> createConvertAuroraToLLVMPass();

} // namespace aurora
} // namespace mlir

#endif // AURORA_CONVERSION_AURORA_TO_LLVM_H
