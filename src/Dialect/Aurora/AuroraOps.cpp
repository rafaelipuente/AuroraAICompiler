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

LogicalResult BiasAddOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto biasType = getBias().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();

  if (!inputType || !biasType || !resultType)
    return success(); // Unranked tensors are allowed; no static check.

  if (biasType.getRank() != 1)
    return emitOpError("bias must be rank-1, got rank ")
           << biasType.getRank();

  if (inputType.getRank() < 1)
    return emitOpError("input must be at least rank-1");

  int64_t biasDim = biasType.getDimSize(0);
  int64_t inputLastDim = inputType.getDimSize(inputType.getRank() - 1);

  if (!ShapedType::isDynamic(biasDim) && !ShapedType::isDynamic(inputLastDim) &&
      biasDim != inputLastDim)
    return emitOpError("bias length (")
           << biasDim << ") must match the last dimension of input ("
           << inputLastDim << ")";

  if (inputType != resultType)
    return emitOpError("result type must match input type, got ")
           << resultType << " vs " << inputType;

  return success();
}

} // namespace aurora
} // namespace mlir

// After our namespace implementations, include the TableGen-generated code
#define GET_OP_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.cpp.inc"
