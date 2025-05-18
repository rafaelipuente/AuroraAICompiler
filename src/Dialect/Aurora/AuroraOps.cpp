#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir {
namespace aurora {

//===----------------------------------------------------------------------===//
// Operation Builders
//===----------------------------------------------------------------------===//

// MatMulOp builder implementation
MatMulOp::Builder::Builder(Value lhs, Value rhs, bool transpose_lhs, bool transpose_rhs) {
  build(lhs.getContext(), /*result=*/nullptr, lhs, rhs, transpose_lhs, transpose_rhs);
}

// LayerNormOp builder implementation
LayerNormOp::Builder::Builder(Value input, std::optional<Value> scale, 
                               std::optional<Value> bias, float epsilon,
                               std::optional<int64_t> axis) {
  build(input.getContext(), /*result=*/nullptr, input, 
        scale.has_value() ? scale.value() : Value(), 
        bias.has_value() ? bias.value() : Value(), 
        FloatAttr::get(input.getType().getContext(), epsilon),
        axis.has_value() ? IntegerAttr::get(
          IntegerType::get(input.getContext(), 64), axis.value()) : nullptr);
}

// FusedAttentionOp builder implementation
FusedAttentionOp::Builder::Builder(Value input, Value weights_query, Value weights_key,
                                   Value weights_value, std::optional<Value> attention_mask,
                                   int64_t num_heads, std::optional<float> scale_factor,
                                   bool causal) {
  build(input.getContext(), /*result=*/nullptr, input, weights_query, weights_key, weights_value,
        attention_mask.has_value() ? attention_mask.value() : Value(),
        IntegerAttr::get(IntegerType::get(input.getContext(), 64), num_heads),
        scale_factor.has_value() 
          ? FloatAttr::get(input.getType().getContext(), scale_factor.value()) 
          : nullptr,
        BoolAttr::get(input.getContext(), causal));
}

//===----------------------------------------------------------------------===//
// MatMulOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get the lhs and rhs operand types
  auto lhsType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto rhsType = operands[1].getType().dyn_cast<RankedTensorType>();
  
  // Both operands must be ranked tensors for shape inference
  if (!lhsType || !rhsType) {
    return emitOptionalError(location, "MatMulOp operands must be ranked tensors");
  }
  
  // Get the matmul attribute values
  auto attr = attributes.getAs<DictionaryAttr>();
  bool transposeLhs = false;
  bool transposeRhs = false;
  
  if (auto transposeAttr = attr.getAs<BoolAttr>("transpose_lhs")) {
    transposeLhs = transposeAttr.getValue();
  }
  
  if (auto transposeAttr = attr.getAs<BoolAttr>("transpose_rhs")) {
    transposeRhs = transposeAttr.getValue();
  }
  
  // Check the input ranks
  auto lhsRank = lhsType.getRank();
  auto rhsRank = rhsType.getRank();
  
  if (lhsRank != 2 || rhsRank != 2) {
    return emitOptionalError(
      location, "MatMulOp requires both operands to be 2D tensors, but got "
      "lhs rank = " + std::to_string(lhsRank) + 
      " and rhs rank = " + std::to_string(rhsRank));
  }
  
  // Get the lhs and rhs shapes
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  
  // Adjust dimensions based on transpose flags
  int64_t lhsRows = transposeLhs ? lhsShape[1] : lhsShape[0];
  int64_t lhsCols = transposeLhs ? lhsShape[0] : lhsShape[1];
  int64_t rhsRows = transposeRhs ? rhsShape[1] : rhsShape[0];
  int64_t rhsCols = transposeRhs ? rhsShape[0] : rhsShape[1];
  
  // Validate inner dimensions match for matrix multiplication
  if (lhsCols != rhsRows) {
    return emitOptionalError(
      location, "MatMulOp requires inner dimensions to match, but got " + 
      std::to_string(lhsCols) + " and " + std::to_string(rhsRows));
  }
  
  // Output shape is [lhsRows, rhsCols]
  SmallVector<int64_t, 2> outputShape{lhsRows, rhsCols};
  
  // Create the output tensor type with the same element type as the input
  auto outputType = RankedTensorType::get(
    outputShape, lhsType.getElementType());
    
  inferredReturnTypes.push_back(outputType);
  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult LayerNormOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get the input tensor type
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  
  // Input must be a ranked tensor for shape inference
  if (!inputType) {
    return emitOptionalError(location, 
                            "LayerNormOp input must be a ranked tensor");
  }
  
  // If scale and bias are provided, verify their shapes are compatible
  if (operands.size() > 1) {
    auto inputRank = inputType.getRank();
    if (inputRank == 0) {
      return emitOptionalError(location, 
                            "LayerNormOp input cannot be a scalar");
    }
    
    // Get the normalization axis, defaulting to the last dimension
    int64_t normAxis = -1;
    if (auto axisAttr = attributes.getAs<DictionaryAttr>().getAs<IntegerAttr>("axis")) {
      normAxis = axisAttr.getInt();
    }
    
    // Convert negative axis to positive
    if (normAxis < 0) {
      normAxis += inputRank;
    }
    
    // Validate the axis is within range
    if (normAxis < 0 || normAxis >= inputRank) {
      return emitOptionalError(location, 
                        "LayerNormOp axis is out of bounds for input rank " + 
                        std::to_string(inputRank));
    }
    
    // Check scale tensor if present
    if (operands.size() > 1 && !operands[1].getType().isa<NoneType>()) {
      auto scaleType = operands[1].getType().dyn_cast<RankedTensorType>();
      if (!scaleType) {
        return emitOptionalError(location, 
                              "LayerNormOp scale must be a ranked tensor");
      }
      
      // Scale should be a 1D tensor with size matching the normalized dimension
      if (scaleType.getRank() != 1 || 
          (scaleType.getDimSize(0) != ShapedType::kDynamic && 
           inputType.getDimSize(normAxis) != ShapedType::kDynamic && 
           scaleType.getDimSize(0) != inputType.getDimSize(normAxis))) {
        return emitOptionalError(location, 
                                "LayerNormOp scale has incompatible shape");
      }
    }
    
    // Check bias tensor if present
    if (operands.size() > 2 && !operands[2].getType().isa<NoneType>()) {
      auto biasType = operands[2].getType().dyn_cast<RankedTensorType>();
      if (!biasType) {
        return emitOptionalError(location, 
                              "LayerNormOp bias must be a ranked tensor");
      }
      
      // Bias should be a 1D tensor with size matching the normalized dimension
      if (biasType.getRank() != 1 || 
          (biasType.getDimSize(0) != ShapedType::kDynamic && 
           inputType.getDimSize(normAxis) != ShapedType::kDynamic && 
           biasType.getDimSize(0) != inputType.getDimSize(normAxis))) {
        return emitOptionalError(location, 
                                "LayerNormOp bias has incompatible shape");
      }
    }
  }
  
  // LayerNorm output has the same shape and type as input
  inferredReturnTypes.push_back(inputType);
  return success();
}

//===----------------------------------------------------------------------===//
// FusedAttentionOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult FusedAttentionOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get the input tensor type
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  
  // Input must be a ranked tensor for shape inference
  if (!inputType) {
    return emitOptionalError(location, 
                        "FusedAttentionOp input must be a ranked tensor");
  }
  
  // Check input rank (expecting at least 3D: [batch, seq_len, hidden_size])
  auto inputRank = inputType.getRank();
  if (inputRank < 3) {
    return emitOptionalError(location, 
                  "FusedAttentionOp input must have at least rank 3, but got " + 
                  std::to_string(inputRank));
  }
  
  // Check that hidden dimension is divisible by num_heads
  auto hiddenSize = inputType.getDimSize(inputRank - 1);
  if (hiddenSize != ShapedType::kDynamic) {
    auto attr = attributes.getAs<DictionaryAttr>();
    auto numHeadsAttr = attr.getAs<IntegerAttr>("num_heads");
    if (!numHeadsAttr) {
      return emitOptionalError(location, 
                          "FusedAttentionOp requires num_heads attribute");
    }
    
    int64_t numHeads = numHeadsAttr.getInt();
    if (numHeads <= 0) {
      return emitOptionalError(location, 
                    "FusedAttentionOp num_heads must be positive, but got " + 
                    std::to_string(numHeads));
    }
    
    if (hiddenSize % numHeads != 0) {
      return emitOptionalError(location, 
                    "FusedAttentionOp hidden dimension (" + 
                    std::to_string(hiddenSize) + 
                    ") must be divisible by num_heads (" + 
                    std::to_string(numHeads) + ")");
    }
  }
  
  // Check weight matrices
  for (int i = 1; i <= 3; i++) {
    auto weightType = operands[i].getType().dyn_cast<RankedTensorType>();
    if (!weightType) {
      return emitOptionalError(location, 
                          "FusedAttentionOp weight must be a ranked tensor");
    }
    
    // Weights should be 2D matrices
    if (weightType.getRank() != 2) {
      return emitOptionalError(location, 
                    "FusedAttentionOp weight must be a 2D tensor, but got rank " + 
                    std::to_string(weightType.getRank()));
    }
    
    // Check that first dimension of weight matches hidden size of input
    auto weightDim0 = weightType.getDimSize(0);
    if (weightDim0 != ShapedType::kDynamic && 
        hiddenSize != ShapedType::kDynamic && 
        weightDim0 != hiddenSize) {
      return emitOptionalError(location, 
                    "FusedAttentionOp weight dimension 0 (" + 
                    std::to_string(weightDim0) + ") should match input hidden size (" + 
                    std::to_string(hiddenSize) + ")");
    }
  }
  
  // Check attention mask if present
  if (operands.size() > 4 && !operands[4].getType().isa<NoneType>()) {
    auto maskType = operands[4].getType().dyn_cast<RankedTensorType>();
    if (!maskType) {
      return emitOptionalError(location, 
                          "FusedAttentionOp mask must be a ranked tensor");
    }
    
    // Mask should be either 2D [seq_len, seq_len] or 4D [batch, num_heads, seq_len, seq_len]
    auto maskRank = maskType.getRank();
    if (maskRank != 2 && maskRank != 4) {
      return emitOptionalError(location, 
                    "FusedAttentionOp mask must have rank 2 or 4, but got " + 
                    std::to_string(maskRank));
    }
  }
  
  // Output has the same shape as the input
  inferredReturnTypes.push_back(inputType);
  return success();
}

//===----------------------------------------------------------------------===//
// ConvOp Implementation
//===----------------------------------------------------------------------===//

// Helper to handle default or provided values for convolution attributes
static void fillDefaultConvAttributes(ConvOp op) {
  MLIRContext *context = op.getContext();
  Builder builder(context);
  
  // If strides aren't specified, default to all ones
  if (!op.getStrides()) {
    // Get input rank to determine number of spatial dimensions
    if (auto inputType = op.getInput().getType().dyn_cast<RankedTensorType>()) {
      int spatialDims = inputType.getRank() - 2; // Exclude batch and channel dims
      if (spatialDims > 0) {
        SmallVector<int64_t> defaultStrides(spatialDims, 1);
        op.setStridesAttr(builder.getI64ArrayAttr(defaultStrides));
      }
    }
  }
  
  // If paddings aren't specified, default to all zeros
  if (!op.getPaddings()) {
    if (auto inputType = op.getInput().getType().dyn_cast<RankedTensorType>()) {
      int spatialDims = inputType.getRank() - 2;
      if (spatialDims > 0) {
        // For each spatial dimension, we need two padding values (start and end)
        SmallVector<int64_t> defaultPaddings(spatialDims * 2, 0);
        op.setPaddingsAttr(builder.getI64ArrayAttr(defaultPaddings));
      }
    }
  }
  
  // If dilations aren't specified, default to all ones
  if (!op.getDilations()) {
    if (auto inputType = op.getInput().getType().dyn_cast<RankedTensorType>()) {
      int spatialDims = inputType.getRank() - 2;
      if (spatialDims > 0) {
        SmallVector<int64_t> defaultDilations(spatialDims, 1);
        op.setDilationsAttr(builder.getI64ArrayAttr(defaultDilations));
      }
    }
  }
  
  // If groups aren't specified, default to 1
  if (!op.getGroups()) {
    op.setGroupsAttr(builder.getI64IntegerAttr(1));
  }
}

//===----------------------------------------------------------------------===//
// ReluOp Type Inference
//===----------------------------------------------------------------------===//

// Relu simply produces an output of the same type as the input
static LogicalResult inferReluReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get the input tensor type
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  
  // Input must be a ranked tensor for shape inference
  if (!inputType) {
    return emitOptionalError(location, "ReluOp input must be a ranked tensor");
  }
  
  // ReLU output has the same shape and type as input
  inferredReturnTypes.push_back(inputType);
  return success();
}

//===----------------------------------------------------------------------===//
// Override the inferReturnTypes for operations that didn't declare the interface in TableGen
//===----------------------------------------------------------------------===//

LogicalResult ConvOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get the input and filter tensor types
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto filterType = operands[1].getType().dyn_cast<RankedTensorType>();
  
  // Both operands must be ranked tensors for shape inference
  if (!inputType || !filterType) {
    return emitOptionalError(location, "ConvOp operands must be ranked tensors");
  }
  
  // Get the input and filter shapes
  auto inputShape = inputType.getShape();
  auto filterShape = filterType.getShape();
  
  // Check if the ranks are valid for convolution
  auto inputRank = inputType.getRank();
  auto filterRank = filterType.getRank();
  
  if (inputRank < 3 || filterRank < 3) {
    return emitOptionalError(location,
      "ConvOp requires input rank >= 3 and filter rank >= 3, but got " +
      std::to_string(inputRank) + " and " + std::to_string(filterRank));
  }
  
  // Extract attributes
  auto attr = attributes.getAs<DictionaryAttr>();
  SmallVector<int64_t> strides;
  SmallVector<int64_t> paddings;
  SmallVector<int64_t> dilations;
  int64_t groups = 1;
  
  if (auto stridesAttr = attr.getAs<ArrayAttr>("strides")) {
    for (auto stride : stridesAttr) {
      if (auto intAttr = stride.dyn_cast<IntegerAttr>()) {
        strides.push_back(intAttr.getInt());
      }
    }
  } else {
    // Default strides to 1 for each spatial dimension
    strides.resize(inputRank - 2, 1);
  }
  
  if (auto paddingsAttr = attr.getAs<ArrayAttr>("paddings")) {
    for (auto padding : paddingsAttr) {
      if (auto intAttr = padding.dyn_cast<IntegerAttr>()) {
        paddings.push_back(intAttr.getInt());
      }
    }
  } else {
    // Default paddings to 0 for each spatial dimension (both start and end)
    paddings.resize((inputRank - 2) * 2, 0);
  }
  
  if (auto dilationsAttr = attr.getAs<ArrayAttr>("dilations")) {
    for (auto dilation : dilationsAttr) {
      if (auto intAttr = dilation.dyn_cast<IntegerAttr>()) {
        dilations.push_back(intAttr.getInt());
      }
    }
  } else {
    // Default dilations to 1 for each spatial dimension
    dilations.resize(inputRank - 2, 1);
  }
  
  if (auto groupsAttr = attr.getAs<IntegerAttr>("groups")) {
    groups = groupsAttr.getInt();
  }
  
  // Check groups validity
  if (groups <= 0) {
    return emitOptionalError(location, 
      "ConvOp groups must be positive, but got " + std::to_string(groups));
  }
  
  // Check input channels divisible by groups
  if (inputShape[1] != ShapedType::kDynamic && 
      inputShape[1] % groups != 0) {
    return emitOptionalError(location,
      "ConvOp input channels (" + std::to_string(inputShape[1]) + 
      ") must be divisible by groups (" + std::to_string(groups) + ")");
  }
  
  // Check filter channels match grouped input channels
  if (filterShape[1] != ShapedType::kDynamic && 
      inputShape[1] != ShapedType::kDynamic && 
      filterShape[1] * groups != inputShape[1]) {
    return emitOptionalError(location,
      "ConvOp filter input channels * groups (" + 
      std::to_string(filterShape[1]) + " * " + std::to_string(groups) + 
      ") must match input channels (" + std::to_string(inputShape[1]) + ")");
  }
  
  // Calculate output shape
  SmallVector<int64_t, 4> outputShape;
  
  // Batch size remains the same
  outputShape.push_back(inputShape[0]);
  
  // Output channels determined by filter
  outputShape.push_back(filterShape[0]);
  
  // Spatial dimensions
  for (unsigned i = 0; i < strides.size(); ++i) {
    int64_t inputSize = inputShape[i + 2];
    int64_t filterSize = filterShape[i + 2];
    int64_t stride = strides[i];
    int64_t dilation = dilations[i];
    
    // For paddings, we need to handle both start and end paddings
    int64_t padStart = (i < paddings.size() / 2) ? paddings[i * 2] : 0;
    int64_t padEnd = (i < paddings.size() / 2) ? paddings[i * 2 + 1] : 0;
    
    // Effective filter size with dilation
    int64_t effectiveFilterSize = filterSize + (filterSize - 1) * (dilation - 1);
    
    // Calculate output size - if input size is dynamic, output is also dynamic
    if (inputSize == ShapedType::kDynamic) {
      outputShape.push_back(ShapedType::kDynamic);
    } else {
      // Output size formula: (input + padStart + padEnd - effectiveFilterSize) / stride + 1
      int64_t outputSize = 
        (inputSize + padStart + padEnd - effectiveFilterSize) / stride + 1;
      
      if (outputSize <= 0) {
        return emitOptionalError(location,
          "ConvOp results in invalid output spatial dimension (" + 
          std::to_string(outputSize) + ") for dimension " + std::to_string(i));
      }
      
      outputShape.push_back(outputSize);
    }
  }
  
  // Create the output tensor type with the same element type as the input
  auto outputType = RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(outputType);
  
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.cpp.inc"

} // namespace aurora
} // namespace mlir