#ifndef AURORA_OPS
#define AURORA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Aurora dialect definition
//===----------------------------------------------------------------------===//

def Aurora_Dialect : Dialect {
  let name = "aurora";
  let cppNamespace = "::mlir::aurora";
  let description = [{
    Aurora dialect for representing and optimizing AI operations.
    
    This dialect provides a representation of AI operations for model compilation
    and optimization, bridging the gap between high-level ML frameworks and 
    low-level execution targets.
  }];
}

//===----------------------------------------------------------------------===//
// Base Aurora operation definition
//===----------------------------------------------------------------------===//

class Aurora_Op<string mnemonic, list<Trait> traits = []> :
    Op<Aurora_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Inference Utilities
//===----------------------------------------------------------------------===//

// Extract shape from a ranked tensor type
def GetRankedTensorShape : NativeCodeCall<"$_self.cast<RankedTensorType>().getShape()">;

// Extract element type from a ranked tensor type
def GetRankedTensorElementType : NativeCodeCall<"$_self.cast<RankedTensorType>().getElementType()">;

// Check if operand is a ranked tensor type
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

//===----------------------------------------------------------------------===//
// Aurora operations
//===----------------------------------------------------------------------===//

def Aurora_ConvOp : Aurora_Op<"conv", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Performs an N-dimensional convolution operation with optional
    stride, padding, dilation, and groups.
    
    Example:
    ```mlir
    %result = aurora.conv(%input, %filter) {
      strides = [1, 1], 
      paddings = [0, 0, 0, 0],
      dilations = [1, 1],
      groups = 1
    } : (tensor<1x64x28x28xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x26x26xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    OptionalAttr<I64ArrayAttr>:$strides,
    OptionalAttr<I64ArrayAttr>:$paddings,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64Attr>:$groups
  );
  
  let results = (outs AnyTensor:$output);
}

def Aurora_MatMulOp : Aurora_Op<"matmul", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Matrix multiplication operation for 2D tensors";
  let description = [{
    Performs a matrix multiplication between two 2D tensors.
    
    The operation computes the matrix product of `lhs` and `rhs`. The `lhs` tensor
    should have shape (M, K) and the `rhs` tensor should have shape (K, N), resulting
    in an output tensor with shape (M, N).
    
    Optionally supports transposing either of the input matrices with the
    `transpose_lhs` and `transpose_rhs` attributes.
    
    Example:
    ```mlir
    // Standard matrix multiplication
    %result = aurora.matmul(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    
    // With transpose attributes
    %result = aurora.matmul(%a, %b) { transpose_lhs = true } : 
        (tensor<8x4xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );
  
  let results = (outs AnyTensor:$output);
  
  // Custom assembly format for improved readability
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict `:` functional-type(operands, results)
  }];
  
  // Add a custom builder to handle creating the operation with inferred result type
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, 
               CArg<"bool", "false">:$transpose_lhs, 
               CArg<"bool", "false">:$transpose_rhs)>
  ];
  
  // Type inference implementation will be defined in C++
  let extraClassDeclaration = [{
    // Type inference
    LogicalResult inferReturnTypes(MLIRContext *context,
                                  std::optional<Location> location,
                                  ValueRange operands,
                                  DictionaryAttr attributes,
                                  RegionRange regions,
                                  SmallVectorImpl<Type> &inferredReturnTypes);
  }];
}

def Aurora_ReluOp : Aurora_Op<"relu", [Pure]> {
  let summary = "Rectified Linear Unit activation function";
  let description = [{
    Applies element-wise ReLU: max(0, x)
    
    Example:
    ```mlir
    %result = aurora.relu(%input) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Aurora_LayerNormOp : Aurora_Op<"layernorm", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Layer normalization operation";
  let description = [{
    Performs layer normalization on the input tensor.
    
    Layer normalization normalizes the activations of the layer for each given example
    in a batch independently, rather than across a batch like batch normalization.
    i.e. applies a transformation that maintains the mean activation within each example
    close to 0 and the activation standard deviation close to 1.
    
    If `epsilon` is provided, it's used as a small constant for numerical stability.
    
    The `axis` attribute specifies which axes to normalize over.
    If not provided, normalizes over the last dimension.
    
    If `scale` and `bias` tensors are provided, they are used for 
    affine transformation after normalization.
    
    Example:
    ```mlir
    // Basic layer normalization over last dimension
    %result = aurora.layernorm(%input) { epsilon = 1.0e-5 } : 
        (tensor<2x512x768xf32>) -> tensor<2x512x768xf32>
    
    // With explicit normalization axis
    %result = aurora.layernorm(%input) { epsilon = 1.0e-5, axis = -1 } : 
        (tensor<2x512x768xf32>) -> tensor<2x512x768xf32>
    
    // With scale and bias
    %result = aurora.layernorm(%input, %scale, %bias) { epsilon = 1.0e-5 } : 
        (tensor<2x512x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<2x512x768xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    Optional<AnyTensor>:$scale,
    Optional<AnyTensor>:$bias,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$epsilon,
    OptionalAttr<I64Attr>:$axis
  );
  
  let results = (outs AnyTensor:$output);
  
  // Custom assembly format
  let assemblyFormat = [{
    `(` $input (`,` $scale^)? (`,` $bias^)? `)` attr-dict `:` functional-type(operands, results)
  }];
  
  // Add a custom builder
  let builders = [
    OpBuilder<(ins "Value":$input, "std::optional<Value>":$scale, 
              "std::optional<Value>":$bias, CArg<"float", "1.0e-5">:$epsilon,
              CArg<"std::optional<int64_t>", "std::nullopt">:$axis)>
  ];
  
  // Type inference implementation
  let extraClassDeclaration = [{
    // For layer normalization, the output type is the same as the input type
    LogicalResult inferReturnTypes(MLIRContext *context,
                                  std::optional<Location> location,
                                  ValueRange operands,
                                  DictionaryAttr attributes,
                                  RegionRange regions,
                                  SmallVectorImpl<Type> &inferredReturnTypes);
  }];
}

def Aurora_FusedAttentionOp : Aurora_Op<"fused_attention", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Fused attention operation";
  let description = [{
    Performs a fused multi-head attention operation, combining query, key, value
    projections with the softmax and attention computation.
    
    This operation is meant to capture vendor-specific fused attention implementations
    that may be optimized for specific hardware. The semantics are similar to the
    attention mechanism used in Transformer models.
    
    The basic computation performed is:
    ```
    Q = input * W_q
    K = input * W_k
    V = input * V_k
    scores = Q * K^T / sqrt(head_size)
    weights = softmax(scores + attention_mask)
    output = weights * V
    ```
    
    The operation supports an optional `attention_mask` for masking certain attention weights.
    The `scale_factor` attribute can override the default scaling of `1/sqrt(head_size)`.
    
    Example:
    ```mlir
    // Basic fused attention with 8 heads
    %result = aurora.fused_attention(%input, %w_query, %w_key, %w_value) {
      num_heads = 8
    } : (tensor<2x512x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) 
        -> tensor<2x512x768xf32>
    
    // With attention mask
    %result = aurora.fused_attention(%input, %w_query, %w_key, %w_value, %mask) {
      num_heads = 8,
      scale_factor = 0.125
    } : (tensor<2x512x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, 
         tensor<768x768xf32>, tensor<2x8x512x512xf32>) -> tensor<2x512x768xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,               // Input tensor [batch, seq_len, hidden_size]
    AnyTensor:$weights_query,       // Query weights [hidden_size, hidden_size]
    AnyTensor:$weights_key,         // Key weights [hidden_size, hidden_size]
    AnyTensor:$weights_value,       // Value weights [hidden_size, hidden_size]
    Optional<AnyTensor>:$attention_mask, // Optional attention mask
    I64Attr:$num_heads,             // Number of attention heads
    OptionalAttr<F32Attr>:$scale_factor, // Optional override for attention scaling
    DefaultValuedAttr<BoolAttr, "false">:$causal // Whether to use causal attention
  );
  
  let results = (outs AnyTensor:$output);
  
  // Custom assembly format
  let assemblyFormat = [{
    `(` $input `,` $weights_query `,` $weights_key `,` $weights_value 
    (`,` $attention_mask^)? `)` attr-dict `:` functional-type(operands, results)
  }];
  
  // Add a custom builder
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$weights_query, "Value":$weights_key,
               "Value":$weights_value, "std::optional<Value>":$attention_mask,
               "int64_t":$num_heads, CArg<"std::optional<float>", "std::nullopt">:$scale_factor,
               CArg<"bool", "false">:$causal)>
  ];
  
  // Type inference implementation
  let extraClassDeclaration = [{
    // The output shape is the same as the input shape
    LogicalResult inferReturnTypes(MLIRContext *context,
                                  std::optional<Location> location,
                                  ValueRange operands,
                                  DictionaryAttr attributes,
                                  RegionRange regions,
                                  SmallVectorImpl<Type> &inferredReturnTypes);
  }];
}

#endif // AURORA_OPS