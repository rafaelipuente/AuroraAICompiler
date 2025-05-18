# Aurora MLIR Dialect

## Introduction

[MLIR (Multi-Level Intermediate Representation)](https://mlir.llvm.org/) is a compiler infrastructure designed to unify the compilation of different programming models and hardware targets. MLIR uses *dialects* as a mechanism to represent and encapsulate domain-specific abstractions, operations, and transformations. 

A custom dialect in MLIR allows developers to:
- Represent domain-specific operations at the right level of abstraction
- Implement domain-specific optimizations
- Progressively lower from higher-level abstractions to low-level implementations
- Reuse existing infrastructure for operations common in different domains

## The Aurora Dialect

The `aurora` dialect is a custom MLIR dialect designed specifically for deep learning operations. It provides a set of operations that represent common neural network building blocks, enabling the AuroraAICompiler to optimize and compile deep learning models for various hardware targets.

The Aurora dialect bridges the gap between high-level ML frameworks (like PyTorch and TensorFlow) and low-level execution targets (CPUs, GPUs, specialized accelerators). It facilitates:

- Representation of neural network operations
- Domain-specific optimizations (fusion, memory layout optimization)
- Target-specific code generation
- Progressive lowering to more efficient implementations

## Supported Operations

The Aurora dialect currently supports the following operations:

1. `aurora.matmul` - Matrix multiplication 
2. `aurora.layernorm` - Layer normalization
3. `aurora.fused_attention` - Multi-head attention fusion
4. `aurora.conv` - N-dimensional convolution
5. `aurora.relu` - Rectified Linear Unit activation

All operations in the Aurora dialect implement proper shape inference through MLIR's `InferTypeOpInterface`, ensuring that the output shapes are correctly determined at compile time.

## Operation Documentation

### aurora.matmul

#### Description
Performs matrix multiplication between two 2D tensors. The operation multiplies two matrices with optional transposition of either input.

#### Operands
- `lhs`: The left-hand side tensor of shape `(M,K)` or `(K,M)` if transposed
- `rhs`: The right-hand side tensor of shape `(K,N)` or `(N,K)` if transposed

#### Attributes
- `transpose_lhs`: Boolean attribute indicating whether to transpose the left-hand side tensor (default: false)
- `transpose_rhs`: Boolean attribute indicating whether to transpose the right-hand side tensor (default: false)

#### Result
- Output tensor of shape `(M,N)` containing the matrix product

#### Example
```mlir
// Standard matrix multiplication
%result = aurora.matmul(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

// With transpose attributes
%result = aurora.matmul(%a, %b) { transpose_lhs = true } : 
    (tensor<8x4xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
```

### aurora.layernorm

#### Description
Performs layer normalization on the input tensor. Layer normalization normalizes the activations of the layer for each given example in a batch independently, rather than across a batch like batch normalization.

#### Operands
- `input`: The input tensor to normalize
- `scale` (optional): A 1D tensor for scaling the normalized values
- `bias` (optional): A 1D tensor for adding a bias to the normalized values

#### Attributes
- `epsilon`: Floating-point attribute specifying a small constant for numerical stability (default: 1.0e-5)
- `axis`: Integer attribute specifying which axis to normalize over (default: last dimension)

#### Result
- Output tensor of the same shape as the input tensor

#### Example
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

### aurora.fused_attention

#### Description
Performs a fused multi-head attention operation, combining query, key, value projections with the softmax and attention computation. This operation is designed to capture vendor-specific fused attention implementations that may be optimized for specific hardware.

#### Operands
- `input`: The input tensor of shape `[batch, seq_len, hidden_size]`
- `weights_query`: The query weights of shape `[hidden_size, hidden_size]`
- `weights_key`: The key weights of shape `[hidden_size, hidden_size]`
- `weights_value`: The value weights of shape `[hidden_size, hidden_size]`
- `attention_mask` (optional): Attention mask tensor

#### Attributes
- `num_heads`: Integer attribute specifying the number of attention heads
- `scale_factor` (optional): Float attribute overriding the default scaling factor of 1/sqrt(head_size)
- `causal`: Boolean attribute indicating whether to use causal attention (default: false)

#### Result
- Output tensor of the same shape as the input tensor

#### Example
```mlir
// Basic fused attention with 8 heads
%result = aurora.fused_attention(%input, %w_query, %w_key, %w_value) {
  num_heads = 8
} : (tensor<2x512x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) 
    -> tensor<2x512x768xf32>

// With attention mask
%result = aurora.fused_attention(%input, %w_query, %w_key, %w_value, %mask) {
  num_heads = 8,
  scale_factor = 0.125,
  causal = true
} : (tensor<2x512x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, 
     tensor<768x768xf32>, tensor<2x8x512x512xf32>) -> tensor<2x512x768xf32>
```

### aurora.conv

#### Description
Performs an N-dimensional convolution operation with optional stride, padding, dilation, and groups.

#### Operands
- `input`: The input tensor of shape `[batch, channels, spatial_dims...]`
- `filter`: The filter/kernel tensor of shape `[out_channels, in_channels/groups, spatial_dims...]`

#### Attributes
- `strides`: Integer array attribute specifying the stride in each spatial dimension (default: all 1's)
- `paddings`: Integer array attribute specifying padding at the beginning and end of each spatial dimension (default: all 0's)
- `dilations`: Integer array attribute specifying the dilation in each spatial dimension (default: all 1's)
- `groups`: Integer attribute specifying the number of groups for grouped convolution (default: 1)

#### Result
- Output tensor of shape `[batch, out_channels, output_spatial_dims...]` with spatial dimensions calculated based on the input, filter, and attributes

#### Example
```mlir
// 2D convolution with explicit attributes
%result = aurora.conv(%input, %filter) {
  strides = [2, 2], 
  paddings = [1, 1, 1, 1],
  dilations = [1, 1],
  groups = 1
} : (tensor<1x64x28x28xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x14x14xf32>

// Grouped convolution
%result = aurora.conv(%input, %filter) {
  strides = [1, 1], 
  groups = 64
} : (tensor<1x64x56x56xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x54x54xf32>
```

### aurora.relu

#### Description
Applies the Rectified Linear Unit activation function element-wise: f(x) = max(0, x).

#### Operands
- `input`: The input tensor of any shape

#### Result
- Output tensor of the same shape as the input tensor

#### Example
```mlir
%result = aurora.relu(%input) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
```

## Shape Inference

All operations in the Aurora dialect implement the `InferTypeOpInterface` from MLIR, which enables compile-time shape and type inference. This has several benefits:

1. **Error Detection**: Shape mismatches and invalid configurations are detected early in the compilation process
2. **Optimization Enablement**: Precise shape information enables more aggressive optimizations
3. **Code Generation**: Concrete shapes allow for more efficient code generation

The shape inference implementations handle both static and dynamic shapes:
- For static shapes, exact output dimensions are calculated
- For dynamic shapes, the dialect uses MLIR's dynamic dimension marker (`ShapedType::kDynamic`) while still enforcing as many constraints as possible

## Integration with ML Frameworks

The Aurora dialect is designed to work seamlessly with existing machine learning frameworks:

### PyTorch Integration (Planned)
- Direct import of PyTorch models via TorchScript
- Custom operators mapping to Aurora dialect operations
- Support for PyTorch's dynamic shape capabilities

### ONNX Integration (Planned)
- Import of ONNX models from any framework that can export to ONNX
- Mapping of ONNX operations to Aurora dialect
- Support for ONNX's static shape and dynamic shape variants

## Future Directions

The Aurora dialect will continue to evolve with:

1. Additional operations for covering more deep learning use cases
2. Extended attributes and configuration options for existing operations
3. More complex fusion patterns
4. Better support for quantized operations
5. Hardware-specific operation variants for specialized accelerators

---

*Last updated: May 18, 2025*
