# Aurora AI Compiler: A Complete Walkthrough

## Introduction

The Aurora AI Compiler is a hybrid C++/Python compiler designed to optimize neural network models for high-performance execution across a variety of hardware targets. Taking inspiration from AmpereOne's Aurora AI accelerators, this project aims to bridge the gap between high-level model definitions (like those in PyTorch or ONNX) and optimized low-level code that can efficiently execute on modern hardware.

Machine learning frameworks have revolutionized how we build and train AI models, but the path from a trained model to efficient deployment remains complex. Traditional approaches often involve multiple translation layers, each introducing overhead and potential performance loss. The Aurora AI Compiler addresses this challenge by providing a direct path from model definition to optimized code through the power of Multi-Level Intermediate Representation (MLIR).

The compiler's name pays homage to the Aurora Borealis — just as these northern lights transform energy into spectacular visible patterns, our compiler transforms abstract model definitions into optimized execution patterns tailored for specific hardware.

## Architecture Overview

The Aurora AI Compiler follows a multi-stage compilation pipeline that progressively lowers abstractions while preserving optimization opportunities:

```
ONNX Model -> Intermediate Representation -> Aurora MLIR -> Optimized Target Code
```

### Stage 1: Model Import
The journey begins with importing models from standard formats like ONNX. The `ONNXLoader` parses model files and extracts the computational graph, tensor shapes, operations, and parameters into a language-agnostic intermediate representation.

### Stage 2: Aurora IR
This intermediate representation serves as the bridge between the source model and our MLIR dialect. It preserves the original model's computation intent while beginning to structure it in a way amenable to our compiler's optimization passes.

### Stage 3: Aurora MLIR Dialect
The heart of the compiler is the Aurora dialect — a domain-specific extension to MLIR that represents AI operations at the right level of abstraction. This is where most of our optimization magic happens, including operation fusion, memory layout optimization, and hardware-specific transformations.

### Stage 4: Code Generation
Finally, the optimized MLIR is lowered to target-specific code (CPU, GPU, or specialized accelerators) through LLVM infrastructure, generating high-performance executables.

This multi-stage approach allows us to apply optimizations at the right level of abstraction — preserving high-level semantic information where it's useful, while eventually producing tightly optimized low-level code.

## The Aurora Dialect

The Aurora dialect is a custom MLIR dialect that encapsulates common deep learning operations with their semantics, constraints, and optimization opportunities. The dialect is defined using TableGen, which generates the necessary C++ code for the MLIR infrastructure.

### Core Operations

#### 1. Matrix Multiplication (`aurora.matmul`)

The matrix multiplication operation is fundamental to neural networks, especially in fully connected layers.

```mlir
%result = aurora.matmul(%lhs, %rhs) { transpose_lhs = false, transpose_rhs = false } : 
    (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
```

Our implementation includes:
- Support for transposing either input matrix
- Comprehensive shape inference
- Broadcasting for batched matrix multiplications
- Specialized versions for different data types

The C++ implementation leverages MLIR's `InferTypeOpInterface` to statically determine output shapes at compile time, enabling further optimizations.

#### 2. Layer Normalization (`aurora.layernorm`)

Layer normalization is crucial for stabilizing training in transformers and other deep networks.

```mlir
%result = aurora.layernorm(%input, %scale, %bias) { epsilon = 1.0e-5, axis = -1 } : 
    (tensor<2x512x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<2x512x768xf32>
```

Our implementation:
- Supports normalization across any axis
- Handles optional scale and bias parameters
- Provides numerical stability via configurable epsilon
- Includes specialized optimizations for common layer norm patterns

#### 3. Fused Attention (`aurora.fused_attention`)

Attention mechanisms are the cornerstone of transformer models, but naive implementations lead to excessive memory usage and kernel launches.

```mlir
%result = aurora.fused_attention(%input, %wq, %wk, %wv) { num_heads = 8 } : 
    (tensor<2x512x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2x512x768xf32>
```

Our fused attention:
- Combines multiple operations (QKV projections, softmax, etc.) into a single op
- Supports configurable attention heads
- Includes optimizations for causal attention in generative models
- Provides specialized implementations for different hardware backends

#### 4. Convolution (`aurora.conv`)

Convolution operations power computer vision models and certain audio processing networks.

```mlir
%result = aurora.conv(%input, %filter) { strides = [2, 2], paddings = [1, 1, 1, 1] } : 
    (tensor<1x64x28x28xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x14x14xf32>
```

Features include:
- Support for N-dimensional convolutions
- Configurable strides, padding, and dilation
- Group convolution support for depthwise separable models
- Shape inference that computes output dimensions

#### 5. ReLU Activation (`aurora.relu`)

The simplest yet one of the most common activation functions:

```mlir
%result = aurora.relu(%input) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
```

While simple, our implementation is optimized for:
- In-place operation where possible
- Vectorization on supporting hardware
- Fusion with preceding or following operations

### Implementation Approach

Each operation in our dialect follows a consistent pattern:
1. TableGen definition that specifies the operation's interface, attributes, and documentation
2. C++ implementation of shape inference and custom verifiers
3. Custom builders to simplify operation creation
4. Specialized lowering patterns for different hardware targets

All operations implement key MLIR interfaces like `InferTypeOpInterface` for static shape determination and `MemoryEffectsOpInterface` for optimization passes.

## ONNX Frontend

The ONNX frontend is the gateway to our compiler, enabling it to work with models from a wide variety of frameworks that can export to ONNX format.

### ONNXLoader

The `ONNXLoader` provides a clean Python API to parse ONNX model files:

```python
loader = ONNXLoader(verbose=True)
graph = loader.load("model.onnx")
```

Under the hood, it:
1. Parses the ONNX protobuf format
2. Extracts computational graph structure
3. Builds a clean intermediate representation with:
   - Operation nodes (with name, type, inputs, outputs, attributes)
   - Tensor info (shapes, types, constant values)
   - Graph-level metadata

This loader handles the complexity of ONNX's operation set versioning and ensures all tensor shapes and types are properly represented.

### ONNXToMLIRConverter

The `ONNXToMLIRConverter` transforms our intermediate representation into Aurora dialect operations:

```python
converter = ONNXToMLIRConverter()
mlir_text = converter.convert(graph)
```

Key features include:
1. Operation mapping from ONNX to Aurora dialect
2. Pattern recognition for fusing operations (e.g., detecting attention patterns)
3. Type conversion between ONNX and MLIR type systems
4. Generation of complete MLIR modules with proper function signatures

The converter uses a combination of direct mappings for simple operations and pattern-based approaches for more complex transformations. For example, it can detect sequences like MatMul→Add→ReLU and convert them to more efficient fused operations where appropriate.

## Benchmarking System

Rigorous benchmarking is essential for a compiler project. Our benchmarking system focuses on multiple dimensions of performance:

### Compile-Time Metrics

1. **Loading Time**: How quickly we can parse and load models
2. **Conversion Time**: Speed of transforming to our dialect
3. **Optimization Time**: Duration of various optimization passes
4. **Code Generation Time**: Time to produce executable code

These metrics help us identify bottlenecks in the compilation pipeline and ensure developer productivity.

### Runtime Metrics

1. **Inference Latency**: End-to-end execution time
2. **Memory Usage**: Peak and average memory consumption
3. **Throughput**: Inferences per second under various batch sizes
4. **Hardware Utilization**: How efficiently we use available compute resources

### Benchmark Suite

Our benchmark suite includes:
- Standard models (ResNet, BERT, etc.) for comparison with other frameworks
- Microbenchmarks that isolate specific operations
- Composite benchmarks that stress specific aspects of the compiler

Results are presented in both human-readable tables and CSV format for further analysis. The system can also automatically generate comparative visualizations to track performance improvements over time.

## Developer Experience

The Aurora AI Compiler is designed to be extensible, allowing developers to add new operations, optimization passes, or target backends.

### Adding New Operations

To add a new operation to the Aurora dialect:

1. Define the operation in TableGen (`include/Aurora/Dialect/Aurora/AuroraOps.td`):
   ```tablegen
   def Aurora_NewOp : Aurora_Op<"new_op", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
     let summary = "My new operation";
     let description = [{ Detailed description... }];
     
     let arguments = (ins AnyTensor:$input, OptionalAttr<F32Attr>:$alpha);
     let results = (outs AnyTensor:$output);
     
     // Type inference implementation
     let extraClassDeclaration = [{ 
       LogicalResult inferReturnTypes(...);
     }];
   }
   ```

2. Implement the C++ logic (`src/Dialect/Aurora/AuroraOps.cpp`):
   ```cpp
   LogicalResult NewOp::inferReturnTypes(...) {
     // Shape inference implementation
   }
   ```

3. Add conversion logic to the ONNXToMLIRConverter:
   ```python
   def _convert_new_op(self, node, graph):
       # Conversion implementation
   ```

### Extending the Compiler

For more substantial extensions:

1. **Adding Optimization Passes**: Create new passes in `lib/Transforms/`
2. **Supporting New Hardware**: Add new lowering patterns in `lib/Targets/`
3. **New Frontend Support**: Implement new loaders in `python/aurora/model_import/`

Our codebase follows LLVM/MLIR coding standards and includes comprehensive documentation and tests to make the development process smooth.

## Future Work

The Aurora AI Compiler project is evolving rapidly, with several exciting directions planned:

### 1. Runtime Execution Engine

We're developing a lightweight execution engine that can directly run Aurora dialect operations without lowering to LLVM IR, providing faster JIT compilation and easier debugging of models.

### 2. vLLM Integration

Integration with the vLLM project will enable efficient serving of large language models, with special attention to:
- Paged attention mechanisms
- KV-cache optimizations
- Continuous batching for inference

### 3. Hardware Backend Support

We're expanding hardware support beyond CPUs and GPUs to include:
- TPUs and other AI accelerators
- FPGA-based systems
- Mobile and edge devices

### 4. Advanced Optimizations

Future compiler passes will include:
- Mixed-precision inference
- Sparsity-aware execution
- Memory-constrained deployment optimizations
- Quantization-aware training support

### 5. Python API Enhancements

We're working on a more streamlined Python API that will allow:
- Direct integration with PyTorch and TensorFlow
- Interactive optimization and debugging tools
- Custom operation definition from Python

## Conclusion

The Aurora AI Compiler represents a significant step forward in bridging the gap between high-level model definitions and optimized execution. By leveraging MLIR's multi-level approach, we provide a flexible yet powerful compilation pipeline that can target diverse hardware while maintaining optimal performance.

Whether you're deploying models to cloud servers or edge devices, the Aurora AI Compiler offers a path to maximize hardware utilization and minimize inference latency. We invite contributions from the community to help expand the project's capabilities and support an even broader range of models and deployment targets.

---

*Last updated: May 18, 2025*
