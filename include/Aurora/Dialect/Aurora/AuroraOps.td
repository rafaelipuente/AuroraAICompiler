//===- AuroraOps.td - Aurora dialect operations -----------*- tablegen -*-===//
//
// This file contains the definitions of the Aurora dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_OPS
#define AURORA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Aurora dialect definition
//===----------------------------------------------------------------------===//

def Aurora_Dialect : Dialect {
  let name = "aurora";
  let cppNamespace = "::mlir::aurora";
  
  let summary = "A dialect for Aurora AI operations";
  let description = [{
    The Aurora dialect is designed for high-performance AI operations.
  }];

  // Use the updated fold API to remove deprecation warnings
  let useFoldAPI = kEmitFoldAdaptorFolder;
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class Aurora_Op<string mnemonic, list<Trait> traits = []> :
    Op<Aurora_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Aurora operations
//===----------------------------------------------------------------------===//

def Aurora_AddOp : Aurora_Op<"add", [
    Pure,
    SameOperandsAndResultType
  ]> {
  let summary = "Element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition on two tensors.
    
    Example:
    ```mlir
    %result = aurora.add %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
  
  let arguments = (ins 
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  
  let results = (outs AnyTensor:$result);
}

def Aurora_ReluOp : Aurora_Op<"relu", [
    Pure,
    SameOperandsAndResultType
  ]> {
  let summary = "ReLU activation function";
  let description = [{
    The "relu" operation implements the Rectified Linear Unit activation function.
    
    Example:
    ```mlir
    %result = aurora.relu %input : tensor<4x4xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def Aurora_MatMulOp : Aurora_Op<"matmul", [
    Pure
  ]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication between two 2D tensors.
    
    Example:
    ```mlir
    %result = aurora.matmul %a, %b : tensor<MxKxf32>, tensor<KxNxf32> -> tensor<MxNxf32>
    ```
  }];
  
  let arguments = (ins 
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  
  let results = (outs AnyTensor:$result);
}

def Aurora_MatMulBiasOp : Aurora_Op<"matmul_bias", [
    Pure
  ]> {
  let summary = "Matrix multiplication with bias addition";
  let description = [{
    Performs matrix multiplication followed by a bias addition.
    
    Example:
    ```mlir
    %result = aurora.matmul_bias %a, %b, %bias : tensor<MxKxf32>, tensor<KxNxf32>, tensor<Nxf32> -> tensor<MxNxf32>
    ```
  }];
  
  let arguments = (ins 
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    AnyTensor:$bias
  );
  
  let results = (outs AnyTensor:$result);
}

def Aurora_ConvOp : Aurora_Op<"conv", [
    Pure
  ]> {
  let summary = "Convolution operation";
  let description = [{
    Performs convolution operation for neural networks.
    
    Example:
    ```mlir
    %result = aurora.conv %input, %filter : tensor<NxCxHxWxf32>, tensor<KxCxFHxFWxf32> -> tensor<NxKxOHxOWxf32>
    ```
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$filter
  );
  
  let results = (outs AnyTensor:$result);
}

def Aurora_LayerNormOp : Aurora_Op<"layernorm", [
    Pure
  ]> {
  let summary = "Layer normalization operation";
  let description = [{
    Applies layer normalization to the input tensor.
    
    Example:
    ```mlir
    %result = aurora.layernorm %input, %gamma, %beta : tensor<*xf32>, tensor<*xf32>, tensor<*xf32> -> tensor<*xf32>
    ```
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$gamma,
    AnyTensor:$beta
  );
  
  let results = (outs AnyTensor:$result);
}

def Aurora_FusedAttentionOp : Aurora_Op<"fused_attention", [
    Pure
  ]> {
  let summary = "Fused attention operation";
  let description = [{
    Implements the entire attention mechanism as a single fused operation for efficiency.
    
    Example:
    ```mlir
    %result = aurora.fused_attention %query, %key, %value : tensor<BxNxHxDkxf32>, tensor<BxMxHxDkxf32>, tensor<BxMxHxDvxf32> -> tensor<BxNxHxDvxf32>
    ```
  }];
  
  let arguments = (ins 
    AnyTensor:$query,
    AnyTensor:$key,
    AnyTensor:$value
  );
  
  let results = (outs AnyTensor:$result);
}

#endif // AURORA_OPS
