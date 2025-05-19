// This MLIR file demonstrates the result after our MatMulBias fusion pass is applied
// The separate matmul and add operations have been fused into a single matmul_bias operation

module {
  func.func @matmul_bias_pattern(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    // Single fused matrix multiplication with bias operation
    %0 = aurora.matmul_bias(%arg0, %arg1, %arg2) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    
    // Return the result
    return %0 : tensor<4x16xf32>
  }
}
