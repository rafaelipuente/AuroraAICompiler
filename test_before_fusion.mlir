// This MLIR file demonstrates the pattern that our MatMulBias fusion pass would optimize
// Here we have a matrix multiplication followed by a bias addition

module {
  func.func @matmul_bias_pattern(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    // First, matrix multiplication operation
    %0 = aurora.matmul(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    
    // Then, bias addition operation
    %1 = aurora.add(%0, %arg2) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    
    // Return the result
    return %1 : tensor<4x16xf32>
  }
}
