// Demonstrates the pattern before MatMulBias fusion:
// matmul followed by bias addition.

module {
  func.func @matmul_bias_pattern(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    %0 = aurora.matmul(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %1 = aurora.bias_add(%0, %arg2) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    return %1 : tensor<4x16xf32>
  }
}
