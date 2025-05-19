// Simple test model for Aurora compiler
module {
  func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
    // Use Aurora MatMul operation
    %0 = "aurora.matmul"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    // Add a ReLU activation
    %1 = "aurora.relu"(%0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }
}
