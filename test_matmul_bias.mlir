module {
  func.func @test_fusion(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    // MatMul operation
    %0 = aurora.matmul(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    
    // Add bias operation
    %1 = aurora.add(%0, %arg2) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    
    // Return the result
    return %1 : tensor<4x16xf32>
  }
}
