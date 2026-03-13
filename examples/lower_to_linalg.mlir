// A small Aurora IR module that demonstrates lowering to Linalg/Arith.
//
// Run:
//   aurora-opt examples/lower_to_linalg.mlir --convert-aurora-to-linalg
//
// Or combine fusion + lowering:
//   aurora-opt examples/matmul_bias_fusion.mlir \
//     --aurora-matmul-bias-fusion --convert-aurora-to-linalg

module {
  func.func @matmul_relu(
      %A: tensor<4x8xf32>,
      %B: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %mm = aurora.matmul(%A, %B)
        : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %r = aurora.relu(%mm) : (tensor<4x16xf32>) -> tensor<4x16xf32>
    return %r : tensor<4x16xf32>
  }
}
