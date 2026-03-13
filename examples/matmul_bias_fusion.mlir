// A two-layer linear network in Aurora IR.
//
// This is the primary demo for AuroraAICompiler. It shows:
//   1. Two matmul+bias_add pairs (the pattern the fusion pass targets).
//   2. A relu activation between layers.
//
// Run the fusion pass with aurora-opt:
//
//   aurora-opt examples/matmul_bias_fusion.mlir --aurora-matmul-bias-fusion
//
// Expected: both matmul+bias_add pairs are fused into aurora.matmul_bias.

module {
  func.func @two_layer_linear(
      %input:  tensor<2x8xf32>,    // batch=2, features=8
      %W0:     tensor<8x16xf32>,   // first layer weights
      %b0:     tensor<16xf32>,     // first layer bias
      %W1:     tensor<16x4xf32>,   // second layer weights
      %b1:     tensor<4xf32>       // second layer bias
  ) -> tensor<2x4xf32> {
    // Layer 0: matmul + bias
    %mm0 = aurora.matmul(%input, %W0)
        : (tensor<2x8xf32>, tensor<8x16xf32>) -> tensor<2x16xf32>
    %z0 = aurora.bias_add(%mm0, %b0)
        : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>

    // Activation
    %a0 = aurora.relu(%z0)
        : (tensor<2x16xf32>) -> tensor<2x16xf32>

    // Layer 1: matmul + bias
    %mm1 = aurora.matmul(%a0, %W1)
        : (tensor<2x16xf32>, tensor<16x4xf32>) -> tensor<2x4xf32>
    %z1 = aurora.bias_add(%mm1, %b1)
        : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>

    return %z1 : tensor<2x4xf32>
  }
}
