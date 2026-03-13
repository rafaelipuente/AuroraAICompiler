// RUN: aurora-opt %s --aurora-matmul-bias-fusion | FileCheck %s

// Verify that the MatMulBiasFusion pass rewrites matmul + bias_add into
// matmul_bias when the matmul result has a single use.

// CHECK-LABEL: func.func @fuse_matmul_bias_add
// CHECK-NOT:     aurora.matmul
// CHECK-NOT:     aurora.bias_add
// CHECK:         aurora.matmul_bias
// CHECK:         return
func.func @fuse_matmul_bias_add(%a: tensor<4x8xf32>,
                                 %b: tensor<8x16xf32>,
                                 %bias: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.matmul(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = aurora.bias_add(%0, %bias) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}
