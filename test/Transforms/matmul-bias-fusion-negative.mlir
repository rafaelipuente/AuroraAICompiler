// RUN: aurora-opt %s --aurora-matmul-bias-fusion | FileCheck %s

// Verify cases where the MatMulBiasFusion pass must NOT fire.

// -----

// The matmul result has two uses: the bias_add and the return.
// The pass requires a single use, so fusion must not happen.

// CHECK-LABEL: func.func @no_fuse_multi_use
// CHECK:         aurora.matmul
// CHECK:         aurora.bias_add
// CHECK-NOT:     aurora.matmul_bias
func.func @no_fuse_multi_use(%a: tensor<4x8xf32>,
                              %b: tensor<8x16xf32>,
                              %bias: tensor<16xf32>)
                              -> (tensor<4x16xf32>, tensor<4x16xf32>) {
  %0 = aurora.matmul(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = aurora.bias_add(%0, %bias) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %0, %1 : tensor<4x16xf32>, tensor<4x16xf32>
}

// -----

// The bias_add's input is not from a matmul. The pattern must not match.

// CHECK-LABEL: func.func @no_fuse_no_matmul
// CHECK:         aurora.bias_add
// CHECK-NOT:     aurora.matmul_bias
func.func @no_fuse_no_matmul(%a: tensor<4x16xf32>,
                              %bias: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.bias_add(%a, %bias) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
