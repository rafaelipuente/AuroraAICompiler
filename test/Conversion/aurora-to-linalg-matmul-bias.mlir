// RUN: aurora-opt %s --convert-aurora-to-linalg | FileCheck %s

// Verify that aurora.matmul_bias lowers to linalg.matmul followed by a
// broadcast bias-add (linalg.generic + arith.addf), without needing the
// fusion pass to be active.

// CHECK-LABEL: func.func @matmul_bias
// CHECK-NOT: aurora.matmul_bias
// CHECK: arith.constant 0.000000e+00
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.yield
func.func @matmul_bias(
    %lhs:  tensor<4x8xf32>,
    %rhs:  tensor<8x16xf32>,
    %bias: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.matmul_bias(%lhs, %rhs, %bias)
      : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
