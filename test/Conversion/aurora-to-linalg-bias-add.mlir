// RUN: aurora-opt %s --convert-aurora-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @bias_add
// CHECK-NOT: aurora.bias_add
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.yield
func.func @bias_add(%input: tensor<4x16xf32>, %bias: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.bias_add(%input, %bias) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
