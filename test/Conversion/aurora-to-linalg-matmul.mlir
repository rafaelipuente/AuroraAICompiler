// RUN: aurora-opt %s --convert-aurora-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @matmul
// CHECK-NOT: aurora.matmul
// CHECK: arith.constant 0.000000e+00
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @matmul(%lhs: tensor<4x8xf32>, %rhs: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.matmul(%lhs, %rhs) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
