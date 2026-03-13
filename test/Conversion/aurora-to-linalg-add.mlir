// RUN: aurora-opt %s --convert-aurora-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK-NOT: aurora.add
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.yield
func.func @add(%lhs: tensor<4x8xf32>, %rhs: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = aurora.add %lhs, %rhs : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
