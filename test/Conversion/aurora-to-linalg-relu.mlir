// RUN: aurora-opt %s --convert-aurora-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @relu
// CHECK-NOT: aurora.relu
// CHECK: linalg.generic
// CHECK: arith.constant 0.000000e+00
// CHECK: arith.max
// CHECK: linalg.yield
func.func @relu(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = aurora.relu(%arg0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
