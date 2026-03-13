// RUN: aurora-opt %s --aurora-matmul-bias-fusion --convert-aurora-to-linalg | FileCheck %s
//
// Full pipeline test: fuse matmul+bias_add, then lower everything to Linalg.
// After both passes no Aurora ops should remain.

// CHECK-LABEL: func.func @fuse_then_lower
// CHECK-NOT: aurora.
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.max
func.func @fuse_then_lower(
    %input: tensor<2x8xf32>,
    %W: tensor<8x16xf32>,
    %b: tensor<16xf32>) -> tensor<2x16xf32> {
  %mm = aurora.matmul(%input, %W)
      : (tensor<2x8xf32>, tensor<8x16xf32>) -> tensor<2x16xf32>
  %z = aurora.bias_add(%mm, %b)
      : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  %a = aurora.relu(%z) : (tensor<2x16xf32>) -> tensor<2x16xf32>
  return %a : tensor<2x16xf32>
}
