// RUN: not aurora-opt %s 2>&1 | FileCheck %s

// Verify that aurora.bias_add rejects a non-rank-1 bias.

func.func @bias_add_wrong_rank(%x: tensor<4x16xf32>, %b: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // CHECK: bias must be rank-1, got rank 2
  %0 = aurora.bias_add(%x, %b) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
