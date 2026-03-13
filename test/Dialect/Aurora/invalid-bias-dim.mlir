// RUN: not aurora-opt %s 2>&1 | FileCheck %s

// Verify that aurora.bias_add rejects a bias whose length does not match
// the last dimension of the input.

func.func @bias_add_dim_mismatch(%x: tensor<4x16xf32>, %b: tensor<8xf32>) -> tensor<4x16xf32> {
  // CHECK: bias length (8) must match the last dimension of input (16)
  %0 = aurora.bias_add(%x, %b) : (tensor<4x16xf32>, tensor<8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
