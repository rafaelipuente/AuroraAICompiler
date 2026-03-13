// RUN: aurora-opt %s | aurora-opt | FileCheck %s

// Verify that all Aurora dialect operations roundtrip through parse/print.
// Each op is parsed by the first aurora-opt, printed to stdout, re-parsed
// by the second aurora-opt, and the final output is checked.

// CHECK-LABEL: func.func @test_add
// CHECK: aurora.add
func.func @test_add(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = aurora.add(%a, %b) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_bias_add
// CHECK: aurora.bias_add
func.func @test_bias_add(%x: tensor<4x16xf32>, %b: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.bias_add(%x, %b) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @test_relu
// CHECK: aurora.relu
func.func @test_relu(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = aurora.relu(%a) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_matmul
// CHECK: aurora.matmul
func.func @test_matmul(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.matmul(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @test_matmul_bias
// CHECK: aurora.matmul_bias
func.func @test_matmul_bias(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>,
                             %bias: tensor<16xf32>) -> tensor<4x16xf32> {
  %0 = aurora.matmul_bias(%a, %b, %bias) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @test_conv
// CHECK: aurora.conv
func.func @test_conv(%input: tensor<1x3x224x224xf32>,
                      %filter: tensor<64x3x7x7xf32>) -> tensor<1x64x218x218xf32> {
  %0 = aurora.conv(%input, %filter) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x218x218xf32>
  return %0 : tensor<1x64x218x218xf32>
}

// CHECK-LABEL: func.func @test_layernorm
// CHECK: aurora.layernorm
func.func @test_layernorm(%input: tensor<2x8xf32>, %gamma: tensor<8xf32>,
                           %beta: tensor<8xf32>) -> tensor<2x8xf32> {
  %0 = aurora.layernorm(%input, %gamma, %beta) : (tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @test_fused_attention
// CHECK: aurora.fused_attention
func.func @test_fused_attention(%q: tensor<2x4x8x16xf32>,
                                 %k: tensor<2x4x8x16xf32>,
                                 %v: tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32> {
  %0 = aurora.fused_attention(%q, %k, %v) : (tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>, tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  return %0 : tensor<2x4x8x32xf32>
}
