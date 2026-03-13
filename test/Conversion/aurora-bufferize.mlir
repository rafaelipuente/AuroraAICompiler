// Requires LLVM 17 (the only supported version).
//
// RUN: aurora-opt %s \
// RUN:   --convert-aurora-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs=true" \
// RUN: | FileCheck %s
//
// Verifies the Aurora -> Linalg -> bufferized-memref stage.
// After one-shot-bufferize with function-boundary bufferization:
//   - Tensor arguments become memref arguments.
//   - tensor.empty() becomes memref.alloc().
//   - linalg ops operate on memrefs.
//   - No tensor-typed SSA values remain inside the function.

// CHECK-LABEL: func.func @relu
// Function signature is rewritten: tensor args become memref args.
// CHECK-SAME: memref<4x4xf32>
// CHECK-NOT:  tensor<
// CHECK:      memref.alloc
// CHECK:      linalg.generic
// CHECK-NOT:  tensor.empty
func.func @relu(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = aurora.relu(%input) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
