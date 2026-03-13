// NOTE: This test uses --one-shot-bufferize with the allow-return-allocs-in-loops
// option, which was added in LLVM 17.  On LLVM 16, replace the option with:
//   allow-return-allocs=true
// If you see "unknown option 'allow-return-allocs-in-loops'", you are on LLVM 16.
//
// RUN: aurora-opt %s \
// RUN:   --convert-aurora-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-in-loops=true" \
// RUN: | FileCheck %s
//
// Verifies the Aurora -> Linalg -> bufferized-memref stage (LLVM 17+).
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
