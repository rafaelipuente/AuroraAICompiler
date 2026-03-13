// NOTE: This test requires LLVM 17+.
//   - allow-return-allocs-in-loops was added in LLVM 17; on LLVM 16 use allow-return-allocs=true.
//   - convert-index-to-llvm was added in LLVM 16 but its behaviour stabilised in LLVM 17.
// If you see "unknown option 'allow-return-allocs-in-loops'" you are on LLVM 16 and must
// run these passes manually with the LLVM 16 option names.
//
// RUN: aurora-opt %s \
// RUN:   --convert-aurora-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-in-loops=true" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-index-to-llvm \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm \
// RUN:   --convert-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN: | FileCheck %s
//
// End-to-end pipeline: Aurora dialect -> LLVM dialect (LLVM 17+).
// After all passes:
//   - No Aurora ops remain.
//   - No Linalg structured ops remain.
//   - No tensor ops remain.
//   - No SCF loops remain.
//   - The function is fully in LLVM dialect.

// CHECK-LABEL: llvm.func @relu
// CHECK-NOT:   aurora.
// CHECK-NOT:   linalg.
// CHECK-NOT:   tensor.
// CHECK-NOT:   scf.for
// CHECK:       llvm.return
func.func @relu(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = aurora.relu(%input) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
