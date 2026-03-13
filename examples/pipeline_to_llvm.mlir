// Full Aurora -> LLVM dialect pipeline demonstration.
//
// This file shows the deepest lowering path currently supported:
//
//   aurora.matmul + aurora.relu (Aurora dialect)
//     -> linalg.matmul + linalg.generic (Linalg on tensors)
//     -> linalg.matmul + linalg.generic on memrefs (after bufferization)
//     -> scf.for loops + memref.load/store (after linalg-to-loops)
//     -> cf.br / cf.cond_br + llvm.load/store (after scf-to-cf)
//     -> llvm.func + llvm.call @malloc + LLVM ops (after to-llvm)
//
// STEP 1 - Aurora ops to Linalg (tensor-based):
//   aurora-opt examples/pipeline_to_llvm.mlir --convert-aurora-to-linalg
//
// STEP 2 - Fuse first, then lower everything to Linalg:
//   aurora-opt examples/matmul_bias_fusion.mlir \
//     --aurora-matmul-bias-fusion --convert-aurora-to-linalg
//
// STEP 3 - Full pipeline to LLVM dialect (requires LLVM 17+):
//   aurora-opt examples/pipeline_to_llvm.mlir \
//     --convert-aurora-to-linalg \
//     --one-shot-bufferize="bufferize-function-boundaries=true \
//                           allow-return-allocs-in-loops=true" \
//     --convert-linalg-to-loops \
//     --convert-scf-to-cf \
//     --convert-index-to-llvm \
//     --convert-arith-to-llvm \
//     --convert-cf-to-llvm \
//     --convert-memref-to-llvm \
//     --convert-func-to-llvm \
//     --reconcile-unrealized-casts
//
// On LLVM 16, replace allow-return-allocs-in-loops with allow-return-allocs.
//
// STEP 3a - Same via pipeline string:
//   aurora-opt examples/pipeline_to_llvm.mlir \
//     --pass-pipeline="builtin.module( \
//       convert-aurora-to-linalg, \
//       one-shot-bufferize{bufferize-function-boundaries=true \
//                          allow-return-allocs-in-loops=true}, \
//       convert-linalg-to-loops, \
//       convert-scf-to-cf, \
//       convert-index-to-llvm, \
//       convert-arith-to-llvm, \
//       convert-cf-to-llvm, \
//       convert-memref-to-llvm, \
//       convert-func-to-llvm, \
//       reconcile-unrealized-casts)"
//
// WHAT STOPS HERE (what the output is NOT):
//   - This produces LLVM *dialect* IR (not LLVM IR text / bitcode).
//   - To produce real machine code, pipe the output through mlir-translate
//     then llc, or use the MLIR JIT / ORC APIs. Those are out of scope
//     for this prototype.

module {
  func.func @matmul_relu(
      %A: tensor<4x8xf32>,
      %B: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %mm = aurora.matmul(%A, %B)
        : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %r = aurora.relu(%mm) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %r : tensor<4x4xf32>
  }
}
