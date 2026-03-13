//===- Passes.td - Aurora conversion pass definitions ------*- tablegen -*-===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_CONVERSION_PASSES
#define AURORA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAuroraToLinalg : Pass<"convert-aurora-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower Aurora dialect ops to Linalg/Arith/Func";
  let description = [{
    Lowers a subset of Aurora dialect operations to standard MLIR dialects:

      aurora.relu        -> linalg.generic + arith.maxf (element-wise max with zero)
      aurora.add         -> linalg.generic + arith.addf (element-wise addition)
      aurora.matmul      -> linalg.matmul (zero-filled accumulator)
      aurora.bias_add    -> linalg.generic (broadcast-add along last dimension)
      aurora.matmul_bias -> linalg.matmul + linalg.generic (matmul then bias-add)

    Ops without a lowering pattern (conv, layernorm, fused_attention) are left
    in place; the conversion uses partial lowering so unlowered ops do not
    cause a failure.
  }];
  let constructor = "::mlir::aurora::createConvertAuroraToLinalgPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // AURORA_CONVERSION_PASSES
