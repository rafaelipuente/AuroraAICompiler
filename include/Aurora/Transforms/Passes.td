//===- Passes.td - Aurora transformation pass definitions --*- tablegen -*-===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_TRANSFORMS_PASSES
#define AURORA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MatMulBiasFusion : Pass<"aurora-matmul-bias-fusion", "::mlir::ModuleOp"> {
  let summary = "Fuse aurora.matmul + aurora.bias_add into aurora.matmul_bias";
  let description = [{
    Detects patterns where an aurora.matmul result feeds directly into an
    aurora.bias_add and rewrites the pair into a single aurora.matmul_bias
    operation. The matmul must have a single use.
  }];
  let constructor = "::mlir::aurora::createMatMulBiasFusionPass()";
}

def Fusion : Pass<"aurora-fusion", "::mlir::ModuleOp"> {
  let summary = "General Aurora operator fusion pass";
  let description = [{
    Applies operator fusion patterns to Aurora IR. Currently contains a
    placeholder Conv+Relu pattern that will be replaced once a proper
    aurora.conv_relu fused op is added to the dialect.
  }];
  let constructor = "::mlir::aurora::createFusionPass()";
}

#endif // AURORA_TRANSFORMS_PASSES
