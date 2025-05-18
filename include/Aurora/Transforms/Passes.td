#ifndef AURORA_TRANSFORMS_PASSES
#define AURORA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def OperationFusion : Pass<"aurora-fusion", "func::FuncOp"> {
  let summary = "Fuse Aurora operations for better performance";
  let description = [{
    This pass performs fusion of Aurora operations to improve performance.
    It looks for patterns like Conv+Relu that can be executed as a single
    fused operation for better performance.
  }];
}

def BufferAllocation : Pass<"aurora-buffer-allocation", "func::FuncOp"> {
  let summary = "Allocate buffers for Aurora operations";
  let description = [{
    This pass performs buffer allocation for Aurora operations,
    transforming them from tensor-based representation to buffer-based
    to prepare for code generation.
  }];
}

def LoopOptimization : Pass<"aurora-loop-opt", "func::FuncOp"> {
  let summary = "Optimize loops in Aurora operations";
  let description = [{
    This pass performs loop optimizations like tiling, unrolling, and
    parallelization for Aurora operations to improve performance on
    target hardware.
  }];
}

#endif // AURORA_TRANSFORMS_PASSES
