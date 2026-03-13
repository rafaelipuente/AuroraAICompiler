//===- MatMulBiasFusion.cpp - MatMul+Bias fusion pass --------------------===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//

#include "Aurora/Transforms/MatMulBiasFusion.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aurora-matmul-bias-fusion"

namespace mlir {
namespace aurora {

#define GEN_PASS_DEF_MATMULBIASFUSION
#include "Aurora/Transforms/Passes.h.inc"

} // namespace aurora
} // namespace mlir

using namespace mlir;

namespace {

/// Detect aurora.matmul feeding into aurora.bias_add and rewrite into
/// aurora.matmul_bias. The bias_add op's verifier guarantees the bias is
/// rank-1 and its length matches the matmul output's last dimension, so
/// the pattern only needs to check for single-use and 2D matmul output.
struct MatMulBiasAddPattern : public OpRewritePattern<aurora::BiasAddOp> {
  using OpRewritePattern<aurora::BiasAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(aurora::BiasAddOp biasAddOp,
                                PatternRewriter &rewriter) const override {
    Value input = biasAddOp.getInput();
    auto matmulOp = input.getDefiningOp<aurora::MatMulOp>();
    if (!matmulOp)
      return failure();

    if (!input.hasOneUse())
      return failure();

    auto matmulOutType = input.getType().dyn_cast<RankedTensorType>();
    if (!matmulOutType || matmulOutType.getRank() != 2)
      return failure();

    Value bias = biasAddOp.getBias();
    Value matLhs = matmulOp.getOperand(0);
    Value matRhs = matmulOp.getOperand(1);

    auto fusedOp = rewriter.create<aurora::MatMulBiasOp>(
        biasAddOp.getLoc(), biasAddOp.getResult().getType(),
        matLhs, matRhs, bias);

    LLVM_DEBUG({
      llvm::dbgs() << "Fusing matmul + bias_add into matmul_bias:\n";
      llvm::dbgs() << "  matmul:   " << matmulOp << "\n";
      llvm::dbgs() << "  bias_add: " << biasAddOp << "\n";
      llvm::dbgs() << "  fused:    " << fusedOp << "\n";
    });

    rewriter.replaceOp(biasAddOp, fusedOp.getResult());
    return success();
  }
};

} // namespace

namespace mlir {
namespace aurora {

struct MatMulBiasFusionPass
    : impl::MatMulBiasFusionBase<MatMulBiasFusionPass> {
  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      patterns.add<MatMulBiasAddPattern>(context);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns),
                                              config)))
        return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createMatMulBiasFusionPass() {
  return std::make_unique<MatMulBiasFusionPass>();
}

} // namespace aurora
} // namespace mlir
