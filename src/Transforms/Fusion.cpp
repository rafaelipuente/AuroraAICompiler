#include "Aurora/Transforms/Fusion.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::aurora;

namespace {
// Pattern to fuse Conv+Relu operations into a single fused operation
struct ConvReluFusionPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    // Try to match the Conv -> Relu pattern
    auto input = reluOp.getInput();
    auto convOp = input.getDefiningOp<ConvOp>();

    if (!convOp)
      return failure();

    // Perform the fusion by creating a new ConvOp with fused ReLU
    // In a real implementation, we'd have a dedicated fused op
    // This is just a placeholder for the actual implementation
    rewriter.replaceOp(reluOp, {convOp.getOutput()});
    return success();
  }
};

struct AuroraOperationFusionPass
    : public PassWrapper<AuroraOperationFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AuroraOperationFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AuroraDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    // Add fusion patterns
    patterns.add<ConvReluFusionPattern>(ctx);
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::aurora::createOperationFusionPass() {
  return std::make_unique<AuroraOperationFusionPass>();
}

// Register the pass
namespace {
#define GEN_PASS_REGISTRATION
#include "Aurora/Transforms/Passes.h.inc"
} // end anonymous namespace
