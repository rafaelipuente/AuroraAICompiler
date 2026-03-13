//===- Fusion.cpp - Aurora general operator fusion pass -------------------===//
//
// Part of the Aurora Compiler Project
//
//===----------------------------------------------------------------------===//

#include "Aurora/Transforms/Fusion.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace aurora {

#define GEN_PASS_DEF_FUSION
#include "Aurora/Transforms/Passes.h.inc"

} // namespace aurora
} // namespace mlir

using namespace mlir;

namespace {

// Placeholder pattern: matches Conv -> Relu but does not perform a real fusion
// because no aurora.conv_relu fused op exists yet. When the fused op is added
// to AuroraOps.td, this pattern should create it instead of dropping relu.
struct ConvReluFusionPattern : public OpRewritePattern<aurora::ReluOp> {
  using OpRewritePattern<aurora::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(aurora::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    auto input = reluOp.getInput();
    auto convOp = dyn_cast_or_null<aurora::ConvOp>(input.getDefiningOp());
    if (!convOp)
      return failure();

    rewriter.replaceOp(reluOp, convOp.getResult());
    return success();
  }
};

} // namespace

namespace mlir {
namespace aurora {

struct FusionPass : impl::FusionBase<FusionPass> {
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      patterns.add<ConvReluFusionPattern>(context);

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

} // namespace aurora
} // namespace mlir
