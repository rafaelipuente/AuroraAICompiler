//===- MatMulBiasFusion.cpp - Implementation of MatMul+Bias fusion -------===//
//
// Aurora AI Compiler
//
//===----------------------------------------------------------------------===//

#include "Aurora/Transforms/MatMulBiasFusion.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aurora-matmul-bias-fusion"

using namespace mlir;

// Anonymous namespace for local patterns
namespace {

/// Pattern to detect and replace MatMul + Add with MatMulBias
struct MatMulAddPattern : public OpRewritePattern<mlir::aurora::AddOp> {
  /// Inherit constructors from OpRewritePattern
  using OpRewritePattern<mlir::aurora::AddOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(mlir::aurora::AddOp addOp,
                              PatternRewriter &rewriter) const override {
    // Check if this is a bias addition by examining operands
    // Get the result of the AddOp (single result)
    Value addOutput = addOp.getResult();
    // Get the operands of the AddOp (binary operation)
    Value lhsOperand = addOp.getOperand(0);
    Value rhsOperand = addOp.getOperand(1);
    
    // The add operation must have a matmul operation as one of its inputs
    Value matmulOutput = nullptr;
    Value bias = nullptr;
    mlir::aurora::MatMulOp matmulOp = nullptr;
    
    // Check which of the add inputs might be from a matmul
    auto *lhsDefiningOp = lhsOperand.getDefiningOp();
    if (auto definingOp = dyn_cast_or_null<mlir::aurora::MatMulOp>(lhsDefiningOp)) {
      matmulOp = definingOp;
      matmulOutput = lhsOperand;
      bias = rhsOperand;
    } else {
      auto *rhsDefiningOp = rhsOperand.getDefiningOp();
      if (auto definingOp = dyn_cast_or_null<mlir::aurora::MatMulOp>(rhsDefiningOp)) {
        matmulOp = definingOp;
        matmulOutput = rhsOperand;
        bias = lhsOperand;
      } else {
        // Neither operand comes from a matmul operation
        return failure();
      }
    }
    
    // Ensure the matmul result has only one use (the add operation)
    // This is to make sure we don't break other uses of the matmul result
    if (!matmulOutput.hasOneUse()) {
      return failure();
    }
    
    // Ensure the bias has the right shape for fusion
    auto matmulOutputType = matmulOutput.getType().dyn_cast<RankedTensorType>();
    auto biasType = bias.getType().dyn_cast<RankedTensorType>();
    
    if (!matmulOutputType || !biasType) {
      return failure(); // Unranked tensors not supported
    }
    
    // Check that bias has correct rank for broadcasting (1D or 2D)
    unsigned biasRank = biasType.getRank();
    if (biasRank > 2) {
      return failure(); // Only 1D and 2D biases supported
    }
    
    // Extract matrix dimensions
    auto matmulOutShape = matmulOutputType.getShape();
    if (matmulOutputType.getRank() != 2) {
      return failure(); // Only 2D matmul supported
    }
    
    int64_t M = matmulOutShape[0];
    int64_t N = matmulOutShape[1];
    
    // Verify bias shape compatibility
    auto biasShape = biasType.getShape();
    bool validBias = false;
    
    if (biasRank == 1) {
      // 1D bias must have length N for broadcasting
      validBias = (biasShape[0] == N);
    } else if (biasRank == 2) {
      // 2D bias must be broadcastable to [M, N]
      validBias = (biasShape[0] == 1 || biasShape[0] == M) &&
                 (biasShape[1] == 1 || biasShape[1] == N);
    }
    
    if (!validBias) {
      return failure();
    }
    
    // Get matmul operands and attributes
    // Get MatMulOp operands
    Value lhs = matmulOp.getOperand(0);
    Value rhs = matmulOp.getOperand(1);
    
    // Create the fused matmul_bias operation
    // Create a MatMulBiasOp using the correct builder signature and proper namespace
    auto fusedOp = rewriter.create<mlir::aurora::MatMulBiasOp>(
        addOp.getLoc(),
        addOutput.getType(),
        lhs, rhs, bias);
    
    LLVM_DEBUG({
      llvm::dbgs() << "Fusing matmul and add ops into matmul_bias:\n";
      llvm::dbgs() << "  matmul: " << matmulOp << "\n";
      llvm::dbgs() << "  add: " << addOp << "\n";
      llvm::dbgs() << "  result: " << fusedOp << "\n";
    });
    
    // Replace the add operation with the fused operation result
    rewriter.replaceOp(addOp, fusedOp.getResult());
    
    return success();
  }
};

} // end anonymous namespace

namespace mlir {
namespace aurora {

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void MatMulBiasFusionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  
  // Add pattern to fusion driver
  RewritePatternSet patterns(context);
  patterns.add<MatMulAddPattern>(context);
  
  // Configure and run the pattern matcher
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = 10;
  
  // Apply to each function in the module
  for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns), config);
  }
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

/// Creates a pass to perform MatMul+Bias fusion
std::unique_ptr<Pass> createMatMulBiasFusionPass() {
  return std::make_unique<mlir::aurora::MatMulBiasFusionPass>();
}

} // namespace aurora
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

// Note: In a full implementation, we would use TableGen to generate pass registration
// For now, we use a simple implementation that matches our header
