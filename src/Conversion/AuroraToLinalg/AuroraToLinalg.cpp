//===- AuroraToLinalg.cpp - Aurora -> Linalg/Arith lowering ---------------===//
//
// Part of the Aurora Compiler Project
//
// Lowers a subset of Aurora dialect ops to Linalg, Arith, and Tensor:
//
//   aurora.relu        -> linalg.generic { arith.maxf(%in, 0.0) }
//   aurora.add         -> linalg.generic { arith.addf }
//   aurora.matmul      -> linalg.matmul (zero-filled accumulator)
//   aurora.bias_add    -> linalg.generic (broadcast-add along last dim)
//   aurora.matmul_bias -> linalg.matmul + linalg.generic (bias broadcast-add)
//
// Uses partial conversion so unlowered ops (conv, layernorm, etc.) remain.
//
//===----------------------------------------------------------------------===//

#include "Aurora/Conversion/Passes.h"
#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aurora;

namespace {

//===----------------------------------------------------------------------===//
// Helper: create a tensor.empty of the same type as `ref`.
//===----------------------------------------------------------------------===//

static Value createEmptyTensorLike(OpBuilder &b, Location loc, Value ref) {
  auto ty = cast<RankedTensorType>(ref.getType());
  SmallVector<Value> dynDims;
  for (int64_t i = 0, e = ty.getRank(); i < e; ++i) {
    if (ty.isDynamicDim(i))
      dynDims.push_back(b.create<tensor::DimOp>(loc, ref, i));
  }
  return b.create<tensor::EmptyOp>(loc, ty.getShape(), ty.getElementType(),
                                   dynDims);
}

//===----------------------------------------------------------------------===//
// aurora.relu -> linalg.generic { arith.maxf(x, 0.0) }
//===----------------------------------------------------------------------===//

struct ReluLowering : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultTy = cast<RankedTensorType>(op.getType());

    Value init = createEmptyTensorLike(rewriter, loc, input);

    int64_t rank = resultTy.getRank();
    SmallVector<AffineMap> indexingMaps(
        2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultTy, /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{init}, indexingMaps, iterators,
        [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value zero = nested.create<arith::ConstantOp>(
              nestedLoc, nested.getFloatAttr(resultTy.getElementType(), 0.0));
          Value relu = nested.create<arith::MaxFOp>(nestedLoc, args[0], zero);
          nested.create<linalg::YieldOp>(nestedLoc, relu);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// aurora.add -> linalg.generic { arith.addf }
//===----------------------------------------------------------------------===//

struct AddLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultTy = cast<RankedTensorType>(op.getType());

    Value init = createEmptyTensorLike(rewriter, loc, lhs);

    int64_t rank = resultTy.getRank();
    SmallVector<AffineMap> indexingMaps(
        3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultTy, /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{init}, indexingMaps, iterators,
        [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value sum = nested.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nestedLoc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// aurora.matmul -> linalg.matmul
//===----------------------------------------------------------------------===//

struct MatMulLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultTy = cast<RankedTensorType>(op.getType());

    // linalg.matmul accumulates into a zero-filled init tensor.
    Value init = createEmptyTensorLike(rewriter, loc, op.getResult());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultTy.getElementType(), 0.0));
    Value filled = rewriter.create<linalg::FillOp>(loc, zero, init)
                       .getResult(0);

    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, resultTy, ValueRange{lhs, rhs}, ValueRange{filled});

    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// aurora.bias_add -> linalg.generic (broadcast-add along last dim)
//
// input: tensor<?x...xNxf32>   bias: tensor<Nxf32>
// The bias is broadcast along all dimensions except the last.
//===----------------------------------------------------------------------===//

struct BiasAddLowering : public OpRewritePattern<BiasAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BiasAddOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value bias = op.getBias();
    auto resultTy = cast<RankedTensorType>(op.getType());

    Value init = createEmptyTensorLike(rewriter, loc, input);

    int64_t rank = resultTy.getRank();

    // input map: identity (d0, d1, ..., d_{rank-1}) -> (d0, d1, ..., d_{rank-1})
    AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);
    // bias map: project to last dim only: (d0, d1, ..., d_{rank-1}) -> (d_{rank-1})
    AffineMap biasMap = AffineMap::get(
        rank, 0, rewriter.getAffineDimExpr(rank - 1), rewriter.getContext());
    // output map: identity
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(rank);

    SmallVector<AffineMap> indexingMaps = {inputMap, biasMap, outputMap};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultTy, /*inputs=*/ValueRange{input, bias},
        /*outputs=*/ValueRange{init}, indexingMaps, iterators,
        [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value sum = nested.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nestedLoc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// aurora.matmul_bias -> linalg.matmul + linalg.generic (bias broadcast-add)
//
// aurora.matmul_bias(lhs: MxK, rhs: KxN, bias: N) -> MxN
// is semantically: matmul(lhs, rhs) then bias_add(result, bias).
// We lower it directly to those two Linalg ops so the full fusion+lowering
// pipeline leaves no Aurora ops behind.
//===----------------------------------------------------------------------===//

struct MatMulBiasLowering : public OpRewritePattern<MatMulBiasOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulBiasOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value bias = op.getBias();
    auto resultTy = cast<RankedTensorType>(op.getType());

    // Step 1: linalg.matmul into a zero-filled accumulator.
    Value mmInit = createEmptyTensorLike(rewriter, loc, op.getResult());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultTy.getElementType(), 0.0));
    Value filled =
        rewriter.create<linalg::FillOp>(loc, zero, mmInit).getResult(0);

    Value mmResult = rewriter
                         .create<linalg::MatmulOp>(
                             loc, resultTy, ValueRange{lhs, rhs},
                             ValueRange{filled})
                         .getResult(0);

    // Step 2: broadcast-add the bias along the last dimension.
    // bias map: (d0, d1) -> (d1)  [project to last dim]
    int64_t rank = resultTy.getRank();
    AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);
    AffineMap biasMap = AffineMap::get(rank, 0,
                                       rewriter.getAffineDimExpr(rank - 1),
                                       rewriter.getContext());
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(rank);

    SmallVector<AffineMap> indexingMaps = {inputMap, biasMap, outputMap};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value biasInit = createEmptyTensorLike(rewriter, loc, op.getResult());
    auto biasAdd = rewriter.create<linalg::GenericOp>(
        loc, resultTy, /*inputs=*/ValueRange{mmResult, bias},
        /*outputs=*/ValueRange{biasInit}, indexingMaps, iterators,
        [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value sum =
              nested.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nestedLoc, sum);
        });

    rewriter.replaceOp(op, biasAdd.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_CONVERTAURORATOLINALG
#include "Aurora/Conversion/Passes.h.inc"

struct ConvertAuroraToLinalgPass
    : public impl::ConvertAuroraToLinalgBase<ConvertAuroraToLinalgPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    // Aurora ops that we lower become illegal; the rest stay legal.
    target.addIllegalOp<ReluOp, AddOp, MatMulOp, BiasAddOp, MatMulBiasOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    patterns.add<ReluLowering, AddLowering, MatMulLowering, BiasAddLowering,
                 MatMulBiasLowering>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::aurora::createConvertAuroraToLinalgPass() {
  return std::make_unique<ConvertAuroraToLinalgPass>();
}
