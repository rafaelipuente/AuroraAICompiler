#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"

using namespace mlir;
using namespace mlir::aurora;

namespace {

class AuroraDialectTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.getOrLoadDialect<AuroraDialect>();
  }
  
  MLIRContext context;
};

// Test creating a basic Aurora operation
TEST_F(AuroraDialectTest, CreateConvOp) {
  // Setup
  OpBuilder builder(&context);
  
  // Create location
  Location loc = builder.getUnknownLoc();
  
  // Create tensor types for input and filter
  auto inputType = RankedTensorType::get({1, 64, 28, 28}, builder.getF32Type());
  auto filterType = RankedTensorType::get({128, 64, 3, 3}, builder.getF32Type());
  auto resultType = RankedTensorType::get({1, 128, 26, 26}, builder.getF32Type());
  
  // Create input and filter values (these would be real values in a full test)
  auto input = builder.create<arith::ConstantOp>(loc, inputType, builder.getZeroAttr(inputType));
  auto filter = builder.create<arith::ConstantOp>(loc, filterType, builder.getZeroAttr(filterType));
  
  // Create attributes
  auto stridesAttr = builder.getI64ArrayAttr({1, 1});
  auto paddingsAttr = builder.getI64ArrayAttr({0, 0, 0, 0});
  auto dilationsAttr = builder.getI64ArrayAttr({1, 1});
  auto groupsAttr = builder.getI64IntegerAttr(1);
  
  // Create AuroraConvOp
  auto convOp = builder.create<ConvOp>(
      loc, resultType, input, filter, 
      stridesAttr, paddingsAttr, dilationsAttr, groupsAttr);
  
  // Verify operation was created successfully
  ASSERT_TRUE(convOp);
  EXPECT_EQ(convOp.getStrides(), stridesAttr);
  EXPECT_EQ(convOp.getPaddings(), paddingsAttr);
  EXPECT_EQ(convOp.getDilations(), dilationsAttr);
  EXPECT_EQ(convOp.getGroups(), groupsAttr);
}

// Test creating a matmul operation
TEST_F(AuroraDialectTest, CreateMatMulOp) {
  // Setup
  OpBuilder builder(&context);
  
  // Create location
  Location loc = builder.getUnknownLoc();
  
  // Create tensor types
  auto lhsType = RankedTensorType::get({4, 8}, builder.getF32Type());
  auto rhsType = RankedTensorType::get({8, 16}, builder.getF32Type());
  auto resultType = RankedTensorType::get({4, 16}, builder.getF32Type());
  
  // Create input values
  auto lhs = builder.create<arith::ConstantOp>(loc, lhsType, builder.getZeroAttr(lhsType));
  auto rhs = builder.create<arith::ConstantOp>(loc, rhsType, builder.getZeroAttr(rhsType));
  
  // Create AuroraMatMulOp
  auto matmulOp = builder.create<MatMulOp>(loc, resultType, lhs, rhs);
  
  // Verify operation was created successfully
  ASSERT_TRUE(matmulOp);
  EXPECT_EQ(matmulOp.getLhs(), lhs.getResult());
  EXPECT_EQ(matmulOp.getRhs(), rhs.getResult());
}

// Test creating a ReLU operation
TEST_F(AuroraDialectTest, CreateReluOp) {
  // Setup
  OpBuilder builder(&context);
  
  // Create location
  Location loc = builder.getUnknownLoc();
  
  // Create tensor type
  auto inputType = RankedTensorType::get({1, 64, 56, 56}, builder.getF32Type());
  
  // Create input value
  auto input = builder.create<arith::ConstantOp>(loc, inputType, builder.getZeroAttr(inputType));
  
  // Create AuroraReluOp
  auto reluOp = builder.create<ReluOp>(loc, inputType, input);
  
  // Verify operation was created successfully
  ASSERT_TRUE(reluOp);
  EXPECT_EQ(reluOp.getInput(), input.getResult());
}

} // namespace
