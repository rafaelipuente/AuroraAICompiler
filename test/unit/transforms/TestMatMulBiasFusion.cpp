//===- TestMatMulBiasFusion.cpp - Test for MatMul+Bias fusion pass --------===//
//
// Aurora AI Compiler
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for the MatMul+Bias fusion pass.
//
//===----------------------------------------------------------------------===//

#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "Aurora/Transforms/MatMulBiasFusion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::aurora;

namespace {

//===----------------------------------------------------------------------===//
// MatMulBias Fusion Tests
//===----------------------------------------------------------------------===//

// Helper function to parse MLIR source string
OwningOpRef<ModuleOp> parseMLIRSource(MLIRContext *context, StringRef source) {
  llvm::SourceMgr sourceMgr;
  auto sourceBuf = llvm::MemoryBuffer::getMemBuffer(source);
  sourceMgr.AddNewSourceBuffer(std::move(sourceBuf), llvm::SMLoc());
  
  OwningOpRef<ModuleOp> module = 
      parseSourceFile<ModuleOp>(sourceMgr, context);
  EXPECT_TRUE(module);
  
  return module;
}

// Helper function to get function op by name
func::FuncOp getFuncByName(ModuleOp module, StringRef name) {
  func::FuncOp result;
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp.getSymName() == name)
      result = funcOp;
  });
  return result;
}

// Helper function to count operations of a certain type
template <typename OpType>
unsigned countOps(func::FuncOp funcOp) {
  unsigned count = 0;
  funcOp.walk([&](OpType op) { count++; });
  return count;
}

// Test cases for MatMulBiasFusion pass
class MatMulBiasFusionTest : public ::testing::Test {
protected:
  MatMulBiasFusionTest() {
    // Register the Aurora dialect in the context
    context.getOrLoadDialect<aurora::AuroraDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
  }
  
  MLIRContext context;
};

// Test basic MatMul+Add fusion pattern
TEST_F(MatMulBiasFusionTest, BasicFusion) {
  // Define the input MLIR module
  const char *moduleStr = R"mlir(
    func.func @matmul_bias_fusion(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
      // MatMul operation
      %0 = aurora.matmul(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
      
      // Add bias operation
      %1 = aurora.add(%0, %arg2) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
      
      // Return the result
      return %1 : tensor<4x16xf32>
    }
  )mlir";

  // Parse the module
  auto module = parseMLIRSource(&context, moduleStr);
  auto funcOp = getFuncByName(module.get(), "matmul_bias_fusion");
  ASSERT_TRUE(funcOp);
  
  // Verify the number of each op type before fusion
  EXPECT_EQ(countOps<MatMulOp>(funcOp), 1u);
  EXPECT_EQ(countOps<AddOp>(funcOp), 1u);
  EXPECT_EQ(countOps<MatMulBiasOp>(funcOp), 0u);

  // Apply the MatMulBiasFusion pass
  PassManager pm(&context);
  pm.addPass(createMatMulBiasFusionPass());
  ASSERT_SUCCESS(pm.run(module.get()));
  
  // Get the function again after transformation
  funcOp = getFuncByName(module.get(), "matmul_bias_fusion");
  ASSERT_TRUE(funcOp);
  
  // Verify the number of each op type after fusion
  EXPECT_EQ(countOps<MatMulOp>(funcOp), 0u);
  EXPECT_EQ(countOps<AddOp>(funcOp), 0u);
  EXPECT_EQ(countOps<MatMulBiasOp>(funcOp), 1u);
}

// Test fusion with transpose attributes
TEST_F(MatMulBiasFusionTest, FusionWithTranspose) {
  // Define the input MLIR module with transpose attributes
  const char *moduleStr = R"mlir(
    func.func @matmul_bias_with_transpose(%arg0: tensor<8x4xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
      // MatMul operation with transpose_lhs attribute
      %0 = aurora.matmul(%arg0, %arg1) { transpose_lhs = true } : (tensor<8x4xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
      
      // Add bias operation
      %1 = aurora.add(%0, %arg2) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
      
      // Return the result
      return %1 : tensor<4x16xf32>
    }
  )mlir";

  // Parse the module
  auto module = parseMLIRSource(&context, moduleStr);
  auto funcOp = getFuncByName(module.get(), "matmul_bias_with_transpose");
  ASSERT_TRUE(funcOp);
  
  // Apply the MatMulBiasFusion pass
  PassManager pm(&context);
  pm.addPass(createMatMulBiasFusionPass());
  ASSERT_SUCCESS(pm.run(module.get()));
  
  // Verify that we now have one MatMulBiasOp and the MatMulOp and AddOp are gone
  funcOp = getFuncByName(module.get(), "matmul_bias_with_transpose");
  EXPECT_EQ(countOps<MatMulOp>(funcOp), 0u);
  EXPECT_EQ(countOps<AddOp>(funcOp), 0u);
  EXPECT_EQ(countOps<MatMulBiasOp>(funcOp), 1u);
  
  // Verify that the transpose attribute was preserved
  bool foundTranspose = false;
  funcOp.walk([&](MatMulBiasOp op) {
    foundTranspose = op.getTransposeLhs();
  });
  EXPECT_TRUE(foundTranspose);
}

// Test no fusion when matmul has multiple uses
TEST_F(MatMulBiasFusionTest, NoFusionWithMultipleUses) {
  // Define the input MLIR module with multiple uses of matmul
  const char *moduleStr = R"mlir(
    func.func @no_fusion_multiple_uses(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
      // MatMul operation
      %0 = aurora.matmul(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
      
      // Add bias operation
      %1 = aurora.add(%0, %arg2) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
      
      // Return both matmul result and bias add result
      return %0, %1 : tensor<4x16xf32>, tensor<4x16xf32>
    }
  )mlir";

  // Parse the module
  auto module = parseMLIRSource(&context, moduleStr);
  auto funcOp = getFuncByName(module.get(), "no_fusion_multiple_uses");
  ASSERT_TRUE(funcOp);
  
  // Apply the MatMulBiasFusion pass
  PassManager pm(&context);
  pm.addPass(createMatMulBiasFusionPass());
  ASSERT_SUCCESS(pm.run(module.get()));
  
  // Verify that the ops were not fused because matmul has multiple uses
  funcOp = getFuncByName(module.get(), "no_fusion_multiple_uses");
  EXPECT_EQ(countOps<MatMulOp>(funcOp), 1u);
  EXPECT_EQ(countOps<AddOp>(funcOp), 1u);
  EXPECT_EQ(countOps<MatMulBiasOp>(funcOp), 0u);
}

} // end namespace
