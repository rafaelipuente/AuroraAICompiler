//===- aurora-opt.cpp - Aurora MLIR pass driver ---------------------------===//
//
// Part of the Aurora Compiler Project
//
// Thin wrapper around MlirOptMain. Registers the Aurora dialect, all Aurora
// passes, and all standard MLIR dialects and passes so the full downstream
// lowering pipeline is available via --pass-pipeline or individual flags:
//
//   # Aurora transformation passes
//   aurora-opt input.mlir --aurora-matmul-bias-fusion
//   aurora-opt input.mlir --convert-aurora-to-linalg
//
//   # Full pipeline to LLVM dialect
//   aurora-opt input.mlir \
//     --convert-aurora-to-linalg \
//     --one-shot-bufferize="bufferize-function-boundaries=true \
//                           allow-return-allocs-in-loops=true" \
//     --convert-linalg-to-loops \
//     --convert-scf-to-cf \
//     --convert-index-to-llvm \
//     --convert-arith-to-llvm \
//     --convert-cf-to-llvm \
//     --convert-memref-to-llvm \
//     --convert-func-to-llvm \
//     --reconcile-unrealized-casts
//
//===----------------------------------------------------------------------===//

#include "Aurora/Conversion/Passes.h"
#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all standard MLIR dialects and passes so the full downstream
  // lowering pipeline (bufferization, Linalg->loops, SCF/CF/arith/memref->LLVM)
  // is available without manual enumeration. aurora-opt is intended as a
  // development tool: the broad registration is appropriate here.
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // Aurora dialect and passes are not part of upstream MLIR.
  registry.insert<mlir::aurora::AuroraDialect>();
  mlir::aurora::registerAuroraPasses();
  mlir::aurora::registerAuroraConversionPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Aurora MLIR optimizer\n", registry));
}
