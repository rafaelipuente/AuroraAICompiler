# Generate dialect files from AuroraDialect.td
set(LLVM_TARGET_DEFINITIONS AuroraDialect.td)
mlir_tablegen(AuroraDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AuroraDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRAuroraDialectIncGen)

# Generate operation files from AuroraOps.td
set(LLVM_TARGET_DEFINITIONS AuroraOps.td)
mlir_tablegen(AuroraOps.h.inc -gen-op-decls)
mlir_tablegen(AuroraOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AuroraOpsIncGen)
