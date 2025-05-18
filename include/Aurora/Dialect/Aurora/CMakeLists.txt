set(LLVM_TARGET_DEFINITIONS AuroraOps.td)
mlir_tablegen(AuroraOps.h.inc -gen-op-decls)
mlir_tablegen(AuroraOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AuroraOpsIncGen)

add_mlir_dialect_library(AuroraDialect
  AuroraDialect.cpp
  AuroraOps.cpp

  DEPENDS
  AuroraOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

target_include_directories(AuroraDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
