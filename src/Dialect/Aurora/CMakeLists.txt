add_mlir_dialect_library(AuroraDialectImpl
  AuroraDialect.cpp
  AuroraOps.cpp

  DEPENDS
  AuroraOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  AuroraDialect
)

target_include_directories(AuroraDialectImpl PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_BINARY_DIR}/include
)
