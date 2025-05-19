# First, declare the Aurora dialect library without any dependencies on itself
add_mlir_dialect_library(AuroraDialect
  AuroraDialect.cpp
  AuroraOps.cpp
  AuroraDialectRegistration.cpp

  DEPENDS
  AuroraOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

target_include_directories(AuroraDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_BINARY_DIR}/include
)
