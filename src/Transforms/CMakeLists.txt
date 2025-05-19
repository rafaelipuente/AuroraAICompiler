add_mlir_library(AuroraTransforms
  Fusion.cpp
  MatMulBiasFusion.cpp

  DEPENDS
  AuroraOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  AuroraDialect
)

target_include_directories(AuroraTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_BINARY_DIR}/include
)
