add_mlir_library(AuroraTransforms
  Fusion.cpp
  MatMulBiasFusion.cpp

  DEPENDS
  AuroraOpsIncGen
  AuroraTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  AuroraDialect
)

target_include_directories(AuroraTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_BINARY_DIR}/include
)
