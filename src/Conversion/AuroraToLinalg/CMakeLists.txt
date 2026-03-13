add_mlir_library(AuroraConversion
  AuroraToLinalg.cpp

  DEPENDS
  AuroraOpsIncGen
  AuroraConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
  AuroraDialect
)

target_include_directories(AuroraConversion PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_BINARY_DIR}/include
)
