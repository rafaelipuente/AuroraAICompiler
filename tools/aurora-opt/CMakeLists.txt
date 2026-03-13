get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(aurora-opt aurora-opt.cpp)

llvm_update_compile_flags(aurora-opt)

target_link_libraries(aurora-opt
  PRIVATE
  # Aurora libraries
  AuroraDialect
  AuroraTransforms
  AuroraConversion

  # All MLIR dialect and conversion libraries (for registerAllDialects /
  # registerAllPasses).  These properties capture every library registered via
  # add_mlir_dialect_library() and add_mlir_conversion_library().
  ${dialect_libs}
  ${conversion_libs}

  # Core MLIR infrastructure
  MLIROptLib
  MLIRPass
  MLIRIR

  # Explicit additions for transform/lowering libraries that may not be
  # captured in the MLIR_CONVERSION_LIBS property depending on the MLIR build:
  MLIRBufferizationTransforms   # one-shot-bufferize
  MLIRLinalgTransforms          # convert-linalg-to-loops
  MLIRSCFToControlFlow          # convert-scf-to-cf
  MLIRArithToLLVM               # convert-arith-to-llvm
  MLIRControlFlowToLLVM         # convert-cf-to-llvm
  MLIRMemRefToLLVM              # convert-memref-to-llvm
  MLIRFuncToLLVM                # convert-func-to-llvm
)

mlir_check_all_link_libraries(aurora-opt)

install(TARGETS aurora-opt DESTINATION bin)
