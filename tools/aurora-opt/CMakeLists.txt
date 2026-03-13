get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(aurora-opt aurora-opt.cpp)

llvm_update_compile_flags(aurora-opt)

target_link_libraries(aurora-opt
  PRIVATE
  AuroraDialect
  AuroraTransforms
  AuroraConversion

  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRPass
  MLIRIR
)

mlir_check_all_link_libraries(aurora-opt)

install(TARGETS aurora-opt DESTINATION bin)
