add_executable(aurora-compile
  aurora-compile.cpp
)

target_link_libraries(aurora-compile
  PRIVATE
  AuroraDialect
  AuroraCompiler
  AuroraTransforms
  AuroraRuntime
  MLIRDialects
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
)

install(TARGETS aurora-compile DESTINATION bin)
