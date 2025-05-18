add_executable(aurora-benchmark
  aurora-benchmark.cpp
)

target_link_libraries(aurora-benchmark
  PRIVATE
  AuroraRuntime
  MLIRIR
  MLIRSupport
)

install(TARGETS aurora-benchmark DESTINATION bin)
