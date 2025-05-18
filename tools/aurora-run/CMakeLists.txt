add_executable(aurora-run
  aurora-run.cpp
)

target_link_libraries(aurora-run
  PRIVATE
  AuroraRuntime
  MLIRIR
  MLIRSupport
)

install(TARGETS aurora-run DESTINATION bin)
