configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

set(AURORA_TEST_DEPENDS
  aurora-opt
)

add_custom_target(check-aurora
  COMMAND ${Python_EXECUTABLE} -m lit
    ${CMAKE_CURRENT_SOURCE_DIR}
    --config-prefix lit
    -sv
    --param aurora_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  DEPENDS ${AURORA_TEST_DEPENDS}
  COMMENT "Running Aurora lit tests"
  USES_TERMINAL
)

# Stub subdirectories (empty; preserved for future use).
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(end_to_end)
