add_library(AuroraRuntime
  AuroraRuntime.cpp
)

target_include_directories(AuroraRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(AuroraRuntime
  PRIVATE
  MLIRIR
  MLIRSupport
)

install(TARGETS AuroraRuntime
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
