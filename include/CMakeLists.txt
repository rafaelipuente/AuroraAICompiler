add_subdirectory(Aurora)

# Add header-only library targets
add_library(AuroraHeaders INTERFACE)
target_include_directories(AuroraHeaders INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Configure header file installation
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Aurora
  DESTINATION include
  COMPONENT devel
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.def"
  PATTERN "*.inc"
)
