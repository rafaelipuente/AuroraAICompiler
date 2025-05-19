# Aurora benchmarks
# Create custom targets for Python scripts

# Copy the benchmark Python script to the build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_resnet.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Create a custom target for the benchmark
add_custom_target(benchmark_resnet ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/benchmark_resnet.py
  COMMENT "Preparing ResNet benchmark script"
)

# Main benchmarks target
add_custom_target(aurora-benchmarks ALL
  DEPENDS benchmark_resnet
  COMMENT "Aurora benchmarks ready"
)
