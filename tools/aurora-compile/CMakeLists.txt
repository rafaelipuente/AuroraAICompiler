add_executable(aurora-compile
  aurora-compile.cpp
  aurora-compile-backend.cpp
)

# Include MLIR and LLVM include paths
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS})

# Add proper linking flags
llvm_update_compile_flags(aurora-compile)

# Link all required libraries
target_link_libraries(aurora-compile
  PRIVATE
  # Aurora libraries
  AuroraDialect
  AuroraTransforms
  AuroraRuntime
  
  # Core MLIR libraries
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  
  # MLIR dialect libraries
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRVectorDialect
  MLIRSCFDialect
  
  # Optional dialect libraries (if available and needed)
  MLIRTransformDialect
  MLIRArmNeonDialect
  MLIRArmSVEDialect
  MLIRX86VectorDialect
  MLIRComplexDialect
  MLIRTosaDialect
)

# Use LLVM config to find and link required LLVM libraries
llvm_config(aurora-compile support core irreader asmparser bitwriter)

install(TARGETS aurora-compile DESTINATION bin)
