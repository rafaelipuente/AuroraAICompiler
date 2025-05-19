add_executable(aurora-jit
  jit_executor.cpp
)

llvm_update_compile_flags(aurora-jit)
target_link_libraries(aurora-jit
  PRIVATE
  MLIRAuroraDialect
  MLIRAuroraToLLVM
  MLIRExecutionEngine
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
)

mlir_check_all_link_libraries(aurora-jit)
