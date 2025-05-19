# Python bindings and modules for Aurora AI Compiler
# For Python modules, we simply copy them to the build directory

set(AURORA_PYTHON_FILES
  aurora/__init__.py
  aurora/compiler.py
  aurora/model_import/__init__.py
  aurora/model_import/onnx_importer.py
  aurora/model_import/onnx_loader.py
  aurora/model_import/onnx_to_mlir.py
  aurora/model_import/pytorch_importer.py
  aurora/runtime/__init__.py
  aurora/runtime/model.py
  setup.py
)

# Copy all Python files to the build directory
foreach(PYTHON_FILE ${AURORA_PYTHON_FILES})
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_FILE}
    ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_FILE}
    COPYONLY
  )
endforeach()

# Create a custom target that doesn't actually build anything
add_custom_target(aurora-python ALL
  COMMENT "Copying Python modules to build directory"
)

# Add install rules for Python modules
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/aurora
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python
  FILES_MATCHING PATTERN "*.py"
)
