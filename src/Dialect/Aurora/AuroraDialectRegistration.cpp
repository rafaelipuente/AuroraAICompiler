#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"

// Here we include the generated implementation for the operation definitions
#include "Aurora/Dialect/Aurora/AuroraOps.cpp.inc"

// This registers the dialect with MLIR
namespace mlir {
namespace aurora {

// Register all the operation TypeIDs in this namespace
namespace {

// Define type IDs for all operations in the Aurora dialect
#define GET_TYPEDEF_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.h.inc"

} // anonymous namespace

} // namespace aurora
} // namespace mlir

// Define operation interfaces so they can be used in the operations 
#define GET_OP_INTERFACE_CLASSES
#include "Aurora/Dialect/Aurora/AuroraOps.h.inc"
