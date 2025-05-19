//===- AuroraDialect.td - Aurora dialect definition ---------*- tablegen -*-===//
//
// This file defines the Aurora dialect in the MLIR system.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_DIALECT_TD
#define AURORA_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Aurora dialect definition
//===----------------------------------------------------------------------===//

def Aurora_Dialect : Dialect {
  let name = "aurora";
  let cppNamespace = "::mlir::aurora";
  
  let summary = "A dialect for Aurora AI operations";
  let description = [{
    The Aurora dialect is designed for high-performance AI operations.
  }];

  // Use the updated fold API to remove deprecation warnings
  let useFoldAPI = kEmitFoldAdaptorFolder;
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Let TableGen generate the dialect declarations

#endif // AURORA_DIALECT_TD
