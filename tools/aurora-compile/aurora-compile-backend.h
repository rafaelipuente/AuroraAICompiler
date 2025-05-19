#ifndef AURORA_COMPILE_BACKEND_H
#define AURORA_COMPILE_BACKEND_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

class AuroraLogger;

// Target Backend Framework
// This abstract class defines the interface for target-specific code generation
class TargetBackend {
public:
  virtual ~TargetBackend() = default;
  
  // Initialize the backend - return false if initialization fails
  virtual bool initialize() { return true; }
  
  // Generate code for the target from the provided MLIR module
  virtual bool generateCode(mlir::ModuleOp module, llvm::raw_ostream &os, 
                          AuroraLogger &logger, bool verbose) = 0;
                          
  // Get the name of this backend
  virtual std::string getName() const = 0;
  
  // Check if this backend is implemented/available
  virtual bool isImplemented() const = 0;
  
  // Print backend-specific information
  virtual void printInfo(AuroraLogger &logger) const;
};

// CPU Backend Implementation
class CPUBackend : public TargetBackend {
public:
  CPUBackend() = default;
  
  bool generateCode(mlir::ModuleOp module, llvm::raw_ostream &os, 
                   AuroraLogger &logger, bool verbose) override;
  
  std::string getName() const override { return "cpu"; }
  
  bool isImplemented() const override { return true; }
};

// GPU Backend Implementation
class GPUBackend : public TargetBackend {
public:
  GPUBackend() = default;
  
  bool generateCode(mlir::ModuleOp module, llvm::raw_ostream &os, 
                   AuroraLogger &logger, bool verbose) override;
  
  std::string getName() const override { return "gpu"; }
  
  bool isImplemented() const override { return false; }
  
  void printInfo(AuroraLogger &logger) const override;
};

// Ampere AICore Backend Implementation
class AmpereAICoreBackend : public TargetBackend {
public:
  AmpereAICoreBackend() = default;
  
  bool generateCode(mlir::ModuleOp module, llvm::raw_ostream &os, 
                   AuroraLogger &logger, bool verbose) override;
  
  std::string getName() const override { return "ampere-aicore"; }
  
  bool isImplemented() const override { return false; }
  
  void printInfo(AuroraLogger &logger) const override;
};

// Target Backend Factory
// Creates the appropriate backend based on the target name
std::unique_ptr<TargetBackend> createTargetBackend(const std::string &targetName, AuroraLogger &logger);

#endif // AURORA_COMPILE_BACKEND_H
