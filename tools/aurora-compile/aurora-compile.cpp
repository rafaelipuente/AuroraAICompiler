#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassRegistry.h"
// Include specific dialect headers instead of InitAllDialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/Chrono.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <map>
#include <set>
#include <string>

// Aurora headers
#include "Aurora/Dialect/Aurora/AuroraDialect.h"
#include "Aurora/Dialect/Aurora/AuroraOps.h"
#include "Aurora/Transforms/Fusion.h"
#include "Aurora/Transforms/MatMulBiasFusion.h"

// Target backend framework
#include "aurora-compile-backend.h"

using namespace mlir;
using namespace mlir::aurora;
using namespace llvm;

static cl::opt<std::string> inputFilename(
    cl::Positional, cl::desc("<input model>"), cl::Required);

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename"), cl::value_desc("filename"), cl::Required);

static cl::opt<bool> emitMLIR(
    "emit-mlir", cl::desc("Output MLIR intermediate representation"), cl::init(false));

static cl::opt<std::string> inputFormat(
    "input-format", cl::desc("Input format (onnx, pytorch)"), cl::init("onnx"));

static cl::opt<std::string> targetDevice(
    "target", cl::desc("Target device (cpu, gpu, ampere-aicore)"), cl::init("cpu"));

static cl::opt<bool> enableOpt(
    "opt", cl::desc("Enable optimizations"), cl::init(true));

static cl::opt<int> optLevel(
    "O", cl::desc("Optimization level"), cl::init(3));

static cl::opt<bool> fuseMatMulBias(
    "fuse-matmul-bias", cl::desc("Enable MatMul+Bias fusion"), cl::init(false));

static cl::opt<bool> debugIR(
    "debug-ir", cl::desc("Print IR before and after passes"), cl::init(false));

static cl::opt<bool> verbose(
    "verbose", cl::desc("Enable verbose output with diagnostic timing information"), cl::init(false));

static cl::opt<bool> colorOutput(
    "aurora-color", cl::desc("Enable colored terminal output"), cl::init(true));

static cl::opt<bool> dumpMermaid(
    "dump-mermaid", cl::desc("Generate a Mermaid diagram of the operation graph"), cl::init(false));

// Terminal color codes
namespace TermColor {
  // Basic colors
  const char* const Reset   = "\033[0m";
  const char* const Red     = "\033[31m";
  const char* const Green   = "\033[32m";
  const char* const Yellow  = "\033[33m";
  const char* const Blue    = "\033[34m";
  const char* const Magenta = "\033[35m";
  const char* const Cyan    = "\033[36m";
  const char* const White   = "\033[37m";
  
  // Text styles
  const char* const Bold      = "\033[1m";
  const char* const Underline = "\033[4m";
  
  // Bright/high-intensity colors (more visible)
  const char* const BrightRed     = "\033[91m";
  const char* const BrightGreen   = "\033[92m";
  const char* const BrightYellow  = "\033[93m";
  const char* const BrightBlue    = "\033[94m";
  const char* const BrightMagenta = "\033[95m";
  const char* const BrightCyan    = "\033[96m";
  const char* const BrightWhite   = "\033[97m";
}

// Logging class for structured, color-coded output
class AuroraLogger {
public:
  enum class Level { DEBUG, INFO, SUCCESS, WARNING, ERROR };
  
  AuroraLogger(bool useColor = true) : useColor_(useColor) {}
  
  void setColorOutput(bool useColor) { useColor_ = useColor; }
  
  llvm::raw_ostream& log(Level level, const std::string& component = "") {
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;
    
    // Format timestamp
    std::stringstream ss;
    ss << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0') 
       << std::setw(3) << ms.count();
    std::string timestamp = ss.str();
    
    // Set color based on level
    if (useColor_) {
      switch (level) {
        case Level::DEBUG:   llvm::outs() << TermColor::BrightBlue; break;
        case Level::INFO:    llvm::outs() << TermColor::BrightWhite; break;
        case Level::SUCCESS: llvm::outs() << TermColor::BrightGreen; break;
        case Level::WARNING: llvm::outs() << TermColor::BrightYellow; break;
        case Level::ERROR:   llvm::outs() << TermColor::BrightRed; break;
      }
    }
    
    // Output log level prefix
    llvm::outs() << "[" << timestamp << "] ";
    if (useColor_) llvm::outs() << TermColor::Bold;
    
    switch (level) {
      case Level::DEBUG:   llvm::outs() << "DEBUG"; break;
      case Level::INFO:    llvm::outs() << "INFO "; break;
      case Level::SUCCESS: llvm::outs() << "DONE "; break;
      case Level::WARNING: llvm::outs() << "WARN "; break;
      case Level::ERROR:   llvm::outs() << "ERROR"; break;
    }
    
    if (useColor_) llvm::outs() << TermColor::Reset;
    
    // Add component name if provided
    if (!component.empty()) {
      if (useColor_) llvm::outs() << TermColor::Cyan;
      llvm::outs() << " [" << component << "]";
      if (useColor_) llvm::outs() << TermColor::Reset;
    }
    
    llvm::outs() << ": ";
    return llvm::outs();
  }
  
  // Timer utility class for performance measurements
  class ScopedTimer {
  private:
    AuroraLogger& logger_;
    std::string operation_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool verbose_;
    
  public:
    ScopedTimer(AuroraLogger& logger, const std::string& operation, 
                bool verbose = false)
      : logger_(logger), operation_(operation), verbose_(verbose) {
      start_ = std::chrono::high_resolution_clock::now();
      if (verbose_) {
        logger_.log(Level::DEBUG, "TIMER") << "Starting " << operation_ << "\n";
      }
    }
    
    ~ScopedTimer() {
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
      if (verbose_) {
        logger_.log(Level::DEBUG, "TIMER") 
          << "Completed " << operation_ << " in " 
          << llvm::format("%.3f", duration / 1000.0) << " seconds\n";
      }
    }
    
    // Get the start time of this timer
    std::chrono::time_point<std::chrono::high_resolution_clock> getStartTime() const {
      return start_;
    }
  };
  
private:
  bool useColor_;
};

// Global logger instance
static AuroraLogger logger;

// Stats collector for operation counts
struct CompilerStats {
  int totalOperations = 0;
  int auroraOperations = 0;
  int fusions = 0;
  int matmulBiasFusions = 0;
  
  void countOperations(Operation* op) {
    totalOperations++;
    if (op->getDialect() && 
        op->getDialect()->getNamespace() == "aurora") {
      auroraOperations++;
    }
  }
  
  void print(raw_ostream& os) {
    os << "Total operations: " << totalOperations << "\n";
    os << "Aurora operations: " << auroraOperations << "\n";
    if (fusions > 0) {
      os << "Operations fused: " << fusions << "\n";
    }
    if (matmulBiasFusions > 0) {
      os << "MatMul+Bias fusions: " << matmulBiasFusions << "\n";
    }
  }
};

// Mermaid diagram generator for visualizing operation graphs
class MermaidDiagramGenerator {
public:
  MermaidDiagramGenerator(mlir::ModuleOp module, AuroraLogger& logger)
    : module_(module), logger_(logger) {}

  // Generate a Mermaid flowchart diagram
  std::string generateDiagram() {
    std::string diagram = "flowchart TD\n";
    std::map<mlir::Operation*, std::string> opIds;
    std::set<std::pair<std::string, std::string>> edges; // Use the standard allocator and comparator

    // First pass: collect all operations and assign IDs
    int opCount = 0;
    module_.walk([&](mlir::Operation* op) {
      // Skip the module operation itself
      if (op == module_.getOperation()) return;

      // Skip terminator operations
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) return;

      // Generate an ID for this operation
      std::string opId = "op" + std::to_string(opCount++);
      opIds[op] = opId;

      // Create node representation
      diagram += "    " + opId + "[\"";
      
      // If this is a named operation, include the name
      if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
        diagram += funcOp.getSymName().str() + "<br/>";
      }
      
      // Display operation type
      diagram += op->getName().getStringRef().str();
      
      // Include tensor shapes if available
      for (auto result : op->getResults()) {
        if (auto type = result.getType().dyn_cast<mlir::TensorType>()) {
          std::string typeStr;
          llvm::raw_string_ostream os(typeStr);
          type.print(os);
          diagram += "<br/>" + typeStr;
          break; // Only show one type to keep diagram clean
        }
      }
      
      diagram += "\"]\n";
      
      // Color-code nodes by dialect
      if (op->getDialect()) {
        std::string dialectName = op->getDialect()->getNamespace().str();
        if (dialectName == "aurora") {
          diagram += "    class " + opId + " auroraOp\n";
        } else if (dialectName == "func") {
          diagram += "    class " + opId + " funcOp\n";
        } else if (dialectName == "arith") {
          diagram += "    class " + opId + " arithOp\n";
        }
      }
    });

    // Second pass: collect all operation edges
    module_.walk([&](mlir::Operation* op) {
      // Skip operations we haven't assigned IDs to
      if (opIds.find(op) == opIds.end()) return;
      
      // For each operand, find its defining op and create an edge
      for (auto operand : op->getOperands()) {
        if (operand.getDefiningOp()) {
          mlir::Operation* defOp = operand.getDefiningOp();
          
          // Skip if the defining op wasn't assigned an ID (e.g., it's a module op)
          if (opIds.find(defOp) == opIds.end()) continue;
          
          // Add the edge: source -> target
          edges.insert({opIds[defOp], opIds[op]});
        }
      }
    });

    // Add all edges to the diagram
    for (const auto& edge : edges) {
      diagram += "    " + edge.first + " --> " + edge.second + "\n";
    }

    // Add style definitions
    diagram += "    classDef auroraOp fill:#c6f5d5,stroke:#22863a,stroke-width:2px\n";
    diagram += "    classDef funcOp fill:#f1e05a,stroke:#b08800,stroke-width:2px\n";
    diagram += "    classDef arithOp fill:#e4e4e4,stroke:#666,stroke-width:1px\n";

    return diagram;
  }

  // Save the diagram to a file
  bool saveDiagram(const std::string& filename) {
    std::string diagram = generateDiagram();
    
    std::error_code ec;
    llvm::raw_fd_ostream outfile(filename, ec);
    if (ec) {
      logger_.log(AuroraLogger::Level::ERROR, "MERMAID") 
        << "Failed to open file for Mermaid diagram: " << ec.message() << "\n";
      return false;
    }
    
    outfile << diagram;
    outfile.close();
    
    logger_.log(AuroraLogger::Level::SUCCESS, "MERMAID") 
      << "Generated Mermaid diagram at " << filename << "\n";
    return true;
  }

private:
  mlir::ModuleOp module_;
  AuroraLogger& logger_;
};

// MLIR pass instrumentation to collect statistics and timing information
class AuroraPassInstrumentation : public PassInstrumentation {
private:
  AuroraLogger& logger_;
  CompilerStats& stats_;
  bool verbose_;
  
public:
  AuroraPassInstrumentation(AuroraLogger& logger, CompilerStats& stats, bool verbose)
    : logger_(logger), stats_(stats), verbose_(verbose) {}
  
  void runBeforePass(Pass* pass, Operation* op) override {
    if (verbose_) {
      StringRef passName = pass->getName();
      logger_.log(AuroraLogger::Level::DEBUG, "PASS")
        << "Starting pass: " << passName << "\n";
    }
  }
  
  void runAfterPass(Pass* pass, Operation* op) override {
    if (verbose_) {
      StringRef passName = pass->getName();
      logger_.log(AuroraLogger::Level::DEBUG, "PASS")
        << "Completed pass: " << passName << "\n";
      
      // Count operations after the pass
      CompilerStats passStats;
      op->walk([&](Operation* op) {
        passStats.countOperations(op);
      });
      
      logger_.log(AuroraLogger::Level::DEBUG, "STATS")
        << "After " << passName << ": " 
        << passStats.totalOperations << " operations (" 
        << passStats.auroraOperations << " Aurora ops)\n";
    }
  }
  
  void runAfterPassFailed(Pass* pass, Operation* op) override {
    StringRef passName = pass->getName();
    logger_.log(AuroraLogger::Level::ERROR, "PASS")
      << "Failed during pass: " << passName << "\n";
  }
};

int main(int argc, char **argv) {
  // Initialize LLVM
  InitLLVM y(argc, argv);
  
  // Register command line options
  cl::ParseCommandLineOptions(argc, argv, "Aurora AI Compiler\n");
  
  // Configure logger
  logger.setColorOutput(colorOutput);
  
  // Initialize compiler stats
  CompilerStats stats;
  
  // Main timer for the entire compilation process
  AuroraLogger::ScopedTimer mainTimer(logger, "Total compilation", verbose);
  
  logger.log(AuroraLogger::Level::INFO, "INIT")
    << "Aurora AI Compiler v0.1.0" << (verbose ? " (verbose mode)" : "") << "\n";
  
  // Setup MLIR context with dialect registry
  AuroraLogger::ScopedTimer registryTimer(logger, "Dialect registration", verbose);
  DialectRegistry registry;
  
  // Register the Aurora dialect
  logger.log(AuroraLogger::Level::INFO, "DIALECT")
    << "Registering Aurora dialect" << "\n";
  registry.insert<aurora::AuroraDialect>();
  
  // Register only the dialects we need and have available
  logger.log(AuroraLogger::Level::INFO, "DIALECT")
    << "Registering standard MLIR dialects" << "\n";
  registry.insert<
    mlir::func::FuncDialect,
    mlir::arith::ArithDialect,
    mlir::memref::MemRefDialect,
    mlir::tensor::TensorDialect,
    mlir::linalg::LinalgDialect,
    mlir::vector::VectorDialect,
    mlir::scf::SCFDialect
  >();
  
  // Note: We removed registerAllDialects to avoid linking errors with dialects
  // that aren't available in the current build configuration
  registryTimer.~ScopedTimer(); // End the registry timer
  
  // Create context with the registry
  AuroraLogger::ScopedTimer contextTimer(logger, "Context creation", verbose);
  logger.log(AuroraLogger::Level::INFO, "CONTEXT")
    << "Creating MLIR context" << "\n";
  MLIRContext context(registry);
  
  // Note: Pass instrumentation would be added here in newer MLIR versions
  // For now, we'll rely on our own timing and diagnostics
  if (verbose) {
    logger.log(AuroraLogger::Level::DEBUG, "CONTEXT")
      << "Using custom diagnostics for pass execution" << "\n";
  }
  contextTimer.~ScopedTimer(); // End the context timer
  
  // Set up the input file
  AuroraLogger::ScopedTimer fileTimer(logger, "File I/O", verbose);
  std::string errorMessage;
  logger.log(AuroraLogger::Level::INFO, "FILE")
    << "Opening input file: " << inputFilename << "\n";
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    logger.log(AuroraLogger::Level::ERROR, "FILE")
      << errorMessage << "\n";
    return 1;
  }
  fileTimer.~ScopedTimer(); // End the file timer
  
  // Parse input based on format
  logger.log(AuroraLogger::Level::INFO, "PARSER")
    << "Processing " << inputFilename << " as " << inputFormat << " format\n";
  
  // Create a module to hold the IR
  OwningOpRef<ModuleOp> module;
  
  // Detect file extension to determine how to process it
  StringRef extension = llvm::sys::path::extension(inputFilename);
  
  if (extension == ".mlir") {
    // Handle MLIR files directly
    AuroraLogger::ScopedTimer parsingTimer(logger, "MLIR parsing", verbose);
    logger.log(AuroraLogger::Level::INFO, "PARSER")
      << "Parsing MLIR file..." << "\n";
    
    // Set up the source manager for the parser
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    
    // Parse the input file
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
      logger.log(AuroraLogger::Level::ERROR, "PARSER")
        << "Error parsing MLIR file" << "\n";
      return 1;
    }
    
    // Verify the module to ensure it's valid
    logger.log(AuroraLogger::Level::INFO, "VERIFIER")
      << "Verifying module..." << "\n";
    if (failed(verify(*module))) {
      logger.log(AuroraLogger::Level::ERROR, "VERIFIER")
        << "Error verifying MLIR module" << "\n";
      return 1;
    }
    
    // Count initial operations if in verbose mode
    if (verbose) {
      module->walk([&](Operation* op) {
        stats.countOperations(op);
      });
      
      logger.log(AuroraLogger::Level::DEBUG, "STATS")
        << "Initial module contains " << stats.totalOperations 
        << " operations (" << stats.auroraOperations << " Aurora ops)" << "\n";
    }
    
    logger.log(AuroraLogger::Level::SUCCESS, "PARSER")
      << "Successfully parsed MLIR with Aurora operations" << "\n";
  } else if (inputFormat == "onnx") {
    AuroraLogger::ScopedTimer importTimer(logger, "ONNX import", verbose);
    logger.log(AuroraLogger::Level::INFO, "IMPORTER")
      << "Importing ONNX model..." << "\n";
    // This would call into the ONNX importer
    // For now, just create an empty module
    module = ModuleOp::create(UnknownLoc::get(&context));
    logger.log(AuroraLogger::Level::WARNING, "IMPORTER")
      << "ONNX import not fully implemented" << "\n";
  } else if (inputFormat == "pytorch") {
    AuroraLogger::ScopedTimer importTimer(logger, "PyTorch import", verbose);
    logger.log(AuroraLogger::Level::INFO, "IMPORTER")
      << "Importing PyTorch model..." << "\n";
    // This would call into the PyTorch importer
    // For now, just create an empty module
    module = ModuleOp::create(UnknownLoc::get(&context));
    logger.log(AuroraLogger::Level::WARNING, "IMPORTER")
      << "PyTorch import not fully implemented" << "\n";
  } else {
    logger.log(AuroraLogger::Level::ERROR, "IMPORTER")
      << "Unsupported input format: " << inputFormat << "\n";
    return 1;
  }
  
  // Apply optimizations if enabled
  if (enableOpt) {
    AuroraLogger::ScopedTimer optTimer(logger, "Optimization pipeline", verbose);
    logger.log(AuroraLogger::Level::INFO, "OPT")
      << "Running optimizations at level O" << optLevel << "..." << "\n";
    
    PassManager pm(&context);
    
    // Enable pass timing in verbose mode
    if (verbose) {
      pm.enableTiming();
    }
    
    // Print the IR before applying passes if debug mode is enabled
    if (debugIR) {
      logger.log(AuroraLogger::Level::DEBUG, "IR")
        << "\n==== IR BEFORE OPTIMIZATION ====\n";
      module->print(llvm::outs());
      logger.log(AuroraLogger::Level::DEBUG, "IR")
        << "\n==== END IR BEFORE OPTIMIZATION ====\n\n";
      
      // Dump the IR to a file before optimization
      std::string beforeOptFilename = outputFilename + ".before-opt.mlir";
      std::error_code EC;
      auto beforeOptFile = std::make_unique<llvm::ToolOutputFile>(
          beforeOptFilename, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        logger.log(AuroraLogger::Level::ERROR, "FILE")
          << "Failed to open debug IR output file: " << beforeOptFilename
          << " - " << EC.message() << "\n";
      } else {
        module->print(beforeOptFile->os());
        beforeOptFile->keep();
        logger.log(AuroraLogger::Level::INFO, "DEBUG")
          << "Dumped IR before optimization to " << beforeOptFilename << "\n";
      }
    }
    
    // Add Aurora-specific optimizations
    if (optLevel >= 1) {
      // Level 1: Basic optimizations
      logger.log(AuroraLogger::Level::INFO, "PASS")
        << "Adding Aurora Fusion pass" << "\n";
      pm.addPass(createFusionPass());
    }
    
    // Add MatMulBias fusion if explicitly requested
    if (fuseMatMulBias) {
      logger.log(AuroraLogger::Level::INFO, "PASS")
        << "Adding MatMul+Bias fusion pass" << "\n";
      pm.addPass(createMatMulBiasFusionPass());
      
      // Always run a pass to clean up after fusion
      logger.log(AuroraLogger::Level::INFO, "PASS")
        << "Adding cleanup pass after fusion" << "\n";
      pm.addPass(mlir::createInlinerPass());
    }
    
    if (optLevel >= 2) {
      // Level 2: More advanced optimizations
      logger.log(AuroraLogger::Level::INFO, "OPT")
        << "Adding level 2 optimization passes" << "\n";
      // Would add more passes in a real implementation
    }
    
    if (optLevel >= 3) {
      // Level 3: Aggressive optimizations
      logger.log(AuroraLogger::Level::INFO, "OPT")
        << "Adding level 3 optimization passes" << "\n";
      // Would add more passes in a real implementation
    }
    
    // Run the optimization pipeline
    logger.log(AuroraLogger::Level::INFO, "OPT")
      << "Running optimization passes..." << "\n";
      
    // Create a timer for just the execution phase
    AuroraLogger::ScopedTimer passExecutionTimer(logger, "Pass execution", verbose);
    if (failed(pm.run(*module))) {
      logger.log(AuroraLogger::Level::ERROR, "OPT")
        << "Optimization failed" << "\n";
      return 1;
    }
    
    // Print simple timing information if verbose
    if (verbose) {
      logger.log(AuroraLogger::Level::DEBUG, "TIMING")
        << "Pass execution completed" << "\n";
      // Note: In newer versions of MLIR, we could use pm.printAsJSON() for detailed timing
    }
    
    // Count operations after optimization if in verbose mode
    if (verbose) {
      CompilerStats afterStats;
      module->walk([&](Operation* op) {
        afterStats.countOperations(op);
      });
      
      logger.log(AuroraLogger::Level::DEBUG, "STATS")
        << "After optimization: " << afterStats.totalOperations 
        << " operations (" << afterStats.auroraOperations << " Aurora ops)" << "\n";
        
      // Show difference in operation count
      int opDiff = stats.totalOperations - afterStats.totalOperations;
      if (opDiff != 0) {
        logger.log(AuroraLogger::Level::DEBUG, "STATS")
          << "Operation count " << (opDiff > 0 ? "reduced by " : "increased by ") 
          << std::abs(opDiff) << " operations" << "\n";
      }
    }
    
    // Generate a Mermaid diagram of the operation graph if requested
    if (dumpMermaid) {
      logger.log(AuroraLogger::Level::INFO, "MERMAID")
        << "Generating Mermaid diagram for operation graph..." << "\n";
      
      // Create the diagram generator
      MermaidDiagramGenerator diagramGenerator(*module, logger);
      
      // Generate filename based on input file
      std::string baseFilename = llvm::sys::path::stem(inputFilename).str();
      std::string diagramFilename = baseFilename + ".mermaid.mmd";
      
      // Save the diagram
      if (diagramGenerator.saveDiagram(diagramFilename)) {
        logger.log(AuroraLogger::Level::SUCCESS, "MERMAID")
          << "Mermaid diagram saved to " << diagramFilename << "\n";
        logger.log(AuroraLogger::Level::INFO, "MERMAID")
          << "Diagram can be rendered with Mermaid tools or viewers" << "\n";
      } else {
        logger.log(AuroraLogger::Level::ERROR, "MERMAID")
          << "Failed to save Mermaid diagram" << "\n";
      }
    }
    
    // Print the IR after applying passes if debug mode is enabled
    if (debugIR) {
      logger.log(AuroraLogger::Level::DEBUG, "IR")
        << "\n==== IR AFTER OPTIMIZATION ====\n";
      module->print(llvm::outs());
      logger.log(AuroraLogger::Level::DEBUG, "IR")
        << "\n==== END IR AFTER OPTIMIZATION ====\n\n";
      
      // Dump the IR to a file after optimization
      std::string afterOptFilename = outputFilename + ".after-opt.mlir";
      std::error_code EC;
      auto afterOptFile = std::make_unique<llvm::ToolOutputFile>(
          afterOptFilename, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        logger.log(AuroraLogger::Level::ERROR, "FILE")
          << "Failed to open debug IR output file: " << afterOptFilename
          << " - " << EC.message() << "\n";
      } else {
        module->print(afterOptFile->os());
        afterOptFile->keep();
        logger.log(AuroraLogger::Level::INFO, "DEBUG")
          << "Dumped IR after optimization to " << afterOptFilename << "\n";
      }
    }
    
    logger.log(AuroraLogger::Level::SUCCESS, "OPT")
      << "Optimization pipeline completed successfully" << "\n";
  }
  
  // Set up the output file
  AuroraLogger::ScopedTimer outputTimer(logger, "Output generation", verbose);
  logger.log(AuroraLogger::Level::INFO, "OUTPUT")
    << "Creating output file: " << outputFilename << "\n";
    
  std::error_code EC;
  auto output = std::make_unique<llvm::ToolOutputFile>(
      outputFilename, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    logger.log(AuroraLogger::Level::ERROR, "OUTPUT")
      << "Failed to open output file: " << EC.message() << "\n";
    return 1;
  }
  
  // Output the result
  if (emitMLIR) {
    // Output the MLIR IR
    logger.log(AuroraLogger::Level::INFO, "OUTPUT")
      << "Emitting MLIR IR to " << outputFilename << "\n";
    module->print(output->os());
    logger.log(AuroraLogger::Level::DEBUG, "OUTPUT")
      << "MLIR content written successfully" << "\n";
  } else {
    // Output the compiled model
    logger.log(AuroraLogger::Level::INFO, "CODEGEN")
      << "Compiling model for " << targetDevice << " to " << outputFilename << "\n";
    
    // Create the appropriate target backend
    auto backend = createTargetBackend(targetDevice, logger);
    
    // Generate code with timing
    AuroraLogger::ScopedTimer codegenTimer(logger, "Code generation for " + targetDevice, verbose);
    
    // Generate code for the target device
    if (!backend->generateCode(*module, output->os(), logger, verbose)) {
      logger.log(AuroraLogger::Level::ERROR, "CODEGEN")
        << "Failed to generate code for target: " << targetDevice << "\n";
      return 1;
    }
    
    logger.log(AuroraLogger::Level::DEBUG, "CODEGEN")
      << "Generated code for target: " << targetDevice << "\n";
  }
  
  // Keep the output file
  output->keep();
  
  // Get final compilation time
  auto compilationEnd = std::chrono::high_resolution_clock::now();
  auto compilationStart = mainTimer.getStartTime();
  auto compilationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
    compilationEnd - compilationStart).count();
    
  // Only print statistics in verbose mode
  if (verbose) {
    // Print the main statistics summary with [STATS] tags
    logger.log(AuroraLogger::Level::INFO) 
      << "[STATS]\n"
      << "  Total operations: " << stats.totalOperations << "\n"
      << "  Aurora operations: " << stats.auroraOperations << "\n"
      << "  Compilation time: " << compilationTimeMs << " ms\n"
      << "[/STATS]\n";
      
    // Print additional detailed statistics in debug level
    logger.log(AuroraLogger::Level::DEBUG, "STATS") << "Additional statistics:\n";
    if (stats.fusions > 0) {
      logger.log(AuroraLogger::Level::DEBUG, "STATS") 
        << "Operations fused: " << stats.fusions << "\n";
    }
    if (stats.matmulBiasFusions > 0) {
      logger.log(AuroraLogger::Level::DEBUG, "STATS") 
        << "MatMul+Bias fusions: " << stats.matmulBiasFusions << "\n";
    }
  }
  
  logger.log(AuroraLogger::Level::SUCCESS, "COMPILE")
    << "Compilation successful" << "\n";
  return 0;
}
