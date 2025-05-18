#include "Aurora/Runtime/AuroraRuntime.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace aurora::runtime;
using namespace llvm;

// Command line options
static cl::opt<std::string> modelFilename(
    "model", cl::desc("Compiled Aurora model file"), cl::Required);

static cl::opt<std::string> inputFilename(
    "input", cl::desc("Input data file"), cl::Required);

static cl::opt<std::string> outputFilename(
    "output", cl::desc("Output data file (default: stdout)"), cl::init("-"));

static cl::opt<std::string> device(
    "device", cl::desc("Execution device (cpu, cuda)"), cl::init("cpu"));

static cl::opt<bool> profile(
    "profile", cl::desc("Enable profiling"), cl::init(false));

// Utility function to read input data
bool readInputData(const std::string &filename, std::vector<float> &data, 
                  std::vector<int64_t> &shape) {
  // In a real implementation, this would read actual data from a file
  // For this placeholder, we'll create dummy data
  
  // Simulate a 1x3x224x224 tensor (typical image input shape)
  shape = {1, 3, 224, 224};
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  
  data.resize(size, 0.0f);
  
  // Fill with some pattern
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<float>(i % 255) / 255.0f;
  }
  
  llvm::outs() << "Read input tensor with shape [";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) llvm::outs() << ", ";
    llvm::outs() << shape[i];
  }
  llvm::outs() << "]\n";
  
  return true;
}

// Utility function to write output data
bool writeOutputData(const std::string &filename, const std::vector<float> &data,
                    const std::vector<int64_t> &shape) {
  // Calculate total size
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  
  if (filename == "-") {
    // Write to stdout
    llvm::outs() << "Output tensor with shape [";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) llvm::outs() << ", ";
      llvm::outs() << shape[i];
    }
    llvm::outs() << "]:\n";
    
    // For large tensors, just print a summary
    if (size > 20) {
      llvm::outs() << "  First 10 values: ";
      for (size_t i = 0; i < std::min(size_t(10), size); ++i) {
        if (i > 0) llvm::outs() << ", ";
        llvm::outs() << std::fixed << std::setprecision(4) << data[i];
      }
      llvm::outs() << "\n";
      
      // Find the maximum value and its index
      size_t maxIdx = 0;
      float maxVal = data[0];
      for (size_t i = 1; i < size; ++i) {
        if (data[i] > maxVal) {
          maxVal = data[i];
          maxIdx = i;
        }
      }
      
      llvm::outs() << "  Max value: " << maxVal << " at index " << maxIdx << "\n";
    } else {
      // Print all values for small tensors
      for (size_t i = 0; i < size; ++i) {
        llvm::outs() << std::fixed << std::setprecision(4) << data[i] << " ";
      }
      llvm::outs() << "\n";
    }
  } else {
    // In a real implementation, this would write to a file
    llvm::outs() << "Writing output tensor to " << filename << "\n";
  }
  
  return true;
}

int main(int argc, char **argv) {
  // Initialize LLVM
  InitLLVM y(argc, argv);
  
  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "Aurora Model Runner\n");
  
  llvm::outs() << "Loading model: " << modelFilename << "\n";
  
  // Load the model
  auto model = AuroraModel::loadFromFile(modelFilename);
  if (!model) {
    llvm::errs() << "Failed to load model: " << modelFilename << "\n";
    return 1;
  }
  
  // Get model metadata
  auto inputNames = model->getInputNames();
  auto outputNames = model->getOutputNames();
  
  if (inputNames.empty() || outputNames.empty()) {
    llvm::errs() << "Model has no inputs or outputs\n";
    return 1;
  }
  
  // Read input data
  std::vector<float> inputData;
  std::vector<int64_t> inputShape;
  if (!readInputData(inputFilename, inputData, inputShape)) {
    llvm::errs() << "Failed to read input data: " << inputFilename << "\n";
    return 1;
  }
  
  // Create input tensor
  AuroraTensor inputTensor(inputShape, "float32", inputData.data());
  
  // Setup execution context
  AuroraExecutionContext context;
  context.setDevice(device);
  context.enableProfiling(profile);
  
  // Prepare input/output map
  std::unordered_map<std::string, AuroraTensor*> inputs;
  inputs[inputNames[0]] = &inputTensor;
  
  // Create output tensors
  std::vector<std::unique_ptr<AuroraTensor>> outputTensors;
  std::unordered_map<std::string, AuroraTensor*> outputs;
  
  for (const auto &name : outputNames) {
    auto shape = model->getOutputShape(name);
    auto dtype = model->getOutputDType(name);
    
    outputTensors.push_back(std::make_unique<AuroraTensor>(shape, dtype));
    outputs[name] = outputTensors.back().get();
  }
  
  // Execute the model
  llvm::outs() << "Executing model on device: " << device << "\n";
  auto startTime = std::chrono::high_resolution_clock::now();
  
  model->execute(inputs, outputs, &context);
  
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  
  llvm::outs() << "Execution time: " << duration.count() << " ms\n";
  
  // Show profiling results if enabled
  if (profile) {
    llvm::outs() << "Profiling results:\n";
    auto profileData = context.getProfile();
    for (const auto &entry : profileData) {
      llvm::outs() << "  " << entry.first << ": " << entry.second << " ms\n";
    }
  }
  
  // Write output data
  if (!outputNames.empty()) {
    auto *outputTensor = outputs[outputNames[0]];
    
    // Convert tensor data to float vector for writing
    size_t outputSize = 1;
    for (auto dim : outputTensor->getShape()) {
      outputSize *= dim;
    }
    
    std::vector<float> outputData(outputSize);
    outputTensor->copyTo(outputData.data(), outputSize * sizeof(float));
    
    if (!writeOutputData(outputFilename, outputData, outputTensor->getShape())) {
      llvm::errs() << "Failed to write output data: " << outputFilename << "\n";
      return 1;
    }
  }
  
  return 0;
}
