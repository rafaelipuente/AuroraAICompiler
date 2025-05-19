//===- AuroraRuntime.h - Aurora runtime declarations --------------*- C++ -*-===//
//
// This file contains the declarations for the Aurora runtime support.
//
//===----------------------------------------------------------------------===//

#ifndef AURORA_RUNTIME_H
#define AURORA_RUNTIME_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace mlir {
namespace aurora {
namespace runtime {

// Tensor class for Aurora runtime
class AuroraTensor {
public:
  // Create a new tensor with the given shape and data type
  AuroraTensor(const std::vector<int64_t> &shape, const std::string &dtype);
  
  // Create a tensor with existing data
  AuroraTensor(const std::vector<int64_t> &shape, const std::string &dtype, void *data);
  
  ~AuroraTensor();
  
  // Accessors
  const std::vector<int64_t> &getShape() const;
  const std::string &getDType() const;
  void *getData() const;
  size_t getSizeInBytes() const;
  
  // Data operations
  void copyFrom(const void *srcData, size_t sizeInBytes);
  void copyTo(void *dstData, size_t sizeInBytes) const;

private:
  std::vector<int64_t> shape_;
  std::string dtype_;
  std::unique_ptr<char[]> data_;
  bool ownsData_;
  size_t sizeInBytes_;
};

// Execution context for Aurora models
class AuroraExecutionContext {
public:
  AuroraExecutionContext();
  
  // Memory management
  void *allocate(size_t sizeInBytes);
  void deallocate(void *ptr);
  
  // Device management
  void setDevice(const std::string &device);
  const std::string &getDevice() const;
  
  // Profiling
  void enableProfiling(bool enable);
  bool isProfilingEnabled() const;
  void addProfilePoint(const std::string &name, double timeMs);
  std::unordered_map<std::string, double> getProfile() const;
  
private:
  std::string device_;
  bool profilingEnabled_;
  std::unordered_map<std::string, double> profile_;
};

// Aurora model class
class AuroraModel {
public:
  AuroraModel();
  // Need non-default destructor for the PIMPL pattern with forward-declared Impl
  ~AuroraModel();
  
  // Model loading
  static std::unique_ptr<AuroraModel> loadFromFile(const std::string &filename);
  static std::unique_ptr<AuroraModel> loadFromMemory(const void *data, size_t sizeInBytes);
  
  // Model information
  size_t getNumInputs() const;
  size_t getNumOutputs() const;
  std::vector<std::string> getInputNames() const;
  std::vector<std::string> getOutputNames() const;
  std::vector<int64_t> getInputShape(const std::string &name) const;
  std::vector<int64_t> getOutputShape(const std::string &name) const;
  std::string getInputDType(const std::string &name) const;
  std::string getOutputDType(const std::string &name) const;
  
  // Model execution
  void execute(
      const std::unordered_map<std::string, AuroraTensor*> &inputs,
      std::unordered_map<std::string, AuroraTensor*> &outputs,
      AuroraExecutionContext *context = nullptr);
  
  // Benchmarking
  std::unordered_map<std::string, double> benchmark(
      const std::unordered_map<std::string, AuroraTensor*> &inputs,
      std::unordered_map<std::string, AuroraTensor*> &outputs,
      int numRuns = 10, 
      int warmupRuns = 3);
  
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Main runtime API
class AuroraRuntime {
public:
  AuroraRuntime();
  ~AuroraRuntime();

  // Initialize the runtime with the given model
  bool initialize(const std::string &modelPath);

  // Execute the model with the given inputs
  bool execute(const std::vector<float> &inputs, std::vector<float> &outputs);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace aurora
} // namespace mlir

#endif // AURORA_RUNTIME_H
