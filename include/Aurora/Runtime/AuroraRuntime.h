#ifndef AURORA_RUNTIME_AURORA_RUNTIME_H
#define AURORA_RUNTIME_AURORA_RUNTIME_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace aurora {
namespace runtime {

// Forward declarations
class AuroraModel;
class AuroraTensor;
class AuroraExecutionContext;

/**
 * AuroraTensor - Represents a multi-dimensional tensor in the Aurora runtime
 */
class AuroraTensor {
public:
  AuroraTensor(const std::vector<int64_t> &shape, const std::string &dtype);
  AuroraTensor(const std::vector<int64_t> &shape, const std::string &dtype, void *data);
  
  // Getters
  const std::vector<int64_t> &getShape() const;
  const std::string &getDType() const;
  void *getData() const;
  size_t getSizeInBytes() const;
  
  // Data transfer
  void copyFrom(const void *srcData, size_t sizeInBytes);
  void copyTo(void *dstData, size_t sizeInBytes) const;
  
private:
  std::vector<int64_t> shape_;
  std::string dtype_;
  std::unique_ptr<char[]> data_;
  size_t sizeInBytes_;
};

/**
 * AuroraExecutionContext - Holds the execution context for a model
 */
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

/**
 * AuroraModel - Represents a compiled Aurora model
 */
class AuroraModel {
public:
  // Constructor loads a compiled model from file
  static std::unique_ptr<AuroraModel> loadFromFile(const std::string &filename);
  
  // Constructor from memory buffer (useful for JIT compilation)
  static std::unique_ptr<AuroraModel> loadFromMemory(const void *data, size_t sizeInBytes);
  
  // Get input/output information
  size_t getNumInputs() const;
  size_t getNumOutputs() const;
  std::vector<std::string> getInputNames() const;
  std::vector<std::string> getOutputNames() const;
  std::vector<int64_t> getInputShape(const std::string &name) const;
  std::vector<int64_t> getOutputShape(const std::string &name) const;
  std::string getInputDType(const std::string &name) const;
  std::string getOutputDType(const std::string &name) const;
  
  // Execute the model
  void execute(const std::unordered_map<std::string, AuroraTensor*> &inputs,
              std::unordered_map<std::string, AuroraTensor*> &outputs,
              AuroraExecutionContext *context = nullptr);
  
  // Benchmark the model execution
  std::unordered_map<std::string, double> benchmark(
      const std::unordered_map<std::string, AuroraTensor*> &inputs,
      std::unordered_map<std::string, AuroraTensor*> &outputs,
      int numRuns = 100, 
      int warmupRuns = 10);
      
private:
  AuroraModel();
  
  // Private implementation details
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace aurora

#endif // AURORA_RUNTIME_AURORA_RUNTIME_H
