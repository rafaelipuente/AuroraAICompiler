//===- standalone_matmul_bias_example.cpp - MatMulBias Fusion Demo --------===//
//
// Aurora AI Compiler
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cmath>

// Simple tensor class for demonstration
class Tensor {
public:
  Tensor(std::vector<int> dims, std::vector<float> data) 
      : dims_(dims), data_(data) {}
  
  // Create a tensor with specific shape filled with a value
  static Tensor zeros(const std::vector<int>& dims) {
    int size = 1;
    for (int d : dims) size *= d;
    return Tensor(dims, std::vector<float>(size, 0.0f));
  }
  
  // Create a tensor with specific shape filled with random values
  static Tensor random(const std::vector<int>& dims) {
    int size = 1;
    for (int d : dims) size *= d;
    std::vector<float> data(size);
    for (int i = 0; i < size; i++) {
      data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Tensor(dims, data);
  }
  
  const std::vector<int>& dims() const { return dims_; }
  const std::vector<float>& data() const { return data_; }
  std::vector<float>& data() { return data_; }
  
  // Get total number of elements
  int size() const {
    int s = 1;
    for (int d : dims_) s *= d;
    return s;
  }
  
  void print(const std::string& name = "") const {
    if (!name.empty()) {
      std::cout << name << ": ";
    }
    
    std::cout << "Tensor[";
    for (size_t i = 0; i < dims_.size(); i++) {
      if (i > 0) std::cout << "x";
      std::cout << dims_[i];
    }
    std::cout << "] = {";
    
    // Print up to 10 elements
    const int max_display = 10;
    for (int i = 0; i < std::min(max_display, size()); i++) {
      if (i > 0) std::cout << ", ";
      std::cout << data_[i];
    }
    
    if (size() > max_display) {
      std::cout << ", ... (" << (size() - max_display) << " more)";
    }
    
    std::cout << "}" << std::endl;
  }

private:
  std::vector<int> dims_;
  std::vector<float> data_;
};

// Separate MatMul operation
Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  // Basic checks
  if (lhs.dims().size() != 2 || rhs.dims().size() != 2) {
    throw std::runtime_error("MatMul requires 2D tensors");
  }
  
  if (lhs.dims()[1] != rhs.dims()[0]) {
    throw std::runtime_error("MatMul dimension mismatch");
  }
  
  int M = lhs.dims()[0];
  int K = lhs.dims()[1];
  int N = rhs.dims()[1];
  
  // Create output tensor
  auto result = Tensor::zeros({M, N});
  
  // Simple matrix multiplication
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += lhs.data()[i * K + k] * rhs.data()[k * N + j];
      }
      result.data()[i * N + j] = sum;
    }
  }
  
  return result;
}

// Separate Add operation (with broadcasting)
Tensor add(const Tensor& lhs, const Tensor& rhs) {
  // For simplicity, we'll handle a specific broadcasting case:
  // Adding a 1D tensor (bias) to a 2D tensor (matrix)
  if (lhs.dims().size() == 2 && rhs.dims().size() == 1) {
    if (lhs.dims()[1] != rhs.dims()[0]) {
      throw std::runtime_error("Add dimension mismatch for broadcasting");
    }
    
    int M = lhs.dims()[0];
    int N = lhs.dims()[1];
    
    // Create output tensor
    auto result = Tensor::zeros({M, N});
    
    // Add with broadcasting
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        result.data()[i * N + j] = lhs.data()[i * N + j] + rhs.data()[j];
      }
    }
    
    return result;
  } else {
    throw std::runtime_error("Unsupported broadcasting configuration");
  }
}

// Fused MatMulBias operation
Tensor matmul_bias(const Tensor& lhs, const Tensor& rhs, const Tensor& bias) {
  // Basic checks
  if (lhs.dims().size() != 2 || rhs.dims().size() != 2) {
    throw std::runtime_error("MatMulBias requires 2D tensors for matmul inputs");
  }
  
  if (lhs.dims()[1] != rhs.dims()[0]) {
    throw std::runtime_error("MatMulBias dimension mismatch for matmul");
  }
  
  if (bias.dims().size() != 1 || bias.dims()[0] != rhs.dims()[1]) {
    throw std::runtime_error("MatMulBias bias dimension mismatch");
  }
  
  int M = lhs.dims()[0];
  int K = lhs.dims()[1];
  int N = rhs.dims()[1];
  
  // Create output tensor
  auto result = Tensor::zeros({M, N});
  
  // Fused matrix multiplication and bias addition
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = bias.data()[j];  // Start with bias value
      for (int k = 0; k < K; k++) {
        sum += lhs.data()[i * K + k] * rhs.data()[k * N + j];
      }
      result.data()[i * N + j] = sum;
    }
  }
  
  return result;
}

// Benchmark function
void benchmark(int iterations) {
  // Create test tensors
  auto A = Tensor::random({128, 256});
  auto B = Tensor::random({256, 512});
  auto C = Tensor::random({512});
  
  // Benchmark unfused operations
  std::cout << "\n==== Benchmark Unfused Operations ====\n";
  
  double unfused_total_time = 0.0;
  for (int i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Unfused operations
    auto matmul_result = matmul(A, B);
    auto final_result = add(matmul_result, C);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    unfused_total_time += elapsed.count();
    
    if (i == 0) {
      matmul_result.print("MatMul Result");
      final_result.print("Final Result (Unfused)");
    }
  }
  
  std::cout << "Unfused Avg Time: " << (unfused_total_time / iterations) << " ms\n";
  
  // Benchmark fused operations
  std::cout << "\n==== Benchmark Fused Operations ====\n";
  
  double fused_total_time = 0.0;
  for (int i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Fused operation
    auto fused_result = matmul_bias(A, B, C);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    fused_total_time += elapsed.count();
    
    if (i == 0) {
      fused_result.print("Final Result (Fused)");
    }
  }
  
  std::cout << "Fused Avg Time: " << (fused_total_time / iterations) << " ms\n";
  
  // Calculate speedup
  double speedup = unfused_total_time / fused_total_time;
  std::cout << "Speedup: " << speedup << "x\n";
}

int main(int argc, char* argv[]) {
  // Seed random number generator
  srand(42);
  
  std::cout << "=== Aurora AI Compiler - MatMulBias Fusion Demo ===\n";
  
  // Create small example tensors
  auto A = Tensor::random({4, 8});
  auto B = Tensor::random({8, 16});
  auto C = Tensor::random({16});
  
  A.print("A");
  B.print("B");
  C.print("C");
  
  std::cout << "\n==== Unfused Computation ====\n";
  
  // Perform unfused computation
  auto matmul_result = matmul(A, B);
  auto unfused_result = add(matmul_result, C);
  
  matmul_result.print("MatMul Result");
  unfused_result.print("Final Result (Unfused)");
  
  std::cout << "\n==== Fused Computation ====\n";
  
  // Perform fused computation
  auto fused_result = matmul_bias(A, B, C);
  
  fused_result.print("Final Result (Fused)");
  
  // Verify results match
  bool results_match = true;
  for (size_t i = 0; i < unfused_result.data().size(); i++) {
    if (std::abs(unfused_result.data()[i] - fused_result.data()[i]) > 1e-5) {
      std::cout << "Mismatch at index " << i << ": "
                << unfused_result.data()[i] << " vs " << fused_result.data()[i] << "\n";
      results_match = false;
      break;
    }
  }
  
  if (results_match) {
    std::cout << "Results Match! ✓\n";
  } else {
    std::cout << "Results Do Not Match! ✗\n";
  }
  
  // Run benchmarks
  std::cout << "\n==== Running Benchmarks ====\n";
  benchmark(100);
  
  std::cout << "\n==== Demo Complete ====\n";
  std::cout << "The MatMulBias fusion optimization has been successfully demonstrated.\n";
  
  return 0;
}
