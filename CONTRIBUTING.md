# Contributing to AuroraAICompiler

Thank you for your interest in contributing to AuroraAICompiler. This document covers the basics of building, testing, and submitting changes.

## Reporting Bugs

1. Search existing issues first.
2. Open a new issue with a clear title, steps to reproduce, expected vs. actual behavior, and any relevant logs.

## Submitting Changes

1. Fork the repository.
2. Create a topic branch from `master`.
3. Make small, focused commits.
4. Submit a pull request with a clear description of the change.

## Development Setup

### Prerequisites

- CMake 3.20+
- LLVM/MLIR 17 (built with `-DLLVM_ENABLE_PROJECTS=mlir`)
- C++17 compatible compiler
- Python 3.8+ (optional, for ONNX loader)

### Build

```bash
git clone https://github.com/rafaelipuente/AuroraAICompiler.git
cd AuroraAICompiler
mkdir build && cd build
cmake -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  ..
ninja
```

You must point `MLIR_DIR` and `LLVM_DIR` at your own LLVM build.

### Testing

Run lit/FileCheck tests with:

```bash
ninja check-aurora
```

This runs all 13 lit/FileCheck tests: dialect roundtrip, verifier, fusion, lowering, bufferization, and full LLVM dialect pipeline. C++ unit tests under `test/unit/` do not currently compile.

### Coding Style

- C++: Follow LLVM coding standards.
- Python: Follow PEP 8.
- Document public APIs. Avoid redundant comments.

## Project Structure

- `include/` -- C++ headers (dialect TableGen, pass declarations)
- `src/` -- C++ implementations (dialect, passes, lowering, runtime)
- `tools/` -- Compiler driver (`aurora-compile`) and supporting tools
- `python/aurora/` -- ONNX import utilities
- `test/` -- Tests (being restructured)
- `models/` -- Model generation scripts and test fixtures
- `docs/` -- Documentation

## Communication

GitHub Issues is the primary channel for discussion, bugs, and feature requests.
