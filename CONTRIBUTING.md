# Contributing to AuroraAICompiler

Thank you for your interest in contributing to AuroraAICompiler! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

All contributors are expected to adhere to the project's Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported by searching the issues
2. If not, open a new issue with a clear title and description
3. Include steps to reproduce, expected behavior, and actual behavior
4. Attach relevant logs, screenshots, or other supporting information

### Suggesting Enhancements

1. Open a new issue with a clear title and description
2. Explain why the enhancement would be valuable
3. Describe how it should work
4. Consider how it might impact other parts of the codebase

### Submitting Changes

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes in small, logical commits
4. Add or update tests to cover your changes
5. Ensure existing tests pass
6. Submit a pull request

### Pull Request Process

1. Update the README.md or documentation with details of changes if needed
2. The PR should work on the main supported platforms (Linux, macOS, Windows)
3. PR will be merged once it passes review and CI checks

## Development Setup

### Prerequisites

- CMake 3.20+
- LLVM/MLIR 16.0+
- Python 3.8+
- C++17 compatible compiler

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/AuroraAICompiler.git
cd AuroraAICompiler

# Configure and build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
make check-aurora
```

### Coding Style

- C++: Follow the LLVM coding standards
- Python: Follow PEP 8 style guide
- Use consistent naming conventions within each language
- Document public APIs with appropriate docstrings/comments

## Project Structure

- `include/` - C++ header files
- `src/` - C++ implementation files
- `python/` - Python bindings and utilities
- `examples/` - Example models and usage
- `tools/` - Command-line utilities
- `test/` - Unit and integration tests
- `benchmarks/` - Performance measurement tools
- `models/` - Sample AI models
- `docs/` - Documentation

## Communication

- GitHub Issues: Bug reports, feature requests, and discussion
- Pull Requests: Code reviews and implementation discussion

Thank you for contributing to AuroraAICompiler!
