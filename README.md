# AuroraAICompiler

A prototype MLIR-based compiler for neural network operators. Defines a custom Aurora dialect, implements operator fusion passes, and provides a progressive lowering pipeline from Aurora IR to LLVM dialect via Linalg and bufferization.

> **Status**: Early prototype. The dialect, a working fusion pass, and a full lowering pipeline from Aurora IR to LLVM dialect work end-to-end on `.mlir` inputs. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for a detailed engineering breakdown.

---

## Quick Demo

After building (see [Building](#building)), run the fusion pass on the included two-layer linear network:

```bash
aurora-opt examples/matmul_bias_fusion.mlir --aurora-matmul-bias-fusion
```

**Input** -- two `matmul` + `bias_add` pairs with a `relu` between layers:

```mlir
func.func @two_layer_linear(%input: tensor<2x8xf32>, %W0: tensor<8x16xf32>,
                             %b0: tensor<16xf32>, %W1: tensor<16x4xf32>,
                             %b1: tensor<4xf32>) -> tensor<2x4xf32> {
  %mm0 = aurora.matmul(%input, %W0) : (tensor<2x8xf32>, tensor<8x16xf32>) -> tensor<2x16xf32>
  %z0  = aurora.bias_add(%mm0, %b0) : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  %a0  = aurora.relu(%z0) : (tensor<2x16xf32>) -> tensor<2x16xf32>
  %mm1 = aurora.matmul(%a0, %W1) : (tensor<2x16xf32>, tensor<16x4xf32>) -> tensor<2x4xf32>
  %z1  = aurora.bias_add(%mm1, %b1) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %z1 : tensor<2x4xf32>
}
```

**Output** -- both pairs fused, `relu` preserved:

```mlir
func.func @two_layer_linear(%input: tensor<2x8xf32>, %W0: tensor<8x16xf32>,
                             %b0: tensor<16xf32>, %W1: tensor<16x4xf32>,
                             %b1: tensor<4xf32>) -> tensor<2x4xf32> {
  %0 = aurora.matmul_bias(%input, %W0, %b0)
         : (tensor<2x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
  %1 = aurora.relu(%0) : (tensor<2x16xf32>) -> tensor<2x16xf32>
  %2 = aurora.matmul_bias(%1, %W1, %b1)
         : (tensor<2x16xf32>, tensor<16x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}
```

Use `--mlir-print-ir-after-all` for step-by-step IR dumps. To go all the way to LLVM dialect, append the full downstream pipeline (see [aurora-opt usage](#aurora-opt-pass-driver)).

---

## What This Project Is

- A custom **MLIR dialect** (`aurora.*`) for neural network operators: MatMul, BiasAdd, Add, Relu, Conv, LayerNorm, FusedAttention, and the fused MatMulBias
- A working **MatMul+Bias fusion pass** that rewrites `aurora.matmul` + `aurora.bias_add` into `aurora.matmul_bias` via MLIR's greedy pattern rewrite infrastructure
- A **pass driver** (`aurora-opt`) built on `MlirOptMain` with standard MLIR debug/timing flags
- A **lowering pass** (`--convert-aurora-to-linalg`) that lowers Aurora ops to Linalg/Arith/Tensor
- A **full downstream pipeline** from Aurora IR to **LLVM dialect** via one-shot bufferization, Linalg-to-loops, and standard conversions -- all driven from `aurora-opt`
- A **wrapper script** (`scripts/aurora-to-obj.sh`) that chains `aurora-opt` → `mlir-translate` → `llc` to produce an object file (`.o`) from Aurora IR when the external LLVM tools are available
- A **compiler driver** (`aurora-compile`) that wraps the fusion pipeline with verbose output and Mermaid diagram generation
- **Lit/FileCheck tests** for dialect roundtrip, verifier enforcement, fusion, lowering, bufferization, and LLVM dialect output
- **GitHub Actions CI** that builds the project, runs `check-aurora`, and smoke-tests the wrapper
- Passes registered via **TableGen** (`Passes.td` + `GEN_PASS_REGISTRATION`), following MLIR standalone-project conventions

## What This Project Is Not

- Not a standalone executable compiler -- `aurora-opt` stops at LLVM *dialect*. The wrapper script produces object files, but a linked executable requires a C/C++ driver with concrete tensor data
- Not a production compiler -- no path from ONNX/PyTorch to executable code
- Not a runtime -- the C++ runtime classes are stubbed
- Not benchmarked -- there are no real performance measurements

---

## Current Status

| Component | State | Notes |
|---|---|---|
| Aurora Dialect (TableGen + C++) | **Working** | 8 ops defined in ODS; `bias_add` has a custom verifier |
| `aurora-opt` (MlirOptMain) | **Working** | Pass driver with `--aurora-matmul-bias-fusion`, `--aurora-fusion`, standard MLIR flags |
| MatMulBias Fusion Pass | **Working** | Fuses `matmul` + `bias_add` -> `matmul_bias`; registered via TableGen |
| ConvRelu Fusion Pass | **Stub** | Pattern skeleton; no fused op defined yet |
| `aurora-compile` (MLIR input) | **Working** | Parses `.mlir`, runs passes, emits IR, generates Mermaid diagrams |
| Aurora -> Linalg lowering | **Working** | `--convert-aurora-to-linalg` lowers relu, add, matmul, bias_add, matmul_bias |
| Bufferization (tensor -> memref) | **Working (LLVM 17+)** | `--one-shot-bufferize`; LLVM 16 needs `allow-return-allocs` instead |
| Aurora -> LLVM dialect | **Working (LLVM 17+)** | Full 9-pass pipeline via `aurora-opt --pass-pipeline` |
| Lit Tests | **Working** | 13 tests; 2 bufferize/LLVM-dialect tests require LLVM 17+ |
| ONNX Loader (Python) | **Working** | Parses ONNX protobuf, builds graph representation |
| ONNX -> Aurora IR emitter | **Broken** | Emits placeholder ops, not valid Aurora dialect |
| C++ Runtime | **Stub** | Tensor/context structures exist; execution is fake |
| CI (GitHub Actions) | **Working** | Builds with LLVM 17, runs `check-aurora`, smoke-tests wrapper |

---

## Architecture

```
  .mlir input (Aurora dialect)
       │
       ▼
  --aurora-matmul-bias-fusion       [optimization: 5 ops -> 3 ops]
       │
       ▼
  --convert-aurora-to-linalg        [lowering: Aurora -> Linalg/Arith on tensors]
       │
       ▼
  --one-shot-bufferize              [tensor -> memref; concrete memory semantics]
       │
       ▼
  --convert-linalg-to-loops         [structured ops -> SCF for loops]
  --convert-scf-to-cf
       │
       ▼
  --convert-{index,arith,cf,memref,func}-to-llvm
  --reconcile-unrealized-casts
       │
       ▼
  LLVM dialect IR  ← aurora-opt stops here
       │
       ▼   (scripts/aurora-to-obj.sh chains these external tools)
  mlir-translate --mlir-to-llvmir
       │
       ▼
  llc -filetype=obj  →  output.o
```

`aurora-compile` wraps the Aurora optimization passes with a richer CLI (Mermaid diagrams, verbose output).

---

## Aurora Dialect Operations

Defined in [`include/Aurora/Dialect/Aurora/AuroraOps.td`](include/Aurora/Dialect/Aurora/AuroraOps.td):

| Operation | Signature | Notes |
|---|---|---|
| `aurora.matmul` | `(tensor, tensor) -> tensor` | Used in fusion |
| `aurora.matmul_bias` | `(tensor, tensor, tensor) -> tensor` | Produced by fusion pass |
| `aurora.bias_add` | `(ND tensor, 1D tensor) -> ND tensor` | Custom verifier: rank-1 bias, last-dim match |
| `aurora.add` | `(tensor, tensor) -> tensor` | Element-wise, `SameOperandsAndResultType` |
| `aurora.relu` | `(tensor) -> tensor` | `SameOperandsAndResultType` |
| `aurora.conv` | `(tensor, tensor) -> tensor` | No stride/padding attributes yet |
| `aurora.layernorm` | `(tensor, tensor, tensor) -> tensor` | No lowering |
| `aurora.fused_attention` | `(tensor, tensor, tensor) -> tensor` | No lowering |

---

## Building

### Prerequisites

- CMake 3.20+
- LLVM/MLIR 16+ (built with `-DLLVM_ENABLE_PROJECTS=mlir`)
  - **LLVM 16+**: Aurora dialect, fusion pass, Aurora → Linalg lowering, all corresponding lit tests
  - **LLVM 17+ required** for: `--one-shot-bufferize="allow-return-allocs-in-loops=true"` and the full Aurora → LLVM dialect pipeline tests (`test/Conversion/aurora-bufferize.mlir`, `test/Conversion/aurora-to-llvm.mlir`). On LLVM 16, substitute `allow-return-allocs=true`.
- C++17 compiler
- Python 3.8+ (optional, for ONNX loader and lit tests)

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

> You must point `MLIR_DIR` and `LLVM_DIR` at your LLVM build. The CMake configuration does not download or build LLVM for you.

### Run Tests

```bash
ninja check-aurora
```

This runs all lit/FileCheck tests: dialect roundtrip, verifier enforcement, fusion, lowering to Linalg, bufferization, and the full pipeline to LLVM dialect. Two tests (`aurora-bufferize.mlir`, `aurora-to-llvm.mlir`) require LLVM 17+ and will fail on LLVM 16 with "unknown option: allow-return-allocs-in-loops".

---

## Usage

### aurora-opt (pass driver)

The primary tool for applying and debugging individual passes:

```bash
# Run the matmul+bias fusion pass
aurora-opt examples/matmul_bias_fusion.mlir --aurora-matmul-bias-fusion

# Roundtrip (parse + print, no passes) -- validates dialect parsing
aurora-opt test_model.mlir

# Step-by-step IR dumps
aurora-opt examples/matmul_bias_fusion.mlir --aurora-matmul-bias-fusion \
  --mlir-print-ir-after-all

# Pass timing
aurora-opt examples/matmul_bias_fusion.mlir --aurora-matmul-bias-fusion \
  --mlir-timing

# Lower Aurora ops to Linalg/Arith (no fusion)
aurora-opt examples/lower_to_linalg.mlir --convert-aurora-to-linalg

# Fuse then lower: the full pipeline to Linalg
aurora-opt examples/matmul_bias_fusion.mlir \
  --aurora-matmul-bias-fusion --convert-aurora-to-linalg

# Bufferize: Linalg-on-tensors -> Linalg-on-memrefs (LLVM 17+)
aurora-opt examples/lower_to_linalg.mlir \
  --convert-aurora-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-in-loops=true"

# Full pipeline to LLVM dialect (LLVM 17+)
aurora-opt examples/pipeline_to_llvm.mlir \
  --convert-aurora-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-in-loops=true" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-index-to-llvm \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

# Same via pipeline string
aurora-opt examples/pipeline_to_llvm.mlir \
  --pass-pipeline="builtin.module(convert-aurora-to-linalg,one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs-in-loops=true},convert-linalg-to-loops,convert-scf-to-cf,convert-index-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"

# List all registered passes (now includes full MLIR standard set)
aurora-opt --help
```

### aurora-to-obj.sh (object file wrapper)

Chains `aurora-opt` → `mlir-translate` → `llc` to produce an object file from Aurora IR. Requires `mlir-translate` and `llc` from an LLVM 17+ installation.

```bash
# Full pipeline: Aurora IR -> object file
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt examples/pipeline_to_llvm.mlir

# With fusion
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt --fuse examples/matmul_bias_fusion.mlir

# Stop at LLVM dialect (no external tools needed)
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt --emit-llvm-dialect examples/pipeline_to_llvm.mlir

# Stop at LLVM IR (.ll)
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt --emit-llvm-ir examples/pipeline_to_llvm.mlir

# Verbose output showing each stage
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt --verbose examples/pipeline_to_llvm.mlir
```

The wrapper produces an object file (`.o`), not a linked executable. The generated code calls `malloc`/`free` and has no `main()`.

### aurora-compile (full pipeline driver)

The user-facing driver with statistics, Mermaid, and colored output:

```bash
# Run fusion and emit transformed IR
aurora-compile test_before_fusion.mlir -o output.mlir --emit-mlir --fuse-matmul-bias

# Generate a Mermaid operation graph
aurora-compile test_model.mlir -o output.mlir --emit-mlir --dump-mermaid

# Verbose mode with pass timing
aurora-compile test_before_fusion.mlir -o output.mlir --emit-mlir --verbose
```

---

## Project Structure

```
include/Aurora/
  Dialect/Aurora/        # TableGen (.td) and C++ headers for Aurora ops
  Transforms/            # Passes.td, pass declarations, generated .inc
  Runtime/               # Runtime class declarations (stubbed)
  Conversion/            # Passes.td + generated .inc for conversion passes

src/
  Dialect/Aurora/        # Dialect and op C++ implementations
  Transforms/            # Pass implementations (MatMulBiasFusion, Fusion)
  Conversion/            # Aurora -> Linalg/Arith lowering
  Runtime/               # Runtime C++ implementation (stubbed)

tools/
  aurora-opt/            # MlirOptMain-based pass driver
  aurora-compile/        # Full pipeline compiler driver
  aurora-run/            # Model runner (stubbed)
  aurora-benchmark/      # Benchmark harness (stubbed)

test/
  Dialect/Aurora/        # Lit tests: op roundtrip, verifier negative tests
  Transforms/            # Lit tests: fusion positive and negative
  Conversion/            # Lit tests: Aurora -> Linalg, bufferization, full LLVM pipeline
  unit/                  # GoogleTest C++ tests (not yet compiling)

examples/                # Demo .mlir files
scripts/                 # aurora-to-obj.sh (Aurora IR -> object file wrapper)
python/aurora/           # ONNX import utilities
models/                  # Model generation scripts
```

---

## Known Issues

- **ConvOp has no attributes**: `aurora.conv` takes only `input` and `filter`. Strides, padding, dilation, and groups are not yet defined in ODS, but downstream unit test code assumes they exist.
- **No lowering for `conv`, `layernorm`, `fused_attention`**: These ops remain in Aurora IR after `--convert-aurora-to-linalg`. The five core ops (relu, add, matmul, bias_add, matmul_bias) are fully lowered.
- **`aurora-fusion` drops relu on conv→relu patterns**: The `--aurora-fusion` pass (`Fusion.cpp`) contains a placeholder `ConvReluFusionPattern` that replaces `conv → relu` with just `conv`, silently discarding the relu. This is a known placeholder; no `aurora.conv_relu` fused op exists yet. Do not run `--aurora-fusion` on IR that depends on relu correctness.
- **Unit tests reference non-existent attributes**: `TestAuroraDialect.cpp` and `TestMatMulBiasFusion.cpp` use ConvOp and MatMulOp attributes that are not in the TableGen definitions.
- **Bufferization/LLVM tests require LLVM 17+**: `test/Conversion/aurora-bufferize.mlir` and `test/Conversion/aurora-to-llvm.mlir` use `allow-return-allocs-in-loops` which was introduced in LLVM 17. On LLVM 16, substitute `allow-return-allocs=true`. These tests will fail on LLVM 16 with "unknown option".
- **Object files are not directly executable**: The wrapper (`scripts/aurora-to-obj.sh`) produces `.o` files whose functions call `malloc`/`free`. Linking into a runnable binary requires a C/C++ driver that sets up concrete tensor data and calls the generated functions.

---

## Next Milestones

1. **Linked executable demo** -- Write a small C driver that calls the generated function with concrete tensor data and links against the object file.
2. **Fix ConvOp** -- Add strides/padding/dilation attributes to `AuroraOps.td` or strip downstream code that references them.
3. **Fix ONNX -> Aurora IR** -- Make `onnx_to_mlir.py` emit valid Aurora dialect ops.
4. **Add canonicalization patterns** -- Fold constant shapes, dead code elimination, etc.

---

## Motivation

This project explores the work of an AI compiler engineer: designing MLIR dialects for neural network operators, implementing transformation passes, and building the infrastructure for progressive lowering to machine code. It is inspired by compiler teams at companies like Ampere Computing, Google, and AMD that build MLIR-based toolchains for ML hardware.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
