# Project Status

Last updated: 2026-03-13

## Mission

Build an MLIR-based compiler that defines a custom dialect for neural network operators, implements pattern-based operator fusion, and demonstrates progressive lowering from high-level Aurora IR toward LLVM IR.

The immediate goal is a credible, well-tested prototype -- not a production compiler.

---

## Working Components

### Aurora Dialect (`include/Aurora/Dialect/Aurora/`, `src/Dialect/Aurora/`)

The dialect is defined via TableGen (`AuroraOps.td`) and registers 8 operations:
`aurora.matmul`, `aurora.matmul_bias`, `aurora.add`, `aurora.bias_add`, `aurora.relu`, `aurora.conv`, `aurora.layernorm`, `aurora.fused_attention`.

`aurora.add` is strict element-wise (`SameOperandsAndResultType`). `aurora.bias_add` adds a rank-1 bias along the last dimension of an ND input, with a custom C++ verifier enforcing rank and dimension constraints.

Dialect initialization, op registration, and MLIR parsing/printing of these ops work.

### MatMul+Bias Fusion Pass (`src/Transforms/MatMulBiasFusion.cpp`)

A real MLIR rewrite pattern that detects `aurora.matmul` followed by `aurora.bias_add` and fuses them into `aurora.matmul_bias`. Uses `OpRewritePattern`, the greedy pattern rewrite driver, and `LLVM_DEBUG` tracing. This is the strongest piece of the repo.

### Pass Driver (`tools/aurora-opt/aurora-opt.cpp`)

A standard `MlirOptMain`-based tool that exposes all registered Aurora passes as CLI flags (e.g. `--aurora-matmul-bias-fusion`). Calls `mlir::registerAllDialects()` and `mlir::registerAllPasses()`, making the full MLIR standard pass set available. Supports the full MLIR debugging surface: `--mlir-print-ir-after-all`, `--mlir-timing`, `--pass-pipeline`. Use `aurora-opt --help` to enumerate available passes. This is the primary tool for demonstrating and testing individual passes.

### Compiler Driver (`tools/aurora-compile/aurora-compile.cpp`)

The driver (793 lines) parses `.mlir` input, registers the Aurora dialect, runs optimization passes, and emits transformed IR. It also generates Mermaid operation graphs. The `--emit-mlir`, `--fuse-matmul-bias`, `--dump-mermaid`, `--verbose`, and `--debug-ir` flags work for `.mlir` inputs.

### Aurora -> Linalg Lowering (`src/Conversion/AuroraToLinalg/AuroraToLinalg.cpp`)

A conversion pass (`--convert-aurora-to-linalg`) that lowers four Aurora ops to standard MLIR:

- `aurora.relu` -> `linalg.generic` + `arith.maximumf`
- `aurora.add` -> `linalg.generic` + `arith.addf`
- `aurora.matmul` -> `linalg.matmul` (with `linalg.fill` for zero-init)
- `aurora.bias_add` -> `linalg.generic` (broadcast-add along last dim)
- `aurora.matmul_bias` -> `linalg.matmul` + `linalg.generic` (matmul then bias-add in two steps)

Registered via TableGen (`include/Aurora/Conversion/Passes.td`). Uses partial conversion, so ops without patterns (`matmul_bias`, `conv`, etc.) survive. Can be composed with the fusion pass: `--aurora-matmul-bias-fusion --convert-aurora-to-linalg`. The full pipeline (fusion + lowering) leaves zero Aurora ops for the five core ops (relu, add, matmul, bias_add, matmul_bias), verified by `test/Conversion/fusion-then-lower.mlir`.

### Aurora -> LLVM Dialect (`aurora-opt` + standard MLIR passes)

`aurora-opt` now calls `mlir::registerAllDialects()` and `mlir::registerAllPasses()`, making the full MLIR downstream pipeline available without any code changes. The complete lowering chain from Aurora IR to LLVM dialect is:

```
--convert-aurora-to-linalg
--one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-in-loops=true"
--convert-linalg-to-loops
--convert-scf-to-cf
--convert-index-to-llvm
--convert-arith-to-llvm
--convert-cf-to-llvm
--convert-memref-to-llvm
--convert-func-to-llvm
--reconcile-unrealized-casts
```

Requires LLVM 17+ for the `allow-return-allocs-in-loops` bufferization option (LLVM 16: use `allow-return-allocs`). Produces LLVM *dialect* IR -- not binary. The next step is `mlir-translate --mlir-to-llvmir` then `llc`.

### Lit/FileCheck Tests (`test/`)

MLIR-standard IR-level tests using `lit` and `FileCheck`. 13 `.mlir` tests in total:

- Dialect roundtrip: `test/Dialect/Aurora/ops.mlir` (8 ops)
- Verifier negative cases: `invalid.mlir`, `invalid-bias-dim.mlir`
- Fusion pass: `matmul-bias-fusion.mlir`, `matmul-bias-fusion-negative.mlir`
- Lowering to Linalg: 5 tests in `test/Conversion/` (relu, add, matmul, bias_add, matmul_bias)
- Integration: `fusion-then-lower.mlir` (fuse + lower with no aurora ops remaining)
- Bufferization (LLVM 17+): `aurora-bufferize.mlir`
- LLVM dialect (LLVM 17+): `aurora-to-llvm.mlir`

The 11 LLVM-16-compatible tests cover the core pipeline. The 2 bufferization/LLVM-dialect tests require LLVM 17+ and will fail on LLVM 16 with "unknown option: allow-return-allocs-in-loops".

Run with `ninja check-aurora`.

### ONNX Loader (`python/aurora/model_import/onnx_loader.py`)

Parses ONNX protobuf files and builds an in-memory graph representation. This code runs and is tested against small models.

---

## Partial / Incomplete

### ConvRelu Fusion (`src/Transforms/Fusion.cpp`)

A 61-line pattern skeleton (`ConvReluFusionPattern`). The pattern matches `aurora.relu` following `aurora.conv` and calls `rewriter.replaceOp(reluOp, convOp.getResult())`. **This silently discards the relu computation** -- there is no `aurora.conv_relu` fused op defined in the dialect to replace it with. Running `--aurora-fusion` on IR that depends on relu correctness produces semantically incorrect output without any warning.

**Status**: Must either (a) define `aurora.conv_relu` in TableGen and produce it, or (b) remove the pass entirely. Do not use `--aurora-fusion` in any pipeline where relu correctness matters.

### ONNX-to-Aurora Emitter (`python/aurora/model_import/onnx_to_mlir.py`)

The ONNX loader works, but the module that converts its output to MLIR emits placeholder ops (`dummy.constant`, `dummy.placeholder`) instead of valid Aurora dialect ops. The emitter cannot produce IR that `aurora-compile` can consume.

**Status**: Needs a rewrite to emit actual `aurora.*` operations.

### aurora-compile ONNX/PyTorch Paths

The `--input-format onnx` and `--input-format pytorch` flags exist in the CLI, but both create empty MLIR modules. No model data reaches the compilation pipeline.

**Status**: Dead code that should be removed until the ONNX emitter is fixed.

---

## Broken / Disabled

### ~~Aurora-to-LLVM Lowering~~ **Replaced**

The broken `AuroraToLLVM.cpp` (262 lines, non-compiling, deprecated APIs, references non-existent ConvOp accessors) has been deleted.

**Replacement**: `src/Conversion/AuroraToLinalg/AuroraToLinalg.cpp` implements `--convert-aurora-to-linalg`, which lowers `aurora.relu`, `aurora.add`, `aurora.matmul`, and `aurora.bias_add` to Linalg/Arith/Tensor ops. Uses partial conversion so unlowered ops (`matmul_bias`, `conv`, `layernorm`, `fused_attention`) remain in place. Tested via lit/FileCheck.

### Backend Codegen (`tools/aurora-compile/aurora-compile-backend.cpp`)

83 lines. Every backend (CPU, GPU, accelerator) writes a hardcoded literal string to the output file. No actual code generation occurs.

**Status**: Remove. Replace later with a real LLVM IR emission path.

### JIT Executor (`tools/aurora-jit/jit_executor.cpp`)

628 lines. Commented out of the CMake build. Contains incorrect pass registration, missing dialect setup, and broken linking assumptions.

**Status**: Disabled. Should be deleted or rewritten from scratch when a real lowering path exists.

### C++ Runtime (`src/Runtime/AuroraRuntime.cpp`)

345 lines. `AuroraTensor` and `AuroraExecutionContext` are partially real structures, but `AuroraModel::loadFromFile()` returns a dummy model, and `executeModel()` performs a memcpy with a hardcoded "10.5 ms" timing result.

**Status**: Stub. Useful as skeleton but cannot execute anything.

### Unit Tests (`test/unit/`)

Both `TestAuroraDialect.cpp` and `TestMatMulBiasFusion.cpp` reference ConvOp attributes and macros (`ASSERT_SUCCESS`) that do not exist. Neither test compiles.

**Status**: Must be rewritten to match the actual ODS definitions.

### ~~CI Workflow~~ **Repaired**

The original workflow referenced non-existent test targets and installed PyTorch for no reason.

**Replacement**: `.github/workflows/build.yml` now installs LLVM 17 from apt.llvm.org, builds with Ninja, runs `ninja check-aurora` (13 lit tests), verifies `aurora-opt --help`, and smoke-tests the `aurora-to-obj.sh` wrapper. No unnecessary dependencies.

---

## Technical Debt

1. **ODS / C++ mismatch**: `aurora.conv` has no attribute accessors in TableGen, but downstream C++ (lowering, tests) assumes `getStrides()`, `getDilation()`, `getPadding()`, `getGroups()`. This is the single biggest source of compile failures.
2. ~~**`aurora.add` vs `aurora.bias_add`**~~: **Resolved.** `aurora.add` is strict element-wise. `aurora.bias_add` handles 1D-bias broadcasting with a custom verifier. The fusion pass targets `aurora.bias_add`.
3. **Root-level CMake uses deprecated patterns**: `include_directories()`, `link_directories()`, `add_definitions()` instead of per-target equivalents. References a non-existent `cmake/` subdirectory.
4. ~~**No lit/FileCheck infrastructure**~~: **Resolved.** Lit tests exist under `test/`, run via `ninja check-aurora`.
5. **GoogleTest unit tests are broken**: `test/unit/` tests reference non-existent ODS attributes and macros. They do not compile.
6. **`docs/aurora_compiler_walkthrough.md`**: A 286-line walkthrough that describes a full ONNX-to-native pipeline as if it works end-to-end. Much of what it describes does not function.

---

## Architectural Risks

- **Object files but not executables**: `scripts/aurora-to-obj.sh` chains `aurora-opt` → `mlir-translate` → `llc` to produce `.o` files. But the generated functions take `memref` descriptors as arguments and call `malloc`/`free`; a C/C++ driver is needed to set up concrete data and link the result into a runnable binary.
- ~~**Monolithic compiler driver**~~: **Mitigated.** `aurora-opt` now provides a standard `MlirOptMain`-based pass driver. `aurora-compile` still exists for richer CLI output.
- ~~**Pass registration is manual**~~: **Resolved.** Passes are registered via `Passes.td` and `GEN_PASS_REGISTRATION`.

---

## Stabilization Prerequisites

Before adding new features, the following should be completed:

1. Fix the ODS definitions so all C++ code compiles against the actual TableGen output (ConvOp attributes).
2. ~~Set up lit/FileCheck and write tests for the MatMulBias fusion pass.~~ **Done.**
3. ~~Rewrite the CI workflow to build the project and run `check-aurora`.~~ **Done.**
4. Remove or clearly quarantine dead code (backend stubs, JIT executor, fake ONNX/PyTorch import paths).

---

## Recommended Near-Term Priorities

1. **P1**: Write a small C driver that links against the generated `.o` to produce a runnable executable demo.
2. **P1**: Fix `AuroraOps.td` / C++ alignment (ConvOp attributes).
3. **P2**: Fix the ONNX-to-Aurora emitter to produce valid IR.
4. **P2**: Add canonicalization patterns for the Aurora dialect.

*Completed*: `aurora.bias_add` op, lit/FileCheck tests, `aurora-opt` driver, TableGen pass registration, Aurora→Linalg lowering (relu, add, matmul, bias_add, matmul_bias), one-shot bufferization, full pipeline to LLVM dialect, `aurora-to-obj.sh` wrapper for object file generation, CI workflow (LLVM 17, `check-aurora`, wrapper smoke test).

---

## Deferred / Stretch

- PyTorch model import
- GPU/accelerator backend targeting
- JIT compilation
- Full ONNX model support (beyond MatMul/Add/Relu)
- Performance benchmarking against reference implementations
- DRR/PDLL declarative rewrite migration
