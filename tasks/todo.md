# AuroraAICompiler Revival — Audit & Plan

*Generated: 2026-03-13 — Senior Compiler Engineer Repo Audit*

---

## 1. Blunt Repo Assessment

### Overview

The AuroraAICompiler repository was created in a single day (May 18, 2025) across 2 commits. It contains ~7,500 lines of non-trivial code (excluding fake `.onnx` files). The project has a **real MLIR dialect** and **one working optimization pass** (MatMulBiasFusion), surrounded by a large amount of **scaffolding, stubs, and misleading claims**. Nothing in the repo has ever been built by CI — the GitHub Actions workflow references targets that don't exist.

### What Is Actually Real

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| Aurora MLIR Dialect (TableGen + C++) | **REAL** | ~330 | Defines MatMul, MatMulBias, Conv, Relu, Add, LayerNorm, FusedAttention ops |
| MatMulBiasFusion pass | **REAL** | ~225 | Working pattern: MatMul + Add → MatMulBias. Proper MLIR rewrite pattern. |
| Fusion pass skeleton | **PARTIAL** | ~60 | ConvRelu "fusion" that just replaces relu result — no fused op exists |
| aurora-compile tool (MLIR path only) | **REAL** | ~800 | Parses .mlir, registers dialect, runs passes, dumps Mermaid. ONNX/PyTorch paths are stubs. |
| ONNXLoader (Python) | **REAL** | ~465 | Loads ONNX protobuf, builds graph representation |
| AuroraTensor / ExecutionContext (C++) | **REAL** | ~200 | Working tensor container and execution context |
| standalone_matmul_bias_example.cpp | **REAL** | ~288 | Pure C++ demo, no MLIR dependency |

### What Is Broken / Will Not Compile

| Component | Issue |
|-----------|-------|
| `AuroraToLLVM.cpp` | References `op.getStrides()`, `op.getPaddings()`, etc. on ConvOp — these attributes don't exist in TableGen. **Will not compile.** |
| `TestAuroraDialect.cpp` | Uses ConvOp constructor with 4 extra attributes not in ODS. **Will not compile.** |
| `TestMatMulBiasFusion.cpp` | Uses undefined `ASSERT_SUCCESS` macro and `transpose_lhs`/`transpose_rhs` attributes that don't exist. **Will not compile.** |
| `jit_executor.cpp` | Adds module-level pass as nested FuncOp pass. Verification disabled with `if (false)`. Wrong dialect set. **Disabled in build, broken anyway.** |
| `onnx_to_mlir.py` | Emits `"dummy.constant"` and `"dummy.placeholder"` ops — not real MLIR. Relative imports broken. |
| ~~`test_matmul_bias.mlir`~~ | **Deleted** (duplicate; `aurora.bias_add` now used everywhere) |
| `test_before_fusion.mlir` | Fixed: uses `aurora.bias_add` with correct shapes |
| `models/matmul_test.mlir`, `matmul_test_output.mlir` | Use `"dummy.constant"` — not parseable |
| CI workflow (`build.yml`) | References `check-aurora-dialect`, `check-aurora-transforms`, `aurora-tests`, `run_tests.py` — **none of these exist** |

### What Is Placeholder / Fake

| Component | Reality |
|-----------|---------|
| `resnet18.onnx`, `mobilenetv2.onnx`, `mnist.onnx` | **HTML pages, not ONNX models** (279 KB each, starts with `<!DOCTYPE html>`) |
| `AuroraRuntime.cpp` — model load/execute | Always returns dummy model. `executeModel` copies input to output with hardcoded 10.5ms timing. |
| `aurora-compile-backend.cpp` | All 3 backends (CPU/GPU/Ampere) write a literal string to file. No codegen. |
| `aurora-run.cpp` | `readInputData` ignores filename, returns dummy 1x3x224x224 tensor. |
| `aurora-benchmark.cpp` | `BaselineModel` simulates execution. Benchmark numbers are fabricated. |
| `python/aurora/compiler.py` | Writes `AURORA_COMPILED_MODEL` string to file. Does not invoke compiler. |
| `python/aurora/runtime/model.py` | Simulated execution with dummy metadata. |
| `python/aurora/model_import/onnx_importer.py` | `convert_to_aurora_ir()` logs a message but does not emit IR. |
| `python/aurora/model_import/pytorch_importer.py` | `convert_to_aurora_ir()` only logs; no IR emission. |
| README benchmarks table | "ResNet18: 122.4ms → 88.1ms, 1.39× speedup" — **entirely fabricated** |
| README "build-passing" badge | Static badge image, not linked to CI. CI has never passed. |
| README example commands | `--input=` and `--model=` flags don't match actual CLI (`positional arg` + `-o`) |

### Architectural Red Flags

1. **No `cmake/` directory** — referenced in `CMAKE_MODULE_PATH` but doesn't exist.
2. **No lit test infrastructure** — tests are C++ files pretending to be unit tests but have no runner, no lit config, no FileCheck.
3. **`src/Parser/` and `src/Codegen/`** — empty directories with placeholder CMakeLists.
4. **`test/end_to_end/` and `test/integration/`** — empty placeholder directories.
5. **Global `include_directories()` / `link_directories()`** — deprecated CMake pattern; should be per-target.
6. **No `.clang-format`**, no pre-commit hooks, no code style enforcement.
7. **MLIR version pinned to 16** — MLIR 19/20 is current in 2026; MLIR 16 is from mid-2023.
8. **No `requirements.txt` or `pyproject.toml`** for Python dependencies.
9. **Test MLIR files scattered in repo root** — `test_model.mlir`, `test_before_fusion.mlir`, etc. should be under `test/`.

---

## 2. Keep / Fix / Remove / Postpone Table

| File/Component | Action | Rationale |
|---|---|---|
| **DIALECT** | | |
| `include/Aurora/Dialect/Aurora/AuroraDialect.td` | **KEEP** | Core dialect definition, real TableGen |
| `include/Aurora/Dialect/Aurora/AuroraOps.td` | **FIX** | Add missing attributes (strides, padding on Conv), add transpose flags on MatMul, fix shape semantics on Add |
| `include/Aurora/Dialect/Aurora/AuroraDialect.h` | **KEEP** | Standard |
| `include/Aurora/Dialect/Aurora/AuroraOps.h` | **KEEP** | Standard |
| `src/Dialect/Aurora/AuroraDialect.cpp` | **KEEP** | Real implementation |
| `src/Dialect/Aurora/AuroraOps.cpp` | **KEEP** | Standard pattern |
| `src/Dialect/Aurora/AuroraDialectRegistration.cpp` | **FIX** | Incorrect `GET_TYPEDEF_CLASSES` usage; clarify or remove |
| **TRANSFORMS** | | |
| `include/Aurora/Transforms/MatMulBiasFusion.h` | **KEEP** | Real pass declaration |
| `include/Aurora/Transforms/Fusion.h` | **FIX** | ConvRelu fusion is a stub; either implement or remove |
| `include/Aurora/Transforms/Passes.h` | **FIX** | Incomplete pass registration |
| `src/Transforms/MatMulBiasFusion.cpp` | **KEEP** | Real, working pass |
| `src/Transforms/Fusion.cpp` | **FIX** | Stub fusion logic; make real or remove |
| **CONVERSION** | | |
| `src/Conversion/AuroraToLLVM/AuroraToLLVM.cpp` | **FIX (major)** | Broken: attributes don't match ODS, missing lowerings. Needs full rewrite to match actual ops. |
| `include/Aurora/Conversion/` (CMake files) | **FIX** | Empty placeholders |
| **RUNTIME** | | |
| `include/Aurora/Runtime/AuroraRuntime.h` | **FIX** | Remove `AuroraRuntime` class or implement it; keep Tensor/Context/Model |
| `src/Runtime/AuroraRuntime.cpp` | **FIX** | Remove all dummy logic. Make `loadFromFile` actually fail if file doesn't exist. |
| **TOOLS** | | |
| `tools/aurora-compile/aurora-compile.cpp` | **KEEP + FIX** | Real MLIR driver; remove fake ONNX/PyTorch import stubs or implement them |
| `tools/aurora-compile/aurora-compile-backend.cpp` | **REMOVE** | Pure placeholder; contributes nothing |
| `tools/aurora-compile/aurora-compile-backend.h` | **REMOVE** | Goes with above |
| `tools/aurora-run/aurora-run.cpp` | **POSTPONE** | Can't work until runtime is real. Keep skeleton but mark as WIP. |
| `tools/aurora-jit/jit_executor.cpp` | **REMOVE** | Broken, disabled, wrong APIs. Rewrite from scratch later. |
| `tools/aurora-jit/sample_inputs.json` | **REMOVE** | Goes with above |
| `tools/aurora-benchmark/aurora-benchmark.cpp` | **POSTPONE** | Requires working runtime. Keep skeleton, mark WIP. |
| `tools/standalone-fusion-demo.cpp` | **REMOVE** | Fake demo that doesn't use Aurora dialect or real passes |
| `standalone_matmul_bias_example.cpp` | **REMOVE from root** | Move to `examples/` if kept, but it's pure C++, not MLIR-related |
| **TESTS** | | |
| `test/unit/dialect/TestAuroraDialect.cpp` | **FIX** | Align with actual ODS definitions |
| `test/unit/transforms/TestMatMulBiasFusion.cpp` | **FIX** | Replace undefined macros, align with actual ops |
| `test_model.mlir` | **MOVE** to `test/lit/` | Valid MLIR; needs proper test infrastructure |
| ~~`test_matmul_bias.mlir`~~ | **DELETED** | Duplicate; demo consolidated in `examples/matmul_bias_fusion.mlir` |
| `test_before_fusion.mlir` | **FIX + MOVE** | Fix shape mismatch, move to `test/lit/` |
| `test_after_fusion.mlir` | **MOVE** to `test/lit/` | Valid |
| `matmul_test_output.mlir` | **REMOVE** | Uses `dummy.constant`, not valid |
| `models/matmul_test.mlir` | **REMOVE** | Uses `dummy.constant`, not valid |
| `test/end_to_end/` | **REMOVE** | Empty placeholder |
| `test/integration/` | **REMOVE** | Empty placeholder |
| **PYTHON** | | |
| `python/aurora/__init__.py` | **KEEP** | Works |
| `python/aurora/compiler.py` | **REMOVE** | Pure placeholder |
| `python/aurora/runtime/model.py` | **REMOVE** | Pure placeholder |
| `python/aurora/runtime/__init__.py` | **REMOVE** | Goes with above |
| `python/aurora/model_import/onnx_loader.py` | **KEEP** | Real ONNX parsing |
| `python/aurora/model_import/onnx_to_mlir.py` | **FIX (major)** | Must emit real Aurora dialect ops, not `dummy.constant` |
| `python/aurora/model_import/onnx_importer.py` | **FIX** | `convert_to_aurora_ir()` needs real implementation |
| `python/aurora/model_import/pytorch_importer.py` | **POSTPONE** | Keep skeleton, mark as WIP; ONNX path is priority |
| `python/aurora/model_import/example_usage.py` | **FIX** | Fix imports |
| `python/setup.py` | **FIX** | Replace with `pyproject.toml` |
| **MODELS** | | |
| `models/resnet18.onnx` | **REMOVE** | HTML file, not ONNX model |
| `models/mobilenetv2.onnx` | **REMOVE** | HTML file, not ONNX model |
| `models/mnist.onnx` | **REMOVE** | HTML file, not ONNX model |
| `models/matmul_test.onnx` | **KEEP** | Tiny real ONNX model (320 bytes) |
| `models/create_test_model.py` | **KEEP** | Real model generation script |
| `models/create_simple_model.py` | **KEEP** | Real model generation script |
| `models/simple_benchmark.py` | **POSTPONE** | Needs working compiler first |
| `models/standalone_test.py` | **POSTPONE** | Needs working compiler first |
| `scripts/download_models.py` | **FIX** | SHA256 hashes are placeholders |
| **BUILD / CI** | | |
| `CMakeLists.txt` (root) | **FIX** | Remove `cmake/` path ref, modernize to per-target includes |
| `.github/workflows/build.yml` | **FIX (major)** | All test targets are fake. Must match real build. Update to LLVM 18/19. |
| `src/Parser/CMakeLists.txt` | **REMOVE** | Empty |
| `src/Codegen/CMakeLists.txt` | **REMOVE** | Empty |
| `benchmarks/CMakeLists.txt` | **KEEP** | Minimal but functional |
| **DOCS** | | |
| `README.md` | **FIX (major)** | Remove fabricated benchmarks, fix CLI examples, add honest status badges, mark what's WIP |
| `CONTRIBUTING.md` | **FIX** | References `make check-aurora` which doesn't exist |
| `docs/dialect.md` | **FIX** | Sync with actual ODS definitions |
| `docs/aurora_compiler_walkthrough.md` | **FIX** | Mark aspirational sections clearly |
| **MISSING FILES (need to add)** | | |
| `.clang-format` | **ADD** | Code style enforcement |
| `requirements.txt` or `pyproject.toml` | **ADD** | Python dependency management |
| `test/lit.cfg.py` + `test/lit.site.cfg.py.in` | **ADD** | Proper MLIR lit test infrastructure |
| `.gitignore` | **FIX** | Add `build/`, `*.pyc`, etc. |

---

## 3. Prioritized 30-Day Revival Plan

### Week 1: Foundation (Days 1–7) — "Make it build and be honest"

**Goal**: A clean repo that compiles, passes CI, and makes no false claims.

- [ ] **Day 1–2: Clean the repo**
  - Remove all fake `.onnx` files (HTML pages)
  - Remove dead files: `standalone-fusion-demo.cpp`, `jit_executor.cpp`, `sample_inputs.json`, `aurora-compile-backend.cpp/.h`
  - Remove dummy MLIR files: `matmul_test_output.mlir`, `models/matmul_test.mlir`
  - Move root-level test MLIR files into `test/`
  - Remove empty directories: `src/Parser/`, `src/Codegen/`, `test/end_to_end/`, `test/integration/`
  - Remove placeholder Python: `python/aurora/compiler.py`, `python/aurora/runtime/`
  - Move `standalone_matmul_bias_example.cpp` to `examples/`
  - Add `.clang-format` (LLVM style)
  - Add `pyproject.toml` with dependencies

- [ ] **Day 3–4: Fix the dialect**
  - Fix `AuroraOps.td`: add `strides`, `paddings`, `dilations`, `groups` to ConvOp; or simplify Conv to match actual C++ usage
  - Fix `AddOp`: either remove `SameOperandsAndResultType` (to allow broadcasting) or fix all test files
  - Fix `AuroraDialectRegistration.cpp`: remove bogus `GET_TYPEDEF_CLASSES`
  - Verify the dialect builds cleanly against MLIR 18 or 19 (update `find_package` version)

- [ ] **Day 5: Fix the build system**
  - Remove `cmake/` from `CMAKE_MODULE_PATH`
  - Modernize: replace global `include_directories` with per-target
  - Update LLVM version requirement to 18+
  - Fix all sub-CMakeLists that reference non-existent targets
  - Ensure `ninja` builds cleanly with zero errors

- [ ] **Day 6–7: Fix CI**
  - Rewrite `.github/workflows/build.yml` to install LLVM 18+
  - Use `actions/checkout@v4`
  - Build only what exists; run only tests that exist
  - Add a simple smoke test: parse `test_model.mlir` with `aurora-compile`
  - Get the green badge

### Week 2: Core Compiler (Days 8–14) — "End-to-end on one op"

**Goal**: Demonstrate MatMul → Aurora IR → LLVM IR for one operation, with passing tests.

- [ ] **Day 8–9: Fix AuroraToLLVM lowering**
  - Strip it down to only lower the ops that exist with correct attributes: MatMul, MatMulBias, Relu, Add
  - Remove Conv/LayerNorm/FusedAttention lowerings (they can't work yet)
  - Ensure tensor→memref conversion is in the pipeline (required before LLVM lowering)
  - Test: lower `test_model.mlir` through the full pipeline to LLVM IR

- [ ] **Day 10–11: Add lit test infrastructure**
  - Create `test/lit.cfg.py` and `test/lit.site.cfg.py.in`
  - Convert test MLIR files to FileCheck-based lit tests
  - Test the fusion pass: before → after with `// RUN: aurora-compile %s | FileCheck %s`
  - Test the lowering: Aurora IR → LLVM IR
  - Wire lit into CMake (`add_lit_testsuite`)

- [ ] **Day 12–13: Fix unit tests**
  - Fix `TestAuroraDialect.cpp` to match actual ODS
  - Fix `TestMatMulBiasFusion.cpp`: use proper assertions, remove non-existent attributes
  - Wire GoogleTest into CMake properly
  - All tests pass

- [ ] **Day 14: Checkpoint — demonstrate the pipeline**
  - Write one clean example: `examples/matmul_fusion.mlir` + documentation
  - `aurora-compile examples/matmul_fusion.mlir -o output.ll --emit-llvm`
  - Document the exact commands that work in README

### Week 3: ONNX Frontend (Days 15–21) — "Real model import"

**Goal**: Import a simple ONNX model (MatMul + Add) into Aurora IR and optimize it.

- [ ] **Day 15–16: Fix onnx_to_mlir.py**
  - Emit real Aurora dialect ops instead of `"dummy.constant"`
  - Support at least: MatMul, Add, Relu, Constant
  - Generate valid `.mlir` that `aurora-compile` can parse

- [ ] **Day 17–18: Create real test models**
  - Use `models/create_simple_model.py` to generate a real small ONNX model
  - Demonstrate: ONNX → Aurora IR → optimized Aurora IR → LLVM IR
  - Add this as a lit test / integration test

- [ ] **Day 19–20: Fix onnx_importer.py**
  - Wire `onnx_loader.py` + `onnx_to_mlir.py` together in `onnx_importer.py`
  - Create a clean CLI: `python -m aurora.model_import.onnx_importer model.onnx -o output.mlir`

- [ ] **Day 21: Wire ONNX import into aurora-compile**
  - When `aurora-compile` receives an `.onnx` file, call the Python importer (or embed logic)
  - Alternatively, document the two-step workflow

### Week 4: Polish & Portfolio (Days 22–30) — "Make it impressive"

**Goal**: Clean documentation, working demos, and recruiter-ready presentation.

- [ ] **Day 22–23: Rewrite README**
  - Honest status section: what works, what's in progress
  - Replace fabricated benchmarks with real measurements (even if modest)
  - Fix all CLI examples to match actual interface
  - Add architecture diagram that reflects reality
  - Link to real CI badge from GitHub Actions

- [ ] **Day 24–25: Add ConvOp support**
  - Add strides/padding/dilations to `AuroraOps.td`
  - Implement Conv lowering in AuroraToLLVM (even as external call)
  - Add Conv test case
  - This rounds out the dialect beyond just MatMul

- [ ] **Day 26–27: Runtime execution (stretch)**
  - Make `AuroraModel::loadFromFile` load real compiled code (via dlopen or JIT)
  - Or: implement a simple interpreter that walks Aurora IR
  - Make `aurora-run` do something real

- [ ] **Day 28–29: Documentation**
  - Rewrite `docs/aurora_compiler_walkthrough.md` to reflect actual capabilities
  - Rewrite `docs/dialect.md` to be auto-generated or perfectly in sync
  - Add a `DESIGN.md` explaining architectural decisions

- [ ] **Day 30: Final polish**
  - Ensure all CI passes
  - Clean git history (squash if desired)
  - Tag v0.2.0 release
  - Update GitHub description

---

## 4. Files to Edit First (Ordered)

The first 10 files to touch, in exact order:

1. **`README.md`** — Remove all fabricated claims immediately. This is the first thing anyone sees.
2. **`include/Aurora/Dialect/Aurora/AuroraOps.td`** — Fix ConvOp attributes, AddOp semantics. Everything downstream depends on this.
3. **`src/Conversion/AuroraToLLVM/AuroraToLLVM.cpp`** — Align with actual ODS or strip to ops that work.
4. **`.github/workflows/build.yml`** — Make CI actually build and test the real code.
5. **`CMakeLists.txt` (root)** — Remove phantom cmake/ path, update LLVM version.
6. **`test/unit/dialect/TestAuroraDialect.cpp`** — Make tests compile against actual ops.
7. **`test/unit/transforms/TestMatMulBiasFusion.cpp`** — Make tests compile against actual ops.
8. **`src/Dialect/Aurora/AuroraDialectRegistration.cpp`** — Fix wrong `GET_TYPEDEF_CLASSES`.
9. **`python/aurora/model_import/onnx_to_mlir.py`** — Emit real Aurora ops instead of `dummy.*`.
10. **`src/Transforms/Fusion.cpp`** — Either implement real ConvRelu fusion or remove it.

---

## 5. Summary Scorecard

| Dimension | Current Grade | After 30 Days (Target) |
|-----------|:---:|:---:|
| Builds cleanly | **F** (CI broken, multiple compile errors) | **A** |
| Tests pass | **F** (no working tests) | **B+** (lit + unit) |
| README honesty | **F** (fabricated benchmarks, wrong CLI) | **A** |
| Dialect quality | **B-** (real but inconsistent) | **A** |
| End-to-end pipeline | **F** (nothing works end-to-end) | **B** (MatMul path) |
| Code quality | **C** (mix of real and placeholder) | **A-** |
| ONNX import | **D** (loader works, emitter broken) | **B** |
| Runtime | **F** (pure stubs) | **C+** (stretch goal) |
| Recruiter impression | **C+** (looks good on surface) | **A** (substance to back it up) |

---

*This audit was conducted file-by-file with zero assumptions. Every verdict is based on reading the actual source code.*
