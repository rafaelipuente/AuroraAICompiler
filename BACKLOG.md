# Modernization Backlog

Practical GitHub-issue-ready backlog for the AuroraAICompiler revival. Each issue is scoped to be completable in 1–3 focused sessions. Issues are ordered within each group; cross-group dependencies are noted.

---

## Foundation Cleanup

### FC-1: Remove fake model files and dead artifacts from repo root

**Priority**: P0

**Why it matters**: Three `.onnx` files in `models/` are HTML pages (279 KB each, start with `<!DOCTYPE html>`). Two `.mlir` files in the root use `"dummy.constant"` ops that cannot be parsed. Several root-level test files belong under `test/`. A reviewer cloning the repo and running `file models/*.onnx` will immediately see the project is broken.

**Files/directories involved**:
- `models/resnet18.onnx` — delete (HTML, not ONNX)
- `models/mobilenetv2.onnx` — delete (HTML, not ONNX)
- `models/mnist.onnx` — delete (HTML, not ONNX)
- `matmul_test_output.mlir` — delete (uses `dummy.constant`)
- `models/matmul_test.mlir` — delete (uses `dummy.constant`)
- `test_model.mlir` — move to `test/lit/`
- ~~`test_matmul_bias.mlir`~~ — **deleted** (duplicate of `test_before_fusion.mlir`; demo now lives in `examples/matmul_bias_fusion.mlir`)
- `test_before_fusion.mlir` — move to `test/lit/` (fix shape issue first, see DC-1)
- `test_after_fusion.mlir` — move to `test/lit/`
- `standalone_matmul_bias_example.cpp` — move to `examples/`

**Acceptance criteria**:
- [ ] No `.onnx` files in `models/` that fail `python -c "import onnx; onnx.load('models/X.onnx')"`
- [ ] No `.mlir` files in repo root
- [ ] No files referencing `dummy.constant` or `dummy.placeholder` ops
- [ ] `models/matmul_test.onnx` (320 bytes, real) remains
- [ ] `models/README.md` updated to reflect what's actually present

---

### FC-2: Remove empty placeholder directories and dead source files

**Priority**: P0

**Why it matters**: Empty `src/Parser/`, `src/Codegen/`, `test/end_to_end/`, `test/integration/` directories and their comment-only CMakeLists files make the repo look scaffolded rather than real. The fake backend (`aurora-compile-backend.cpp`) that writes literal strings to files and the disabled JIT executor add dead weight.

**Files/directories involved**:
- `src/Parser/CMakeLists.txt` — delete (contains only a comment)
- `src/Codegen/CMakeLists.txt` — delete (contains only a comment)
- `src/Parser/` — delete directory
- `src/Codegen/` — delete directory
- `test/end_to_end/CMakeLists.txt` — delete
- `test/integration/CMakeLists.txt` — delete
- `test/end_to_end/` — delete directory
- `test/integration/` — delete directory
- `tools/aurora-compile/aurora-compile-backend.cpp` — delete
- `tools/aurora-compile/aurora-compile-backend.h` — delete
- `tools/aurora-jit/` — delete entire directory (disabled, broken, wrong APIs)
- `tools/standalone-fusion-demo.cpp` — delete (fake demo, doesn't use real passes)
- `python/aurora/compiler.py` — delete (writes `"AURORA_COMPILED_MODEL"` string, not a compiler)
- `python/aurora/runtime/model.py` — delete (simulated execution)
- `python/aurora/runtime/__init__.py` — delete

**Acceptance criteria**:
- [ ] `src/CMakeLists.txt` no longer references `Parser` or `Codegen` subdirectories
- [ ] `test/CMakeLists.txt` no longer references `end_to_end` or `integration`
- [ ] `tools/CMakeLists.txt` no longer references `aurora-jit`
- [ ] `aurora-compile.cpp` compiles without `aurora-compile-backend.h` (remove the backend codegen path or replace with `--emit-mlir`-only output)
- [ ] No Python file whose primary output is a hardcoded string

---

### FC-3: Remove dead code paths from aurora-compile driver

**Priority**: P1

**Why it matters**: `aurora-compile.cpp` has ONNX and PyTorch import branches (lines 527–549) that create empty modules and log "not implemented". The backend dispatch (lines 738–757) calls into deleted `aurora-compile-backend.cpp`. These paths give a false impression of functionality and will cause link errors after FC-2.

**Files involved**:
- `tools/aurora-compile/aurora-compile.cpp`
- `tools/aurora-compile/CMakeLists.txt`

**Acceptance criteria**:
- [ ] `aurora-compile` only accepts `.mlir` input; other formats return a clear error ("ONNX import not yet implemented, use python/aurora/model_import/onnx_loader.py")
- [ ] `--input-format` flag removed or restricted to `mlir`
- [ ] `--target` flag and backend dispatch removed
- [ ] `--emit-mlir` becomes the default (and only) output mode
- [ ] The binary compiles and links cleanly with no references to deleted backend files

---

## Build / Tooling Modernization

### BT-1: Fix root CMakeLists.txt and get a clean ninja build

**Priority**: P0  
**Depends on**: FC-2

**Why it matters**: The root `CMakeLists.txt` references a `cmake/` directory that doesn't exist, uses deprecated global `include_directories()` / `link_directories()` / `add_definitions()`, and the `docs` subdirectory `add_subdirectory` will fail without Doxygen. After FC-2, several `add_subdirectory` calls will point at deleted directories.

**Files involved**:
- `CMakeLists.txt` (root)
- `src/CMakeLists.txt`
- `tools/CMakeLists.txt`
- `test/CMakeLists.txt`

**Acceptance criteria**:
- [ ] `cmake/` removed from `CMAKE_MODULE_PATH` (directory doesn't exist)
- [ ] `include_directories()` replaced with `target_include_directories()` on each library/executable
- [ ] `link_directories()` replaced with `target_link_libraries()` (already mostly correct)
- [ ] `add_definitions(${LLVM_DEFINITIONS})` replaced with `target_compile_definitions()`
- [ ] `add_subdirectory(docs)` guarded properly or removed
- [ ] All references to deleted subdirectories removed
- [ ] `cmake -G Ninja .. && ninja` completes with zero errors on a system with LLVM/MLIR installed

---

### BT-2: Rewrite CI workflow to build and smoke-test real components

**Priority**: P0  
**Depends on**: BT-1

**Why it matters**: The current `.github/workflows/build.yml` references four targets that don't exist (`check-aurora-dialect`, `check-aurora-transforms`, `aurora-tests`, `run_tests.py`). CI has never passed. A broken CI badge is the fastest way to signal an abandoned project.

**Files involved**:
- `.github/workflows/build.yml`

**Acceptance criteria**:
- [ ] Uses `actions/checkout@v4` (not v3)
- [ ] Installs LLVM/MLIR 18 (or latest available in Ubuntu packages)
- [ ] `ninja` build completes without error
- [ ] Smoke test: `aurora-compile` parses `test/lit/test_model.mlir`, emits output, exits 0
- [ ] Smoke test: `aurora-compile` with `--fuse-matmul-bias` on a fusion test case, exits 0
- [ ] No references to non-existent targets or binaries
- [ ] CI passes on push to master

---

### BT-3: Add .clang-format and pyproject.toml

**Priority**: P1

**Why it matters**: No code style enforcement exists. The C++ code mixes naming conventions and formatting. The Python package uses `setup.py` with a fake author email (`info@auroraai.example.com`) and a broken `open("../README.md")` call. `torch` is listed as a hard dependency even though it's only needed for the optional PyTorch importer.

**Files involved**:
- `.clang-format` — new file (LLVM style, standard for MLIR projects)
- `python/pyproject.toml` — new file, replacing `python/setup.py`
- `python/setup.py` — delete

**Acceptance criteria**:
- [ ] `.clang-format` uses LLVM's BasedOnStyle
- [ ] `pyproject.toml` lists only `numpy` and `onnx` as required dependencies; `torch` and `pytest` are optional extras
- [ ] Author/email/URL fields are accurate
- [ ] `pip install -e python/` succeeds

---

## Dialect and IR Cleanup

### ~~DC-1: Fix AddOp to support broadcasting~~ **RESOLVED**

**Resolution**: Introduced `aurora.bias_add` as a separate op with a custom verifier (rank-1 bias, last-dim match, result type = input type). `aurora.add` remains strict element-wise with `SameOperandsAndResultType`. The fusion pass now matches `matmul` + `bias_add` -> `matmul_bias`. All test files updated. Demo in `examples/matmul_bias_fusion.mlir`.
- [ ] MatMulBias fusion pass still works on the corrected test case
- [ ] All `.mlir` test files under `test/lit/` parse without error

---

### DC-2: Add Conv attributes or strip downstream code that references them

**Priority**: P0

**Why it matters**: `ConvOp` in `AuroraOps.td` has only `input` and `filter` operands. But `AuroraToLLVM.cpp` calls `op.getStrides()`, `op.getPaddings()`, `op.getDilations()`, `op.getGroups()`, and `TestAuroraDialect.cpp` passes those as builder arguments. This mismatch makes both files uncompilable. Must choose one direction and commit.

**Files involved**:
- `include/Aurora/Dialect/Aurora/AuroraOps.td` — add attributes to `ConvOp`, OR:
- `src/Conversion/AuroraToLLVM/AuroraToLLVM.cpp` — remove `ConvOpLowering` entirely
- `test/unit/dialect/TestAuroraDialect.cpp` — fix `CreateConvOp` test

**Recommended approach**: Add `strides`, `paddings`, `dilations` as `OptionalAttr<I64ArrayAttr>` and `groups` as `DefaultValuedOptionalAttr<I64Attr, "1">` to `ConvOp`. This is the standard representation and makes the test and lowering code valid.

**Acceptance criteria**:
- [ ] `ConvOp` ODS definition and all C++ usage of `ConvOp` are consistent
- [ ] `TestAuroraDialect.cpp` `CreateConvOp` test compiles
- [ ] `AuroraToLLVM.cpp` `ConvOpLowering` compiles (may still not be functionally correct — that's LC-1)

---

### DC-3: Remove or replace the broken ConvRelu fusion pass

**Priority**: P1

**Why it matters**: `Fusion.cpp` contains `ConvReluFusionPattern` which replaces Relu with the Conv result — this silently drops the activation and produces incorrect IR. No `aurora.conv_relu` fused op exists. The pass runs by default at `-O1` in `aurora-compile`, meaning every user gets incorrect output without knowing it.

**Files involved**:
- `src/Transforms/Fusion.cpp`
- `include/Aurora/Transforms/Fusion.h`
- `tools/aurora-compile/aurora-compile.cpp` (line 594: `pm.addPass(createFusionPass())`)

**Option A**: Define `aurora.conv_relu` in ODS, rewrite the pattern to emit it. Full fix but more work.  
**Option B**: Remove the pass entirely and remove the `-O1` integration in `aurora-compile`. Honest and fast.

**Acceptance criteria**:
- [ ] No pass produces incorrect IR (either the pass is removed or it emits a correct fused op)
- [ ] `aurora-compile` at default settings does not silently drop operations

---

### DC-4: Fix AuroraDialectRegistration.cpp

**Priority**: P1

**Why it matters**: Uses `#define GET_TYPEDEF_CLASSES` (for type definitions) with the ops include file. The dialect has no custom types, so this is dead code at best and a source of subtle ODR/symbol issues at worst. It also re-includes `AuroraOps.cpp.inc` which is already included in `AuroraOps.cpp`.

**Files involved**:
- `src/Dialect/Aurora/AuroraDialectRegistration.cpp`

**Acceptance criteria**:
- [ ] File either serves a clear purpose (e.g., standalone registration entry point for tools) or is deleted
- [ ] No duplicate includes of `.cpp.inc` files across the `src/Dialect/Aurora/` directory
- [ ] Build produces no warnings about duplicate symbols

---

## Lowering / Conversion

### LC-1: Rewrite AuroraToLLVM for MatMul, MatMulBias, Relu, Add only

**Priority**: P1  
**Depends on**: DC-1, DC-2

**Why it matters**: The current `AuroraToLLVM.cpp` is uncompilable (see DC-2) and architecturally flawed (no bufferization, deprecated APIs, missing lowering patterns for half the dialect). A stripped-down version that lowers just the four ops used in the working fusion pipeline would demonstrate the full Aurora IR → LLVM IR path — the single most important missing capability.

**Files involved**:
- `src/Conversion/AuroraToLLVM/AuroraToLLVM.cpp` — rewrite
- `src/Conversion/CMakeLists.txt` — wire the library into the build
- `include/Aurora/Conversion/AuroraToLLVM/` — update header if needed

**Acceptance criteria**:
- [ ] Lowering patterns exist for `MatMulOp`, `MatMulBiasOp`, `ReluOp`, `AddOp`
- [ ] Pipeline includes a bufferization step (tensor → memref) before LLVM conversion
- [ ] Does not reference `populateAffineToStdConversionPatterns` or `populateStdToLLVMConversionPatterns`
- [ ] `aurora-compile test_model.mlir -o output.ll --emit-llvm` produces valid LLVM IR (new `--emit-llvm` flag)
- [ ] Conv/LayerNorm/FusedAttention lowerings are NOT included (out of scope until those ops are stable)
- [ ] A lit test verifies the lowering output

---

## Testing

### T-1: Add lit/FileCheck test infrastructure

**Priority**: P0  
**Depends on**: BT-1, FC-1

**Why it matters**: The project has zero automated tests that actually run. MLIR projects universally use lit + FileCheck. Without this, every change risks silent regressions, and the project cannot credibly claim any functionality works.

**Files involved**:
- `test/lit.cfg.py` — new file
- `test/lit.site.cfg.py.in` — new file (configured by CMake)
- `test/CMakeLists.txt` — add `configure_file` for lit site config and `add_lit_testsuite`
- `CMakeLists.txt` (root) — pass `LLVM_EXTERNAL_LIT` to test config
- `test/lit/parse_aurora_ops.mlir` — new test: parse each Aurora op, verify roundtrip
- `test/lit/matmul_bias_fusion.mlir` — new test: verify MatMulBias fusion transforms IR correctly

**Acceptance criteria**:
- [ ] `ninja check-aurora` runs lit tests and reports pass/fail
- [ ] At least two passing lit tests:
  1. Parse a `.mlir` file with all 7 Aurora ops, verify it roundtrips through `aurora-compile --emit-mlir`
  2. Run `--fuse-matmul-bias` on an input with `matmul` + `add`, FileCheck for `aurora.matmul_bias` in output
- [ ] CI runs `ninja check-aurora` and gates on it

---

### T-2: Fix and wire GoogleTest unit tests

**Priority**: P1  
**Depends on**: DC-1, DC-2

**Why it matters**: `TestAuroraDialect.cpp` and `TestMatMulBiasFusion.cpp` exist but don't compile due to ODS mismatches and undefined macros. The CMakeLists files create placeholder targets that print comments instead of building tests. Fixing these gives real unit test coverage of dialect construction and pass behavior.

**Files involved**:
- `test/unit/dialect/TestAuroraDialect.cpp` — fix ConvOp test to match ODS, register Arith dialect
- `test/unit/transforms/TestMatMulBiasFusion.cpp` — replace `ASSERT_SUCCESS` with `ASSERT_TRUE(succeeded(...))`, remove `transpose_lhs`/`transpose_rhs` references, fix AddOp shape mismatch in test IR
- `test/unit/dialect/CMakeLists.txt` — replace placeholder `add_custom_target` with `add_executable` + `target_link_libraries` + `gtest_main`
- `test/unit/transforms/CMakeLists.txt` — same

**Acceptance criteria**:
- [ ] `ninja AuroraDialectTests` builds and produces an executable
- [ ] `ninja AuroraTransformsTests` builds and produces an executable
- [ ] `CreateMatMulOp`, `CreateReluOp` tests pass
- [ ] `CreateConvOp` test passes (after DC-2 fix)
- [ ] `BasicFusion` and `NoFusionWithMultipleUses` tests pass (after DC-1 fix)
- [ ] `FusionWithTranspose` test is either fixed (if transpose attrs are added to ODS) or removed
- [ ] CI runs both test binaries

---

## Rewrites / Canonicalization

### RC-1: Add canonicalization patterns for Aurora ops

**Priority**: P2  
**Depends on**: T-1

**Why it matters**: None of the Aurora ops define `hasCanonicalizer = 1` or `hasFolder = 1`. Basic canonicalization patterns (e.g., `add(x, 0) → x`, `relu(relu(x)) → relu(x)`, `matmul` with identity) would demonstrate MLIR best practices, improve the optimization pipeline, and provide additional test surface.

**Files involved**:
- `include/Aurora/Dialect/Aurora/AuroraOps.td` — add `hasCanonicalizer = 1` to target ops
- `src/Dialect/Aurora/AuroraOps.cpp` — implement `getCanonicalizationPatterns` for each op
- `test/lit/canonicalize_aurora.mlir` — new lit test

**Acceptance criteria**:
- [ ] At least two canonicalization patterns implemented (e.g., `relu(relu(x))` folding, identity matmul)
- [ ] Patterns fire when `--mlir-pass-pipeline="builtin.module(canonicalize)"` is run
- [ ] Lit test verifies the patterns

---

## Documentation / Demo Quality

### DD-1: Fix CONTRIBUTING.md to match real build/test workflow

**Priority**: P1  
**Depends on**: BT-1, T-1

**Why it matters**: `CONTRIBUTING.md` tells contributors to run `make check-aurora` — a target that doesn't exist. The build instructions don't mention the required `MLIR_DIR`/`LLVM_DIR` CMake variables. Contributors who follow these instructions will fail immediately.

**Files involved**:
- `CONTRIBUTING.md`

**Acceptance criteria**:
- [ ] Build instructions include explicit `cmake` invocation with `-DMLIR_DIR` and `-DLLVM_DIR`
- [ ] Test instructions reference real targets (`ninja check-aurora` after T-1)
- [ ] No references to `make check-aurora` or any non-existent target
- [ ] Code style section references `.clang-format` (after BT-3)

---

### DD-2: Sync docs/dialect.md and docs/aurora_compiler_walkthrough.md with reality

**Priority**: P2  
**Depends on**: DC-1, DC-2

**Why it matters**: `docs/dialect.md` describes ops with attributes that don't exist in ODS. `docs/aurora_compiler_walkthrough.md` describes ONNX import, runtime execution, and benchmarking as if they work. A recruiter or contributor who reads these and then looks at the code will lose trust.

**Files involved**:
- `docs/dialect.md`
- `docs/aurora_compiler_walkthrough.md`

**Acceptance criteria**:
- [ ] `dialect.md` op signatures match `AuroraOps.td` exactly
- [ ] `aurora_compiler_walkthrough.md` has a clear "Status" annotation on each section: Working / In Progress / Planned
- [ ] No section describes a capability that does not exist without a disclaimer

---

### DD-3: Create a reproducible demo script

**Priority**: P2  
**Depends on**: BT-1, T-1, FC-1

**Why it matters**: There is no single command a reviewer can run to see the project do something real. A shell script that builds, runs the compiler on a test case, shows before/after IR, and generates a Mermaid diagram would make the project immediately tangible.

**Files involved**:
- `examples/demo.sh` — new file
- `examples/matmul_fusion_input.mlir` — new file (clean test case)

**Acceptance criteria**:
- [ ] `./examples/demo.sh` completes in under 5 minutes on a machine with LLVM installed
- [ ] Output shows: (1) input IR, (2) `aurora-compile` invocation, (3) output IR with `matmul_bias` op, (4) Mermaid diagram file
- [ ] Script exits non-zero if any step fails
- [ ] Works after a fresh `git clone && mkdir build && cd build && cmake .. && ninja`

---

## Nice-to-Have Future Work

These should not be started until all P0 and P1 issues above are closed.

### FW-1: Fix ONNX → Aurora IR emitter to produce valid dialect ops

**Priority**: P2

**Why it matters**: `onnx_to_mlir.py` (604 lines) has real structure — op dispatch table, attention pattern detection, shape propagation — but falls back to `"dummy.constant"` and `"dummy.placeholder"` ops. Fixing this to emit real Aurora ops for at least `MatMul`, `Add`, `Relu`, and `Constant` would close the ONNX → Aurora IR gap.

**Files involved**:
- `python/aurora/model_import/onnx_to_mlir.py` — rewrite `_convert_*` methods and constant emission
- `python/aurora/model_import/onnx_loader.py` — no changes needed (working)
- Fix relative import: `from onnx_loader import` → `from .onnx_loader import`

**Acceptance criteria**:
- [ ] `ONNXToMLIRConverter.convert()` on a MatMul+Add ONNX model produces `.mlir` that `aurora-compile` can parse
- [ ] No `dummy.constant` or `dummy.placeholder` ops in output
- [ ] Integration test: `create_simple_model.py` → `onnx_to_mlir.py` → `aurora-compile` → success

---

### FW-2: Implement aurora.conv_relu fused op and pattern

**Priority**: P2  
**Depends on**: DC-2, DC-3

**Why it matters**: Conv+Relu fusion is the most common optimization in inference compilers. Having a correct implementation alongside MatMulBias fusion would demonstrate the project handles multiple fusion patterns.

**Files involved**:
- `include/Aurora/Dialect/Aurora/AuroraOps.td` — add `Aurora_ConvReluOp`
- `src/Transforms/Fusion.cpp` — rewrite `ConvReluFusionPattern` to emit the new op
- `test/lit/conv_relu_fusion.mlir` — new lit test

**Acceptance criteria**:
- [ ] `aurora.conv_relu` op defined in ODS with same operands as `ConvOp` plus result type
- [ ] Fusion pass rewrites `conv` → `relu` into `conv_relu`
- [ ] Lit test verifies the transformation

---

### FW-3: Implement runtime function stubs for LLVM lowering

**Priority**: P2  
**Depends on**: LC-1

**Why it matters**: The LLVM lowering emits `call @aurora_runtime_matmul(...)` etc., but these symbols are never defined. Providing a small C library with naive implementations would make the lowered code linkable and executable — closing the full pipeline.

**Files involved**:
- `src/Runtime/aurora_runtime_kernels.c` — new file (naive matmul, relu, add implementations)
- `src/Runtime/CMakeLists.txt` — build the C library
- `tools/aurora-run/` — could eventually link against it

**Acceptance criteria**:
- [ ] `aurora_runtime_matmul`, `aurora_runtime_relu`, `aurora_runtime_add` are implemented in C
- [ ] The LLVM IR produced by LC-1 can be compiled with `clang` and linked against this library
- [ ] A test demonstrates: Aurora IR → LLVM IR → compile → link → run → correct output

---

## Dependency Graph

```
FC-1 ──┐
FC-2 ──┼──▶ BT-1 ──▶ BT-2 ──▶ (CI green)
       │      │
       │      ▼
       │    T-1 ──▶ DD-3
       │      │
DC-1 ──┤      │
DC-2 ──┤      ▼
       ├──▶ T-2
       │
       ▼
     LC-1 ──▶ FW-3
       │
DC-3 ──▶ FW-2

FC-3 ──▶ (independent, do after FC-2)
BT-3 ──▶ (independent, do anytime)
DD-1 ──▶ (after BT-1 + T-1)
DD-2 ──▶ (after DC-1 + DC-2)
RC-1 ──▶ (after T-1)
FW-1 ──▶ (after DC-1, independent of C++ work)
```

---

## Summary by Priority

| Priority | Count | Issues |
|---|---|---|
| **P0** | 6 | FC-1, FC-2, BT-1, BT-2, DC-1, DC-2, T-1 |
| **P1** | 6 | FC-3, BT-3, DC-3, DC-4, LC-1, T-2, DD-1 |
| **P2** | 5 | RC-1, DD-2, DD-3, FW-1, FW-2, FW-3 |

P0 issues establish a compilable, honest, testable baseline. P1 issues build the first real end-to-end pipeline. P2 issues expand coverage and make the project portfolio-grade.
