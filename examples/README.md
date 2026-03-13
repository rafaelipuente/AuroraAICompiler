# Examples

Demo inputs for AuroraAICompiler.

## matmul_bias_fusion.mlir

A two-layer linear network in Aurora IR. Contains two `matmul` + `bias_add` pairs separated by a `relu` activation.

```bash
aurora-opt examples/matmul_bias_fusion.mlir --aurora-matmul-bias-fusion
```

Expected: both `matmul` + `bias_add` pairs are fused into `aurora.matmul_bias`. The `relu` is preserved.

## lower_to_linalg.mlir

A `matmul` followed by `relu`, lowered entirely to standard MLIR dialects (Linalg/Arith/Tensor).

```bash
aurora-opt examples/lower_to_linalg.mlir --convert-aurora-to-linalg
```

Expected: `aurora.matmul` becomes `linalg.matmul` + `linalg.fill`, `aurora.relu` becomes `linalg.generic` with `arith.maxf`.

## pipeline_to_llvm.mlir

Demonstrates the full lowering path from Aurora dialect to **LLVM dialect** IR.

### Step 1 — Aurora ops to Linalg (tensors)

```bash
aurora-opt examples/pipeline_to_llvm.mlir --convert-aurora-to-linalg
```

### Step 2 — Bufferize: tensor → memref

```bash
aurora-opt examples/pipeline_to_llvm.mlir \
  --convert-aurora-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries=true allow-return-allocs-in-loops=true"
```

### Step 3 — Full pipeline to LLVM dialect

```bash
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
```

Expected output: `llvm.func`, `llvm.call @malloc`, `llvm.load`/`llvm.store`, LLVM arithmetic ops. No `aurora.*`, `linalg.*`, or `tensor.*` ops remain.

## Full pipeline: fuse then lower to LLVM dialect

```bash
aurora-opt examples/matmul_bias_fusion.mlir \
  --aurora-matmul-bias-fusion \
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
```

## What these demos demonstrate

| Capability | Evidence |
|---|---|
| Custom MLIR dialect parses and prints correctly | `aurora-opt` loads files without errors |
| Pattern-based operator fusion works | Two fusion instances fused in one function |
| Progressive lowering to standard MLIR | Aurora ops become Linalg/Arith ops |
| Bufferization: tensors → memrefs | `--one-shot-bufferize` converts the representation |
| Full pipeline to LLVM dialect | `aurora.matmul` + `aurora.relu` produce `llvm.func` / `llvm.call @malloc` |
| Object file generation | `scripts/aurora-to-obj.sh` chains aurora-opt → mlir-translate → llc |
| Verifier enforces IR constraints | `aurora.bias_add` requires rank-1 bias, last-dim match |
| MlirOptMain integration | `--mlir-print-ir-after-all`, `--mlir-timing`, `--pass-pipeline` work |

## Object file generation (wrapper script)

The wrapper script `scripts/aurora-to-obj.sh` chains all three stages:

```bash
# Full pipeline: Aurora IR -> object file
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt examples/pipeline_to_llvm.mlir
# Output: examples/pipeline_to_llvm.o

# With fusion
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt --fuse examples/matmul_bias_fusion.mlir

# Stop at LLVM IR (no llc needed)
scripts/aurora-to-obj.sh --aurora-opt build/bin/aurora-opt --emit-llvm-ir examples/pipeline_to_llvm.mlir
```

Requires `mlir-translate` and `llc` from LLVM 17 in your PATH (or pass `--translate` / `--llc`).

## What stops here

The wrapper produces object files (`.o`), not linked executables. The generated
functions take `memref` descriptor structs as arguments and call `malloc`/`free`.
To run the code, you need a C/C++ driver that:

1. Allocates input tensors
2. Calls the generated function (e.g., `matmul_relu`)
3. Reads the output tensor

This driver is the next milestone.
