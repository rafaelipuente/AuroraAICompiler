#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
#
# aurora-to-obj.sh -- Aurora IR -> object file wrapper
#
# Runs the Aurora->Linalg->LLVM-dialect pipeline via aurora-opt, then
# translates to LLVM IR and compiles to an object file using external
# LLVM tools (mlir-translate, llc or clang).
#
# Prerequisites (all from an LLVM 17+ installation):
#   - aurora-opt      (build from this repo)
#   - mlir-translate  (from LLVM/MLIR build or system package)
#   - llc or clang    (from LLVM build or system package)
#
# Limitations:
#   - Only supports the Aurora ops that have lowering patterns:
#     relu, add, matmul, bias_add, matmul_bias.
#   - IR containing conv, layernorm, or fused_attention will fail at
#     the convert-aurora-to-linalg stage.
#   - Requires LLVM 17+ for the one-shot-bufferize options used.
#   - Produces an object file (.o), not a linked executable. The
#     generated code calls malloc/free and has no main(); linking
#     into a runnable binary requires a driver with concrete tensors.
#
#===----------------------------------------------------------------------===//

set -euo pipefail

PROG="$(basename "$0")"

usage() {
  cat <<EOF
Usage: $PROG [options] <input.mlir>

Compile Aurora dialect IR to an object file (.o) through the full
MLIR lowering pipeline.

Pipeline:
  aurora-opt   (Aurora -> Linalg -> bufferize -> loops -> LLVM dialect)
  mlir-translate --mlir-to-llvmir  (LLVM dialect -> LLVM IR)
  llc -filetype=obj               (LLVM IR -> object file)

Options:
  -o <file>           Output file (default: <input>.o)
  --aurora-opt <path> Path to aurora-opt   (default: search PATH)
  --translate <path>  Path to mlir-translate (default: search PATH)
  --llc <path>        Path to llc          (default: search PATH)
  --emit-llvm-dialect Stop after aurora-opt (emit LLVM dialect IR)
  --emit-llvm-ir      Stop after mlir-translate (emit LLVM IR .ll)
  --fuse              Run --aurora-matmul-bias-fusion before lowering
  --verbose           Print each pipeline stage
  --help              Show this help
EOF
}

# --- defaults ---
INPUT=""
OUTPUT=""
AURORA_OPT="${AURORA_OPT:-aurora-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLC="${LLC:-llc}"
EMIT_STAGE="obj"   # llvm-dialect | llvm-ir | obj
FUSE=false
VERBOSE=false

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)            OUTPUT="$2"; shift 2 ;;
    --aurora-opt)  AURORA_OPT="$2"; shift 2 ;;
    --translate)   MLIR_TRANSLATE="$2"; shift 2 ;;
    --llc)         LLC="$2"; shift 2 ;;
    --emit-llvm-dialect) EMIT_STAGE="llvm-dialect"; shift ;;
    --emit-llvm-ir)      EMIT_STAGE="llvm-ir"; shift ;;
    --fuse)        FUSE=true; shift ;;
    --verbose)     VERBOSE=true; shift ;;
    --help)        usage; exit 0 ;;
    -*)            echo "$PROG: unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)             INPUT="$1"; shift ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "$PROG: error: no input file specified" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "$PROG: error: input file not found: $INPUT" >&2
  exit 1
fi

BASENAME="${INPUT%.mlir}"
if [[ -z "$OUTPUT" ]]; then
  case "$EMIT_STAGE" in
    llvm-dialect) OUTPUT="${BASENAME}.llvm.mlir" ;;
    llvm-ir)      OUTPUT="${BASENAME}.ll" ;;
    obj)          OUTPUT="${BASENAME}.o" ;;
  esac
fi

# --- tool checks ---
check_tool() {
  local name="$1" path="$2" hint="$3"
  if ! command -v "$path" &>/dev/null; then
    echo "$PROG: error: $name not found at '$path'" >&2
    echo "  $hint" >&2
    exit 1
  fi
}

check_tool "aurora-opt"    "$AURORA_OPT"    "Build this repo or set --aurora-opt /path/to/aurora-opt"
if [[ "$EMIT_STAGE" != "llvm-dialect" ]]; then
  check_tool "mlir-translate" "$MLIR_TRANSLATE" "Install LLVM 17+: apt install mlir-17-tools, or set --translate /path/to/mlir-translate"
fi
if [[ "$EMIT_STAGE" == "obj" ]]; then
  check_tool "llc"            "$LLC"            "Install LLVM 17+: apt install llvm-17, or set --llc /path/to/llc"
fi

log() { $VERBOSE && echo "[$PROG] $*" >&2 || true; }

# --- stage 1: aurora-opt (Aurora -> LLVM dialect) ---
OPT_ARGS=()
if $FUSE; then
  OPT_ARGS+=("--aurora-matmul-bias-fusion")
fi
OPT_ARGS+=(
  "--convert-aurora-to-linalg"
  "--empty-tensor-to-alloc-tensor"
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs=true"
  "--convert-linalg-to-loops"
  "--convert-scf-to-cf"
  "--convert-index-to-llvm"
  "--convert-arith-to-llvm"
  "--convert-cf-to-llvm"
  "--finalize-memref-to-llvm"
  "--convert-func-to-llvm"
  "--reconcile-unrealized-casts"
)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

LLVM_DIALECT="$TMPDIR/llvm_dialect.mlir"

log "Stage 1: aurora-opt ${OPT_ARGS[*]} $INPUT -> $LLVM_DIALECT"
"$AURORA_OPT" "$INPUT" "${OPT_ARGS[@]}" -o "$LLVM_DIALECT"

if [[ "$EMIT_STAGE" == "llvm-dialect" ]]; then
  cp "$LLVM_DIALECT" "$OUTPUT"
  log "Done: LLVM dialect IR written to $OUTPUT"
  echo "$OUTPUT"
  exit 0
fi

# --- stage 2: mlir-translate (LLVM dialect -> LLVM IR) ---
LLVM_IR="$TMPDIR/output.ll"

log "Stage 2: mlir-translate --mlir-to-llvmir $LLVM_DIALECT -> $LLVM_IR"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$LLVM_DIALECT" -o "$LLVM_IR"

if [[ "$EMIT_STAGE" == "llvm-ir" ]]; then
  cp "$LLVM_IR" "$OUTPUT"
  log "Done: LLVM IR written to $OUTPUT"
  echo "$OUTPUT"
  exit 0
fi

# --- stage 3: llc (LLVM IR -> object file) ---
log "Stage 3: llc -filetype=obj $LLVM_IR -> $OUTPUT"
"$LLC" -filetype=obj "$LLVM_IR" -o "$OUTPUT"

log "Done: object file written to $OUTPUT"
echo "$OUTPUT"
