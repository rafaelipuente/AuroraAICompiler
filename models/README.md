# Models

Test model files and generation scripts for AuroraAICompiler.

## Contents

- `matmul_test.onnx` -- Tiny MatMul ONNX model (320 bytes). Used for import testing.
- `create_test_model.py` -- Generates ONNX test models with MatMul, Add, and Relu ops.
- `create_simple_model.py` -- Generates a minimal MatMul ONNX model.
- `simple_benchmark.py` -- Benchmark script (requires working compiler pipeline; not functional yet).
- `standalone_test.py` -- Standalone test runner (requires working compiler pipeline; not functional yet).

## Generating Models

```bash
cd models
python create_simple_model.py
python create_test_model.py
```

Both scripts require `onnx` and `numpy` to be installed.

## Note

Large pretrained models (ResNet, MobileNet, etc.) are not included in this repository.
Use `scripts/download_models.py` to fetch them if needed.
