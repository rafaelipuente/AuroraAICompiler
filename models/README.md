# Example Models

This directory contains example AI models for testing the AuroraAICompiler.

## Available Models

- `resnet50.onnx`: ResNet-50 image classification model (placeholder)

## Downloading Models

Since ONNX model files can be large, they are not included directly in the repository.
You can download them using the provided scripts:

```bash
# Download ResNet-50 model
python ../scripts/download_models.py resnet50
```

## Model Sources

- ResNet-50: https://github.com/onnx/models/tree/main/vision/classification/resnet
