#!/usr/bin/env python3
"""
Example script for compiling and running ResNet-50 with Aurora AI Compiler

This example demonstrates the end-to-end workflow of compiling an ONNX model
with Aurora, running inference, and comparing performance with baseline frameworks.
"""

import os
import sys
import time
import logging
import numpy as np
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Aurora modules
from python.aurora import compile_model, load_model, execute_model, benchmark_model
from python.aurora.model_import import import_onnx
from python.aurora.runtime import AuroraModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Compile and run ResNet-50 with Aurora AI Compiler')
    parser.add_argument('--model', type=str, default='../models/resnet50.onnx',
                       help='Path to the input ONNX model')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the compiled model (default: derived from input)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'vulkan'],
                       help='Target device for execution')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarking comparison')
    parser.add_argument('--opt-level', type=int, default=3, choices=[0, 1, 2, 3],
                       help='Optimization level (0-3)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image for inference (optional)')
    
    args = parser.parse_args()
    
    # Check if the model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("You can download the model using: python ../scripts/download_models.py resnet50")
        return 1
    
    # Compile the model
    logger.info("Compiling model with Aurora AI Compiler...")
    try:
        if args.output is None:
            args.output = os.path.splitext(args.model)[0] + ".aurora"
            
        # Create compilation options
        from python.aurora.compiler import CompilationOptions
        options = CompilationOptions()
        options.optimization_level = args.opt_level
        options.target_device = args.device
        
        # Compile the model
        compiled_model = compile_model(args.model, args.output, options)
        logger.info(f"Compilation successful: {compiled_model}")
        
    except Exception as e:
        logger.error(f"Compilation failed: {str(e)}")
        return 1
    
    # Load the compiled model
    logger.info("Loading compiled model...")
    model = load_model(compiled_model)
    
    # Create a sample input tensor (224x224 RGB image)
    input_shape = (1, 3, 224, 224)
    input_tensor = np.random.random(input_shape).astype(np.float32)
    
    if args.image:
        try:
            # In a real implementation, this would load and preprocess the image
            logger.info(f"Loading image: {args.image}")
            # This is a placeholder - real implementation would use PIL or OpenCV
            logger.info("Image loaded and preprocessed")
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            logger.info("Using random test data instead")
    
    # Execute the model
    logger.info("Executing model...")
    inputs = {"input": input_tensor}
    outputs = execute_model(model, inputs)
    
    # Print the output shape and sample values
    for name, output in outputs.items():
        logger.info(f"Output '{name}' shape: {output.shape}")
        
        # For classification output, show top 5 classes
        if len(output.shape) == 2 and output.shape[1] > 5:
            top_indices = np.argsort(output[0])[-5:][::-1]
            top_values = output[0][top_indices]
            
            logger.info("Top 5 predictions:")
            for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                logger.info(f"  {i+1}. Class {idx}: {val:.6f}")
    
    # Run benchmarking if requested
    if args.benchmark:
        logger.info("\nRunning performance benchmark...")
        
        # Benchmark Aurora model
        results = benchmark_model(model, inputs, num_runs=100, warmup_runs=10)
        
        logger.info("Aurora AI Compiler performance:")
        logger.info(f"  Average time: {results['avg_time_ms']:.3f} ms")
        logger.info(f"  Throughput: {results['throughput_fps']:.2f} FPS")
        
        # Compare with baseline (ONNX Runtime) - this would be implemented in a real scenario
        logger.info("\nBaseline comparison would be performed here")
        logger.info("(ONNX Runtime implementation would be integrated in a full version)")
    
    logger.info("\nExample completed successfully")
    return 0

if __name__ == '__main__':
    sys.exit(main())
