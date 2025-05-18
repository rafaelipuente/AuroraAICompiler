#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the ONNX loader

This script shows how to:
1. Load an ONNX model
2. Analyze its structure 
3. Access operation nodes and tensors
"""

import os
import sys
import argparse
from onnx_loader import ONNXLoader, GraphInfo


def analyze_model(model_path):
    """Analyze an ONNX model and print useful information"""
    print(f"Analyzing ONNX model: {model_path}")
    
    # Create the loader
    loader = ONNXLoader(verbose=True)
    
    # Load the model
    graph = loader.load(model_path)
    
    # Print basic graph summary
    loader.display_graph_summary(graph)
    
    # Perform additional analysis
    print("\n-- Layer Analysis --")
    
    # Compute and display statistics
    conv_nodes = [op for op in graph.operations if op.op_type == "Conv"]
    gemm_nodes = [op for op in graph.operations if op.op_type == "Gemm"]
    relu_nodes = [op for op in graph.operations if op.op_type == "Relu"]
    
    print(f"Convolution layers: {len(conv_nodes)}")
    print(f"Gemm/FC layers: {len(gemm_nodes)}")
    print(f"ReLU activations: {len(relu_nodes)}")
    
    # Calculate total parameters
    total_params = 0
    for tensor in graph.tensors.values():
        if tensor.is_constant and tensor.constant_value is not None:
            total_params += tensor.constant_value.size
    
    print(f"Total parameters: {total_params:,}")
    
    # Find potential Aurora operations mapping
    print("\n-- Aurora Dialect Mapping --")
    aurora_ops = {
        "MatMul": "aurora.matmul",
        "Conv": "aurora.conv",
        "Relu": "aurora.relu",
        "LayerNormalization": "aurora.layernorm",
    }
    
    for op_type, aurora_op in aurora_ops.items():
        count = sum(1 for op in graph.operations if op.op_type == op_type)
        if count > 0:
            print(f"  • {op_type} -> {aurora_op}: {count} instances")
    
    # Custom mapping for attention patterns
    attention_pattern_count = 0
    for i in range(len(graph.operations) - 3):
        # Simple heuristic: MatMul -> Softmax -> MatMul pattern might be attention
        if (graph.operations[i].op_type == "MatMul" and
            graph.operations[i+1].op_type == "Softmax" and
            graph.operations[i+2].op_type == "MatMul"):
            attention_pattern_count += 1
    
    if attention_pattern_count > 0:
        print(f"  • Potential attention patterns -> aurora.fused_attention: {attention_pattern_count} instances")


def main():
    parser = argparse.ArgumentParser(description="ONNX Model Analyzer Example")
    parser.add_argument("model_path", help="Path to the ONNX model file")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    try:
        analyze_model(args.model_path)
        return 0
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
