#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified benchmark for ONNX to MLIR conversion
"""

import os
import sys
import time
import statistics
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct imports to avoid dependency issues
from python.aurora.model_import.onnx_loader import ONNXLoader
from python.aurora.model_import.onnx_to_mlir import ONNXToMLIRConverter

def benchmark_model(model_path, output_dir="benchmark_results", num_runs=3, verbose=False):
    """Run a simple benchmark on an ONNX model
    
    Args:
        model_path: Path to the ONNX model
        output_dir: Directory for benchmark results
        num_runs: Number of runs for averaging performance
        verbose: Enable verbose logging
        
    Returns:
        Dictionary of benchmark results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Benchmarking model: {model_path}")
    print(f"Number of runs: {num_runs}")
    
    # Get model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Benchmark loading
    load_times = []
    graph = None
    
    print("Measuring load time...")
    for i in range(num_runs):
        start_time = time.time()
        loader = ONNXLoader(verbose=verbose)
        graph = loader.load(model_path)
        end_time = time.time()
        
        load_time_ms = (end_time - start_time) * 1000
        load_times.append(load_time_ms)
        
        print(f"  Run {i+1}: {load_time_ms:.2f} ms")
    
    # Calculate average load time
    avg_load_time_ms = statistics.mean(load_times)
    print(f"Average load time: {avg_load_time_ms:.2f} ms")
    
    # Benchmark conversion
    convert_times = []
    mlir_text = None
    
    print("Measuring conversion time...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{os.path.splitext(os.path.basename(model_path))[0]}_{timestamp}.mlir"
    mlir_path = os.path.join(output_dir, output_name)
    
    for i in range(num_runs):
        start_time = time.time()
        converter = ONNXToMLIRConverter(verbose=verbose)
        mlir_text = converter.convert(graph)
        end_time = time.time()
        
        convert_time_ms = (end_time - start_time) * 1000
        convert_times.append(convert_time_ms)
        
        print(f"  Run {i+1}: {convert_time_ms:.2f} ms")
    
    # Calculate average convert time
    avg_convert_time_ms = statistics.mean(convert_times)
    print(f"Average convert time: {avg_convert_time_ms:.2f} ms")
    
    # Write MLIR to file
    with open(mlir_path, 'w') as f:
        f.write(mlir_text)
    
    # Get MLIR size
    mlir_size_mb = os.path.getsize(mlir_path) / (1024 * 1024)
    print(f"MLIR size: {mlir_size_mb:.2f} MB")
    
    # Count operations in the model
    num_operations = len(graph.operations)
    
    # Count converted operations
    num_conversions = sum(1 for line in mlir_text.split('\n') if ' = aurora.' in line)
    
    # Calculate conversion rate
    conversion_rate = (num_operations / (avg_convert_time_ms / 1000)) if avg_convert_time_ms > 0 else 0
    
    # Print results in a nice table
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Model name: {os.path.basename(model_path)}")
    print(f"Operations: {num_operations}")
    print(f"Inputs: {len(graph.inputs)}")
    print(f"Outputs: {len(graph.outputs)}")
    print(f"Load time: {avg_load_time_ms:.2f} ms")
    print(f"Convert time: {avg_convert_time_ms:.2f} ms")
    print(f"Total time: {(avg_load_time_ms + avg_convert_time_ms):.2f} ms")
    print(f"MLIR file size: {mlir_size_mb:.2f} MB")
    print(f"Conversion rate: {conversion_rate:.2f} ops/sec")
    print(f"Operations converted: {num_conversions}/{num_operations} ({num_conversions/num_operations*100:.1f}%)")
    print(f"MLIR output: {mlir_path}")
    
    return {
        "model_name": os.path.basename(model_path),
        "model_size_mb": model_size_mb,
        "num_operations": num_operations,
        "load_time_ms": avg_load_time_ms,
        "convert_time_ms": avg_convert_time_ms,
        "mlir_size_mb": mlir_size_mb,
        "mlir_path": mlir_path,
        "num_inputs": len(graph.inputs),
        "num_outputs": len(graph.outputs),
        "num_conversions": num_conversions,
        "conversion_rate": conversion_rate
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ONNX to MLIR benchmark")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        benchmark_model(args.model, args.output_dir, args.runs, args.verbose)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
