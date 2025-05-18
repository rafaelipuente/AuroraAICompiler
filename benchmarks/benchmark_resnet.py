#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark for ONNX to MLIR conversion of ResNet models

This benchmark:
- Loads ResNet models from ONNX files
- Converts them to MLIR using the Aurora dialect
- Measures performance metrics
- Outputs results in CSV and a pretty table
"""

import os
import sys
import time
import argparse
import statistics
import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Aurora modules
from python.aurora.model_import.onnx_loader import ONNXLoader, GraphInfo
from python.aurora.model_import.onnx_to_mlir import ONNXToMLIRConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('aurora.benchmark')


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    model_name: str
    model_path: str
    model_size_mb: float
    num_operations: int
    load_time_ms: float
    convert_time_ms: float
    mlir_size_mb: float
    mlir_path: str
    timestamp: str
    num_inputs: int
    num_outputs: int
    num_parameters: int
    num_conversions: int
    conversion_rate: float  # ops/sec


class Benchmark:
    """Benchmark for ONNX to MLIR conversion process"""
    
    def __init__(
        self, 
        model_path: str, 
        output_dir: str = "benchmark_results", 
        num_runs: int = 3,
        verbose: bool = False
    ):
        """Initialize the benchmark
        
        Args:
            model_path: Path to the ONNX model
            output_dir: Directory for benchmark results
            num_runs: Number of runs for averaging performance
            verbose: Enable verbose logging
        """
        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.basename(model_path)
        self.output_dir = output_dir
        self.num_runs = num_runs
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set log level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Validate model path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def _get_model_size_mb(self, file_path: str) -> float:
        """Get the size of a file in MB
        
        Args:
            file_path: Path to the file
            
        Returns:
            Size in MB
        """
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def _count_parameters(self, graph: GraphInfo) -> int:
        """Count the number of parameters in a model
        
        Args:
            graph: Model graph
            
        Returns:
            Number of parameters
        """
        params = 0
        for tensor in graph.tensors.values():
            if tensor.is_constant and tensor.constant_value is not None:
                params += tensor.constant_value.size
        return params
    
    def benchmark_load(self) -> tuple[GraphInfo, float]:
        """Benchmark the ONNX model loading process
        
        Returns:
            Tuple of (GraphInfo, load_time_ms)
        """
        logger.info(f"Benchmarking model loading: {self.model_name}")
        
        # Run multiple times and calculate average
        load_times = []
        graph = None
        
        for i in range(self.num_runs):
            logger.debug(f"Load run {i+1}/{self.num_runs}")
            
            # Measure loading time
            start_time = time.time()
            loader = ONNXLoader(verbose=self.verbose)
            graph = loader.load(self.model_path)
            end_time = time.time()
            
            load_time_ms = (end_time - start_time) * 1000
            load_times.append(load_time_ms)
            
            logger.debug(f"Run {i+1} load time: {load_time_ms:.2f} ms")
        
        # Calculate average load time
        avg_load_time_ms = statistics.mean(load_times)
        logger.info(f"Average load time: {avg_load_time_ms:.2f} ms")
        
        return graph, avg_load_time_ms
    
    def benchmark_convert(self, graph: GraphInfo) -> tuple[str, float, str]:
        """Benchmark the conversion process from ONNX to MLIR
        
        Args:
            graph: GraphInfo object
            
        Returns:
            Tuple of (mlir_text, convert_time_ms, mlir_path)
        """
        logger.info(f"Benchmarking model conversion: {self.model_name}")
        
        # Generate output MLIR path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{os.path.splitext(self.model_name)[0]}_{timestamp}.mlir"
        mlir_path = os.path.join(self.output_dir, output_name)
        
        # Run multiple times and calculate average
        convert_times = []
        mlir_text = None
        
        for i in range(self.num_runs):
            logger.debug(f"Convert run {i+1}/{self.num_runs}")
            
            # Measure conversion time
            start_time = time.time()
            converter = ONNXToMLIRConverter(verbose=self.verbose)
            mlir_text = converter.convert(graph)
            end_time = time.time()
            
            convert_time_ms = (end_time - start_time) * 1000
            convert_times.append(convert_time_ms)
            
            logger.debug(f"Run {i+1} convert time: {convert_time_ms:.2f} ms")
        
        # Calculate average convert time
        avg_convert_time_ms = statistics.mean(convert_times)
        logger.info(f"Average convert time: {avg_convert_time_ms:.2f} ms")
        
        # Write MLIR to file (only once)
        with open(mlir_path, 'w') as f:
            f.write(mlir_text)
        
        logger.info(f"MLIR written to: {mlir_path}")
        
        return mlir_text, avg_convert_time_ms, mlir_path
    
    def run(self) -> BenchmarkResult:
        """Run the benchmark
        
        Returns:
            BenchmarkResult object
        """
        logger.info(f"Starting benchmark for {self.model_name}")
        
        # Get model size
        model_size_mb = self._get_model_size_mb(self.model_path)
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        # Benchmark loading
        graph, load_time_ms = self.benchmark_load()
        
        # Benchmark conversion
        mlir_text, convert_time_ms, mlir_path = self.benchmark_convert(graph)
        
        # Get MLIR size
        mlir_size_mb = self._get_model_size_mb(mlir_path)
        logger.info(f"MLIR size: {mlir_size_mb:.2f} MB")
        
        # Count operations in the model
        num_operations = len(graph.operations)
        
        # Count parameters in the model
        num_parameters = self._count_parameters(graph)
        
        # Count converted operations (non-dummy/non-skipped)
        num_conversions = sum(1 for line in mlir_text.split('\n') 
                             if ' = aurora.' in line)
        
        # Calculate conversion rate (ops/sec)
        conversion_rate = (num_operations / (convert_time_ms / 1000)) if convert_time_ms > 0 else 0
        
        # Create benchmark result
        result = BenchmarkResult(
            model_name=self.model_name,
            model_path=self.model_path,
            model_size_mb=model_size_mb,
            num_operations=num_operations,
            load_time_ms=load_time_ms,
            convert_time_ms=convert_time_ms,
            mlir_size_mb=mlir_size_mb,
            mlir_path=mlir_path,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_inputs=len(graph.inputs),
            num_outputs=len(graph.outputs),
            num_parameters=num_parameters,
            num_conversions=num_conversions,
            conversion_rate=conversion_rate
        )
        
        logger.info("Benchmark completed successfully")
        return result
    
    def save_results_csv(self, result: BenchmarkResult):
        """Save benchmark results to a CSV file
        
        Args:
            result: BenchmarkResult to save
        """
        # Generate CSV path
        csv_path = os.path.join(self.output_dir, "benchmark_results.csv")
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_path)
        
        # Convert result to dictionary
        result_dict = {k: v for k, v in result.__dict__.items()}
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_dict.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(result_dict)
        
        logger.info(f"Results saved to CSV: {csv_path}")
    
    def print_results_table(self, result: BenchmarkResult):
        """Print benchmark results in a pretty table
        
        Args:
            result: BenchmarkResult to print
        """
        # Create a horizontal line
        hline = '+' + '-' * 50 + '+' + '-' * 30 + '+'
        
        print("\n")
        print("=== Aurora ONNX to MLIR Benchmark Results ===")
        print(hline)
        print(f"| {'Metric':<48} | {'Value':<28} |")
        print(hline)
        print(f"| {'Model Name':<48} | {result.model_name:<28} |")
        print(f"| {'Model Size':<48} | {result.model_size_mb:.2f} MB{' ':<21} |")
        print(f"| {'Number of Operations':<48} | {result.num_operations:<28} |")
        print(f"| {'Number of Inputs':<48} | {result.num_inputs:<28} |")
        print(f"| {'Number of Outputs':<48} | {result.num_outputs:<28} |")
        print(f"| {'Number of Parameters':<48} | {result.num_parameters:,}{' ':<28-len(f'{result.num_parameters:,}')} |")
        print(hline)
        print(f"| {'Load Time':<48} | {result.load_time_ms:.2f} ms{' ':<20} |")
        print(f"| {'Convert Time':<48} | {result.convert_time_ms:.2f} ms{' ':<20} |")
        print(f"| {'Total Time':<48} | {(result.load_time_ms + result.convert_time_ms):.2f} ms{' ':<20} |")
        print(hline)
        print(f"| {'MLIR File Size':<48} | {result.mlir_size_mb:.2f} MB{' ':<21} |")
        print(f"| {'Conversion Rate':<48} | {result.conversion_rate:.2f} ops/sec{' ':<17} |")
        print(f"| {'Operations Converted':<48} | {result.num_conversions}/{result.num_operations} ({result.num_conversions/result.num_operations*100:.1f}%){' ':<4} |")
        print(hline)
        print(f"| {'Timestamp':<48} | {result.timestamp:<28} |")
        print(hline)
        print(f"MLIR output: {result.mlir_path}")
        print("\n")


def main():
    """Main function for the benchmark"""
    parser = argparse.ArgumentParser(description="Benchmark ONNX to MLIR conversion for ResNet")
    
    parser.add_argument(
        "--model", 
        default="resnet50.onnx",
        help="Path to the ONNX model (default: resnet50.onnx)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for benchmark results (default: benchmark_results)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for averaging performance (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run benchmark
        benchmark = Benchmark(
            model_path=args.model,
            output_dir=args.output_dir,
            num_runs=args.runs,
            verbose=args.verbose
        )
        
        result = benchmark.run()
        
        # Save and print results
        benchmark.save_results_csv(result)
        benchmark.print_results_table(result)
        
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
