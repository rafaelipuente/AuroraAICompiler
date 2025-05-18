#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone test script for ONNX to MLIR conversion
This script directly imports the necessary code from the source files
to avoid package dependency issues.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure direct imports from the source files
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aurora.test")

# Import directly from the source files 
# (avoiding package structure with dependencies on torch)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python/aurora/model_import'))
from onnx_loader import ONNXLoader, GraphInfo
from onnx_to_mlir import ONNXToMLIRConverter

def test_conversion(model_path, output_path=None, verbose=False):
    """Test the ONNX to MLIR conversion process
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the MLIR file (optional)
        verbose: Enable verbose output
    """
    print(f"Testing ONNX to MLIR conversion for: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    # Create output directory if needed
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    else:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base_name}_output.mlir"
    
    try:
        # Step 1: Load the ONNX model
        print("1. Loading ONNX model...")
        start_time = time.time()
        loader = ONNXLoader(verbose=verbose)
        graph = loader.load(model_path)
        load_time = time.time() - start_time
        print(f"   - Loaded model with {len(graph.operations)} operations")
        print(f"   - Load time: {load_time*1000:.2f} ms")
        
        # Step 2: Convert to MLIR
        print("2. Converting to MLIR...")
        start_time = time.time()
        converter = ONNXToMLIRConverter(verbose=verbose)
        mlir_text = converter.convert(graph)
        convert_time = time.time() - start_time
        print(f"   - Conversion time: {convert_time*1000:.2f} ms")
        
        # Step 3: Write to file
        print(f"3. Writing MLIR to {output_path}...")
        with open(output_path, 'w') as f:
            f.write(mlir_text)
        print(f"   - MLIR file size: {os.path.getsize(output_path)/1024:.2f} KB")
        
        # Step 4: Analyze the resulting MLIR
        aurora_ops = sum(1 for line in mlir_text.split('\n') if ' = aurora.' in line)
        print(f"4. Analysis:")
        print(f"   - Aurora operations: {aurora_ops}")
        print(f"   - Conversion rate: {aurora_ops/len(graph.operations)*100:.1f}%")
        
        print(f"\nConversion completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ONNX to MLIR conversion")
    parser.add_argument("--model", 
                       default="matmul_test.onnx", 
                       help="Path to the ONNX model")
    parser.add_argument("--output", 
                       help="Path to save the MLIR output (optional)")
    parser.add_argument("--verbose", 
                       action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    success = test_conversion(args.model, args.output, args.verbose)
    sys.exit(0 if success else 1)
