#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a simple ONNX model for testing
"""
import os
import numpy as np
import onnx
from onnx import TensorProto, helper

def create_matmul_model(output_path):
    """Create a simple MatMul model"""
    # Create inputs
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [5, 10])
    
    # Create outputs
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
    
    # Create initializers
    w_data = np.random.randn(5, 10).astype(np.float32)
    w_init = helper.make_tensor('W', TensorProto.FLOAT, [5, 10], w_data.flatten().tolist())
    
    # Create node
    node = helper.make_node(
        'MatMul',
        ['X', 'W'],
        ['Y'],
        name='MatMul'
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        'matmul_model',
        [X],
        [Y],
        [w_init]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='aurora_test')
    model.opset_import[0].version = 13
    
    # Check model
    onnx.checker.check_model(model)
    
    # Save model
    onnx.save(model, output_path)
    print(f"Saved model to: {output_path}")

if __name__ == "__main__":
    create_matmul_model("matmul_test.onnx")
