#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a simple ONNX model for testing
"""

import os
import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper

# Create a simple model with MatMul and Relu
def create_simple_model(output_path="simple_model.onnx"):
    # Create inputs
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [5, 10])
    
    # Create outputs
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
    
    # Create initializers
    w_value = np.random.randn(5, 10).astype(np.float32)
    W_init = helper.make_tensor('W', TensorProto.FLOAT, [5, 10], w_value.flatten().tolist())
    
    # Create nodes
    matmul_node = helper.make_node(
        'MatMul',
        ['X', 'W'],
        ['matmul_output'],
        name='matmul'
    )
    
    relu_node = helper.make_node(
        'Relu',
        ['matmul_output'],
        ['Y'],
        name='relu'
    )
    
    # Create graph
    graph = helper.make_graph(
        [matmul_node, relu_node],
        'simple_model',
        [X],  # inputs
        [Y],  # outputs
        [W_init]  # initializers
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='aurora_test')
    
    # Save model
    onnx.save(model, output_path)
    print(f"Created ONNX model: {output_path}")
    return output_path

# Create a more complex model that includes conv, matmul, and layernorm patterns
def create_complex_model(output_path="complex_model.onnx"):
    # Create inputs (batch, channels, height, width)
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    
    # Create outputs
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
    
    # Create initializers (weights)
    # Conv weights (out_channels, in_channels, kernel_h, kernel_w)
    conv_w = np.random.randn(16, 3, 3, 3).astype(np.float32)
    conv_w_init = helper.make_tensor('conv_w', TensorProto.FLOAT, [16, 3, 3, 3], conv_w.flatten().tolist())
    
    # FC weights
    fc_w = np.random.randn(16*111*111, 128).astype(np.float32)
    fc_w_init = helper.make_tensor('fc_w', TensorProto.FLOAT, [16*111*111, 128], fc_w.flatten().tolist())
    
    # Output layer weights
    out_w = np.random.randn(128, 10).astype(np.float32)
    out_w_init = helper.make_tensor('out_w', TensorProto.FLOAT, [128, 10], out_w.flatten().tolist())
    
    # Create scale and bias for layer norm
    scale = np.ones(128).astype(np.float32)
    scale_init = helper.make_tensor('scale', TensorProto.FLOAT, [128], scale.tolist())
    
    bias = np.zeros(128).astype(np.float32)
    bias_init = helper.make_tensor('bias', TensorProto.FLOAT, [128], bias.tolist())
    
    # Create nodes
    # Convolution
    conv_node = helper.make_node(
        'Conv',
        ['input', 'conv_w'],
        ['conv_output'],
        name='conv1',
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1]
    )
    
    # Relu after conv
    relu1_node = helper.make_node(
        'Relu',
        ['conv_output'],
        ['relu1_output'],
        name='relu1'
    )
    
    # Reshape for FC
    shape_node = helper.make_node(
        'Shape',
        ['relu1_output'],
        ['shape'],
        name='shape'
    )
    
    gather_node = helper.make_node(
        'Gather',
        ['shape', 'zero'],
        ['batch_size'],
        name='gather',
        axis=0
    )
    
    # Constant for dimension 0
    zero_const = helper.make_tensor('zero', TensorProto.INT64, [], [0])
    zero_node = helper.make_node(
        'Constant',
        [],
        ['zero'],
        name='zero_const',
        value=zero_const
    )
    
    # Constant for flattened dimension
    flat_shape_const = helper.make_tensor('flat_shape_value', TensorProto.INT64, [2], [1, -1])
    flat_shape_node = helper.make_node(
        'Constant',
        [],
        ['flat_shape'],
        name='flat_shape_const',
        value=flat_shape_const
    )
    
    reshape_node = helper.make_node(
        'Reshape',
        ['relu1_output', 'flat_shape'],
        ['reshaped'],
        name='reshape'
    )
    
    # Fully connected (MatMul)
    matmul_node = helper.make_node(
        'MatMul',
        ['reshaped', 'fc_w'],
        ['fc_output'],
        name='fc'
    )
    
    # Layer Normalization
    layer_norm_node = helper.make_node(
        'LayerNormalization',
        ['fc_output', 'scale', 'bias'],
        ['norm_output'],
        name='layernorm',
        epsilon=1e-5
    )
    
    # Relu
    relu2_node = helper.make_node(
        'Relu',
        ['norm_output'],
        ['relu2_output'],
        name='relu2'
    )
    
    # Output layer
    output_node = helper.make_node(
        'MatMul',
        ['relu2_output', 'out_w'],
        ['output'],
        name='output_layer'
    )
    
    # Create graph
    graph = helper.make_graph(
        [zero_node, flat_shape_node, conv_node, relu1_node, shape_node, gather_node, reshape_node, 
         matmul_node, layer_norm_node, relu2_node, output_node],
        'complex_model',
        [X],  # inputs
        [Y],  # outputs
        [conv_w_init, fc_w_init, out_w_init, scale_init, bias_init]  # initializers
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='aurora_test')
    
    # Save model
    onnx.save(model, output_path)
    print(f"Created complex ONNX model: {output_path}")
    return output_path

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create simple model
    simple_model_path = os.path.join("models", "simple_model.onnx")
    create_simple_model(simple_model_path)
    
    # Create complex model
    complex_model_path = os.path.join("models", "complex_model.onnx")
    create_complex_model(complex_model_path)
