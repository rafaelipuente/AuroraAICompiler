#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX to MLIR converter for the Aurora AI Compiler

This module provides functionality to:
- Convert an intermediate GraphInfo representation to MLIR text
- Map ONNX operations to Aurora dialect operations
- Emit valid MLIR code for the Aurora compiler
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass

from onnx_loader import ONNXLoader, GraphInfo, OperationNode, TensorInfo, DataType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aurora.onnx_to_mlir")


class ONNXToMLIRConverter:
    """Converts ONNX models to MLIR using the Aurora dialect"""
    
    def __init__(self, verbose: bool = False):
        """Initialize the ONNX to MLIR converter
        
        Args:
            verbose: Whether to print detailed debugging information
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Map of ONNX operations to Aurora dialect operations
        self.op_map = {
            "MatMul": self._convert_matmul,
            "Gemm": self._convert_gemm,
            "Conv": self._convert_conv,
            "Relu": self._convert_relu,
            "LayerNormalization": self._convert_layernorm,
            # Special case pattern for attention
            "_Attention": self._convert_attention,
        }
        
        # Track variables created in the MLIR
        self.value_map = {}  # Maps tensor names to MLIR SSA values
        self.tensor_types = {}  # Maps tensor names to MLIR tensor types
    
    def _get_tensor_type_str(self, tensor: TensorInfo) -> str:
        """Convert a TensorInfo to an MLIR tensor type string
        
        Args:
            tensor: TensorInfo object
            
        Returns:
            MLIR tensor type string
        """
        # Map Aurora data types to MLIR types
        type_map = {
            DataType.FLOAT: "f32",
            DataType.FLOAT16: "f16",
            DataType.INT8: "i8",
            DataType.INT32: "i32",
            DataType.INT64: "i64",
            DataType.BOOL: "i1",
            DataType.UNKNOWN: "f32"  # Default to f32 for unknown types
        }
        
        mlir_type = type_map.get(tensor.data_type, "f32")
        
        # Handle shape
        shape_str = ""
        if tensor.shape:
            dim_strs = []
            for dim in tensor.shape:
                if dim is None:
                    dim_strs.append("?")  # Dynamic dimension
                else:
                    dim_strs.append(str(dim))
            shape_str = "x".join(dim_strs)
        else:
            shape_str = "?"  # Unknown shape
        
        return f"tensor<{shape_str}x{mlir_type}>"
    
    def _get_mlir_value(self, tensor_name: str) -> str:
        """Get the MLIR SSA value for a tensor
        
        Args:
            tensor_name: Name of the tensor
            
        Returns:
            MLIR SSA value
        """
        return self.value_map.get(tensor_name, f"%{tensor_name.replace('.', '_')}")

    def _detect_attention_patterns(self, graph: GraphInfo) -> List[List[OperationNode]]:
        """Detect attention patterns in the graph
        
        Args:
            graph: GraphInfo object
            
        Returns:
            List of lists of nodes forming attention patterns
        """
        attention_patterns = []
        
        # For simplicity, we'll detect a basic pattern:
        # Three MatMul operations (Q, K, V) -> MatMul (QK) -> Softmax -> MatMul (Attention)
        
        for i in range(len(graph.operations) - 5):
            # Check if we have a sequence that might be attention
            window = graph.operations[i:i+6]
            
            # Simple heuristic: 3 matmuls for Q,K,V followed by the attention computation
            if (all(op.op_type == "MatMul" for op in window[:3]) and
                window[3].op_type == "MatMul" and
                window[4].op_type == "Softmax" and
                window[5].op_type == "MatMul"):
                
                attention_patterns.append(window)
        
        logger.info(f"Detected {len(attention_patterns)} potential attention patterns")
        return attention_patterns

    def _convert_matmul(self, node: OperationNode, graph: GraphInfo) -> str:
        """Convert MatMul operation to Aurora dialect
        
        Args:
            node: OperationNode for MatMul
            graph: GraphInfo containing the model
            
        Returns:
            MLIR code for aurora.matmul operation
        """
        input_a = self._get_mlir_value(node.inputs[0])
        input_b = self._get_mlir_value(node.inputs[1])
        output = self._get_mlir_value(node.outputs[0])
        
        # Get tensor types
        type_a = self.tensor_types.get(node.inputs[0], "tensor<??xf32>")
        type_b = self.tensor_types.get(node.inputs[1], "tensor<??xf32>")
        type_out = self.tensor_types.get(node.outputs[0], "tensor<??xf32>")
        
        # Generate the operation
        self.value_map[node.outputs[0]] = output
        return f"{output} = aurora.matmul({input_a}, {input_b}) : ({type_a}, {type_b}) -> {type_out}"
    
    def _convert_gemm(self, node: OperationNode, graph: GraphInfo) -> str:
        """Convert Gemm operation to Aurora dialect matmul with attributes
        
        Args:
            node: OperationNode for Gemm
            graph: GraphInfo containing the model
            
        Returns:
            MLIR code for aurora.matmul operation
        """
        input_a = self._get_mlir_value(node.inputs[0])
        input_b = self._get_mlir_value(node.inputs[1])
        output = self._get_mlir_value(node.outputs[0])
        
        # Get tensor types
        type_a = self.tensor_types.get(node.inputs[0], "tensor<??xf32>")
        type_b = self.tensor_types.get(node.inputs[1], "tensor<??xf32>")
        type_out = self.tensor_types.get(node.outputs[0], "tensor<??xf32>")
        
        # Extract Gemm attributes
        transpose_a = False
        transpose_b = False
        
        for attr in node.attributes:
            if attr.name == "transA" and attr.value == 1:
                transpose_a = True
            elif attr.name == "transB" and attr.value == 1:
                transpose_b = True
        
        # Generate the operation with transpose attributes if needed
        attrs = []
        if transpose_a:
            attrs.append("transpose_lhs = true")
        if transpose_b:
            attrs.append("transpose_rhs = true")
        
        attr_str = ""
        if attrs:
            attr_str = " { " + ", ".join(attrs) + " }"
        
        self.value_map[node.outputs[0]] = output
        return f"{output} = aurora.matmul({input_a}, {input_b}){attr_str} : ({type_a}, {type_b}) -> {type_out}"
    
    def _convert_conv(self, node: OperationNode, graph: GraphInfo) -> str:
        """Convert Conv operation to Aurora dialect
        
        Args:
            node: OperationNode for Conv
            graph: GraphInfo containing the model
            
        Returns:
            MLIR code for aurora.conv operation
        """
        input_tensor = self._get_mlir_value(node.inputs[0])
        filter_tensor = self._get_mlir_value(node.inputs[1])
        output = self._get_mlir_value(node.outputs[0])
        
        # Get tensor types
        type_input = self.tensor_types.get(node.inputs[0], "tensor<??xf32>")
        type_filter = self.tensor_types.get(node.inputs[1], "tensor<??xf32>")
        type_out = self.tensor_types.get(node.outputs[0], "tensor<??xf32>")
        
        # Extract Conv attributes
        strides = []
        paddings = []
        dilations = []
        groups = 1
        
        for attr in node.attributes:
            if attr.name == "strides":
                strides = attr.value
            elif attr.name == "pads":
                # Convert from [begin_axis0, begin_axis1, ..., end_axis0, end_axis1, ...] to
                # [begin_axis0, end_axis0, begin_axis1, end_axis1, ...]
                pads = attr.value
                if pads and len(pads) % 2 == 0:
                    half = len(pads) // 2
                    paddings = []
                    for i in range(half):
                        paddings.extend([pads[i], pads[i + half]])
                else:
                    paddings = pads
            elif attr.name == "dilations":
                dilations = attr.value
            elif attr.name == "group":
                groups = attr.value
        
        # Prepare attribute strings
        attrs = []
        if strides:
            attrs.append(f"strides = [{', '.join(map(str, strides))}]")
        if paddings:
            attrs.append(f"paddings = [{', '.join(map(str, paddings))}]")
        if dilations:
            attrs.append(f"dilations = [{', '.join(map(str, dilations))}]")
        if groups != 1:
            attrs.append(f"groups = {groups}")
        
        attr_str = ""
        if attrs:
            attr_str = " { " + ", ".join(attrs) + " }"
        
        self.value_map[node.outputs[0]] = output
        return f"{output} = aurora.conv({input_tensor}, {filter_tensor}){attr_str} : ({type_input}, {type_filter}) -> {type_out}"
    
    def _convert_relu(self, node: OperationNode, graph: GraphInfo) -> str:
        """Convert Relu operation to Aurora dialect
        
        Args:
            node: OperationNode for Relu
            graph: GraphInfo containing the model
            
        Returns:
            MLIR code for aurora.relu operation
        """
        input_tensor = self._get_mlir_value(node.inputs[0])
        output = self._get_mlir_value(node.outputs[0])
        
        # Get tensor types
        type_input = self.tensor_types.get(node.inputs[0], "tensor<??xf32>")
        type_out = self.tensor_types.get(node.outputs[0], "tensor<??xf32>")
        
        self.value_map[node.outputs[0]] = output
        return f"{output} = aurora.relu({input_tensor}) : ({type_input}) -> {type_out}"
    
    def _convert_layernorm(self, node: OperationNode, graph: GraphInfo) -> str:
        """Convert LayerNormalization operation to Aurora dialect
        
        Args:
            node: OperationNode for LayerNormalization
            graph: GraphInfo containing the model
            
        Returns:
            MLIR code for aurora.layernorm operation
        """
        input_tensor = self._get_mlir_value(node.inputs[0])
        
        # Get scale and bias if available
        scale = None
        bias = None
        if len(node.inputs) > 1:
            scale = self._get_mlir_value(node.inputs[1])
        if len(node.inputs) > 2:
            bias = self._get_mlir_value(node.inputs[2])
        
        output = self._get_mlir_value(node.outputs[0])
        
        # Get tensor types
        type_input = self.tensor_types.get(node.inputs[0], "tensor<??xf32>")
        type_out = self.tensor_types.get(node.outputs[0], "tensor<??xf32>")
        
        # Extract LayerNormalization attributes
        epsilon = 1.0e-5
        axis = -1
        
        for attr in node.attributes:
            if attr.name == "epsilon":
                epsilon = attr.value
            elif attr.name == "axis":
                axis = attr.value
        
        # Build operand list
        operands = [input_tensor]
        if scale:
            operands.append(scale)
        if bias:
            operands.append(bias)
        
        operands_str = ", ".join(operands)
        
        # Prepare attribute string
        attrs = []
        attrs.append(f"epsilon = {epsilon:.7f}")
        attrs.append(f"axis = {axis}")
        
        attr_str = " { " + ", ".join(attrs) + " }"
        
        # Build types string
        types_list = [type_input]
        if scale:
            scale_type = self.tensor_types.get(node.inputs[1], "tensor<?xf32>")
            types_list.append(scale_type)
        if bias:
            bias_type = self.tensor_types.get(node.inputs[2], "tensor<?xf32>")
            types_list.append(bias_type)
        
        types_str = ", ".join(types_list)
        
        self.value_map[node.outputs[0]] = output
        return f"{output} = aurora.layernorm({operands_str}){attr_str} : ({types_str}) -> {type_out}"
    
    def _convert_attention(self, attention_nodes: List[OperationNode], graph: GraphInfo) -> str:
        """Convert an attention pattern to Aurora's fused_attention op
        
        Args:
            attention_nodes: List of OperationNodes forming an attention pattern
            graph: GraphInfo containing the model
            
        Returns:
            MLIR code for aurora.fused_attention operation
        """
        # This is a simplified implementation assuming a specific attention pattern
        # In a real-world scenario, more sophisticated pattern matching would be needed
        
        if len(attention_nodes) < 6:
            logger.warning("Attention pattern doesn't have enough nodes")
            return "// Skipped attention due to insufficient nodes"
        
        # Assume the first three MatMuls are for Q, K, V
        qkv_inputs = []
        for i in range(3):
            qkv_inputs.append(self._get_mlir_value(attention_nodes[i].inputs[0]))
        
        # Use the last MatMul output as our output
        output = self._get_mlir_value(attention_nodes[5].outputs[0])
        
        # Extract query, key, value weight matrices from the first three MatMuls
        q_weights = self._get_mlir_value(attention_nodes[0].inputs[1])
        k_weights = self._get_mlir_value(attention_nodes[1].inputs[1])
        v_weights = self._get_mlir_value(attention_nodes[2].inputs[1])
        
        # In this simplified implementation, we'll use the first input for the attention
        input_tensor = qkv_inputs[0]
        
        # Determine input and output types
        input_type = self.tensor_types.get(attention_nodes[0].inputs[0], "tensor<??xf32>")
        q_weights_type = self.tensor_types.get(attention_nodes[0].inputs[1], "tensor<??xf32>")
        k_weights_type = self.tensor_types.get(attention_nodes[1].inputs[1], "tensor<??xf32>")
        v_weights_type = self.tensor_types.get(attention_nodes[2].inputs[1], "tensor<??xf32>")
        output_type = self.tensor_types.get(attention_nodes[5].outputs[0], "tensor<??xf32>")
        
        # Heuristic to estimate number of heads - typically hidden_size / 64 or similar
        # We'll default to 8 which is common in many models
        num_heads = 8
        
        self.value_map[attention_nodes[5].outputs[0]] = output
        
        return (f"{output} = aurora.fused_attention({input_tensor}, {q_weights}, {k_weights}, {v_weights}) "
                f"{{ num_heads = {num_heads} }} : "
                f"({input_type}, {q_weights_type}, {k_weights_type}, {v_weights_type}) -> {output_type}")
    
    def _define_input_func_signature(self, graph: GraphInfo) -> str:
        """Define the MLIR function signature for the model inputs
        
        Args:
            graph: GraphInfo containing the model
            
        Returns:
            MLIR function signature
        """
        arg_strs = []
        for i, inp in enumerate(graph.inputs):
            mlir_type = self._get_tensor_type_str(inp)
            arg_name = f"%arg{i}"
            arg_strs.append(f"{arg_name}: {mlir_type}")
            
            # Add to value map for later reference
            self.value_map[inp.name] = arg_name
            self.tensor_types[inp.name] = mlir_type
        
        # Create output types
        output_types = []
        for out in graph.outputs:
            mlir_type = self._get_tensor_type_str(out)
            output_types.append(mlir_type)
            self.tensor_types[out.name] = mlir_type
        
        return f"func.func @main({', '.join(arg_strs)}) -> ({', '.join(output_types)})"

    def _define_output_return(self, graph: GraphInfo) -> str:
        """Define the MLIR return statement for the model outputs
        
        Args:
            graph: GraphInfo containing the model
            
        Returns:
            MLIR return statement
        """
        output_vals = []
        for out in graph.outputs:
            output_vals.append(self._get_mlir_value(out.name))
        
        return f"return {', '.join(output_vals)} : {', '.join(self.tensor_types[out.name] for out in graph.outputs)}"
    
    def _analyze_and_prep_graph(self, graph: GraphInfo):
        """Analyze the graph and prepare for conversion
        
        Args:
            graph: GraphInfo to analyze
        """
        # Analyze constant tensors and add them to the value map
        for name, tensor in graph.tensors.items():
            if tensor.is_constant:
                # For simplicity, we won't actually convert constants to MLIR
                # In a real implementation, these would become dense/sparse attribute-based tensors
                const_name = f"%cst_{name.replace('.', '_')}"
                self.value_map[name] = const_name
                self.tensor_types[name] = self._get_tensor_type_str(tensor)
        
        # Add tensor types for non-constant tensors
        for op in graph.operations:
            for tensor_name in op.inputs + op.outputs:
                if tensor_name in graph.tensors and tensor_name not in self.tensor_types:
                    self.tensor_types[tensor_name] = self._get_tensor_type_str(graph.tensors[tensor_name])
    
    def convert(self, graph: GraphInfo) -> str:
        """Convert GraphInfo to MLIR text using the Aurora dialect
        
        Args:
            graph: GraphInfo to convert
            
        Returns:
            MLIR text representation
        """
        logger.info(f"Converting model {graph.name} to MLIR")
        
        # Reset state
        self.value_map = {}
        self.tensor_types = {}
        
        # Analyze the graph and prepare for conversion
        self._analyze_and_prep_graph(graph)
        
        # Start building the MLIR module
        mlir_lines = []
        mlir_lines.append("module {")
        
        # Define the function signature
        func_sig = self._define_input_func_signature(graph)
        mlir_lines.append(f"  {func_sig} {{")
        
        # Add constant declarations (simplified)
        # In a full implementation, these would be actual constants with values
        for name, value in self.value_map.items():
            if value.startswith("%cst_"):
                # Get the tensor type
                tensor_type = self.tensor_types.get(name, "tensor<??xf32>")
                mlir_lines.append(f"    {value} = \"dummy.constant\"() : () -> {tensor_type}")
        
        # Process attention patterns
        attention_patterns = self._detect_attention_patterns(graph)
        attention_outputs = set()
        
        for pattern in attention_patterns:
            # Extract all outputs from the pattern to skip these nodes later
            for node in pattern:
                for out in node.outputs:
                    attention_outputs.add(out)
            
            # Convert the pattern
            mlir_op = self._convert_attention(pattern, graph)
            mlir_lines.append(f"    {mlir_op}")
        
        # Process normal operations
        for node in graph.operations:
            # Skip nodes that are part of attention patterns
            if any(out in attention_outputs for out in node.outputs):
                continue
                
            # Skip unsupported operations
            if node.op_type not in self.op_map:
                logger.debug(f"Skipping unsupported operation: {node.op_type}")
                mlir_lines.append(f"    // Skipped unsupported op: {node.op_type} - {node.name}")
                
                # Create a placeholder for the output to allow further ops to reference it
                for output in node.outputs:
                    if output not in self.value_map:
                        output_value = self._get_mlir_value(output)
                        output_type = self.tensor_types.get(output, "tensor<??xf32>")
                        self.value_map[output] = output_value
                        
                        # Add a placeholder operation
                        mlir_lines.append(f"    {output_value} = \"dummy.placeholder\"() : () -> {output_type}")
                
                continue
            
            # Convert the operation to MLIR
            convert_func = self.op_map[node.op_type]
            mlir_op = convert_func(node, graph)
            mlir_lines.append(f"    {mlir_op}")
        
        # Define the return statement
        return_stmt = self._define_output_return(graph)
        mlir_lines.append(f"    {return_stmt}")
        
        # Close the function and module
        mlir_lines.append("  }")
        mlir_lines.append("}")
        
        logger.info("Conversion completed successfully")
        return "\n".join(mlir_lines)

    def convert_to_file(self, graph: GraphInfo, output_file: str):
        """Convert GraphInfo to MLIR and write to a file
        
        Args:
            graph: GraphInfo to convert
            output_file: Path to output file
        """
        mlir_text = self.convert(graph)
        
        logger.info(f"Writing MLIR to {output_file}")
        with open(output_file, 'w') as f:
            f.write(mlir_text)
        
        logger.info(f"Successfully wrote MLIR to {output_file}")


def example_conversion():
    """Example function to demonstrate ONNX to MLIR conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ONNX model to Aurora MLIR")
    parser.add_argument(
        "--model", 
        required=True,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--output",
        default="output.mlir",
        help="Output MLIR file (default: output.mlir)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load the ONNX model
        logger.info(f"Loading ONNX model from {args.model}")
        onnx_loader = ONNXLoader(verbose=args.verbose)
        graph = onnx_loader.load(args.model)
        
        # Convert to MLIR
        converter = ONNXToMLIRConverter(verbose=args.verbose)
        converter.convert_to_file(graph, args.output)
        
        logger.info(f"Successfully converted {args.model} to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(example_conversion())
