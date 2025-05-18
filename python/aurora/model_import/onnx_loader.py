#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX model loader for the Aurora AI Compiler

This module provides functionality to:
- Load ONNX model files
- Parse the ONNX graph structure
- Convert to an intermediate representation for the Aurora compiler
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import onnx
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aurora.onnx_loader")


class DataType(Enum):
    """Enumeration of supported data types"""
    FLOAT = "float"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    UNKNOWN = "unknown"


@dataclass
class TensorInfo:
    """Information about a tensor in the ONNX model"""
    name: str
    shape: List[Union[int, None]]  # None for dynamic dimensions
    data_type: DataType
    is_constant: bool = False
    constant_value: Optional[np.ndarray] = None


@dataclass
class AttributeInfo:
    """Information about an attribute in an ONNX node"""
    name: str
    type: str  # 'float', 'int', 'string', 'tensor', 'graph', etc.
    value: Any


@dataclass
class OperationNode:
    """Represents a node in the ONNX computational graph"""
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: List[AttributeInfo] = field(default_factory=list)
    domain: str = ""  # Domain for custom operations
    doc_string: str = ""


@dataclass
class GraphInfo:
    """Represents an ONNX graph with its operations and tensors"""
    name: str
    operations: List[OperationNode]
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    tensors: Dict[str, TensorInfo]
    model_path: str = ""
    opset_version: int = 0


class ONNXLoader:
    """Loads and parses ONNX models into an intermediate representation"""

    def __init__(self, verbose: bool = False):
        """Initialize the ONNX loader
        
        Args:
            verbose: Whether to print detailed debugging information
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)

    def _convert_onnx_type_to_aurora(self, onnx_type: int) -> DataType:
        """Convert ONNX data type to Aurora's data type enum
        
        Args:
            onnx_type: ONNX data type value
            
        Returns:
            Corresponding Aurora data type enum value
        """
        type_map = {
            onnx.TensorProto.FLOAT: DataType.FLOAT,
            onnx.TensorProto.FLOAT16: DataType.FLOAT16,
            onnx.TensorProto.INT8: DataType.INT8,
            onnx.TensorProto.INT32: DataType.INT32,
            onnx.TensorProto.INT64: DataType.INT64,
            onnx.TensorProto.BOOL: DataType.BOOL,
        }
        return type_map.get(onnx_type, DataType.UNKNOWN)

    def _extract_attribute_info(self, attribute) -> AttributeInfo:
        """Extract attribute information from an ONNX attribute
        
        Args:
            attribute: ONNX attribute object
            
        Returns:
            AttributeInfo object containing the parsed attribute data
        """
        attr_type = attribute.type
        attr_name = attribute.name
        
        # Extract the value based on attribute type
        if attr_type == onnx.AttributeProto.FLOAT:
            attr_value = attribute.f
            attr_type_str = "float"
        elif attr_type == onnx.AttributeProto.INT:
            attr_value = attribute.i
            attr_type_str = "int"
        elif attr_type == onnx.AttributeProto.STRING:
            attr_value = attribute.s.decode('utf-8') if isinstance(attribute.s, bytes) else attribute.s
            attr_type_str = "string"
        elif attr_type == onnx.AttributeProto.TENSOR:
            attr_value = onnx.numpy_helper.to_array(attribute.t)
            attr_type_str = "tensor"
        elif attr_type == onnx.AttributeProto.GRAPH:
            # For simplicity, we're not recursively parsing nested graphs
            attr_value = "nested_graph"
            attr_type_str = "graph"
        elif attr_type == onnx.AttributeProto.FLOATS:
            attr_value = list(attribute.floats)
            attr_type_str = "floats"
        elif attr_type == onnx.AttributeProto.INTS:
            attr_value = list(attribute.ints)
            attr_type_str = "ints"
        elif attr_type == onnx.AttributeProto.STRINGS:
            attr_value = [s.decode('utf-8') if isinstance(s, bytes) else s for s in attribute.strings]
            attr_type_str = "strings"
        else:
            attr_value = "unsupported_attribute_type"
            attr_type_str = "unknown"
            
        return AttributeInfo(name=attr_name, type=attr_type_str, value=attr_value)

    def _extract_tensor_info(self, value_info) -> TensorInfo:
        """Extract tensor information from an ONNX ValueInfoProto
        
        Args:
            value_info: ONNX ValueInfoProto object
            
        Returns:
            TensorInfo object containing the parsed tensor information
        """
        name = value_info.name
        
        # Extract type information
        type_proto = value_info.type
        if type_proto.HasField("tensor_type"):
            tensor_type = type_proto.tensor_type
            
            # Data type
            data_type = self._convert_onnx_type_to_aurora(tensor_type.elem_type)
            
            # Shape information
            shape = []
            if tensor_type.HasField("shape"):
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(int(dim.dim_value))
                    else:
                        # Dynamic dimension
                        shape.append(None)
            
            return TensorInfo(name=name, shape=shape, data_type=data_type)
        else:
            # For sequence or map types (not fully supported yet)
            return TensorInfo(name=name, shape=[], data_type=DataType.UNKNOWN)

    def _extract_constant_tensors(self, model: onnx.ModelProto) -> Dict[str, TensorInfo]:
        """Extract constant tensors from the ONNX model
        
        Args:
            model: ONNX model
            
        Returns:
            Dictionary mapping tensor names to TensorInfo objects
        """
        constant_tensors = {}
        
        # Process initializers (they define constant values)
        for initializer in model.graph.initializer:
            name = initializer.name
            numpy_array = onnx.numpy_helper.to_array(initializer)
            data_type = self._convert_onnx_type_to_aurora(initializer.data_type)
            
            tensor_info = TensorInfo(
                name=name,
                shape=list(numpy_array.shape),
                data_type=data_type,
                is_constant=True,
                constant_value=numpy_array
            )
            
            constant_tensors[name] = tensor_info
            
        # Also handle Constant nodes
        for node in model.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        tensor = attr.t
                        name = node.output[0]
                        numpy_array = onnx.numpy_helper.to_array(tensor)
                        data_type = self._convert_onnx_type_to_aurora(tensor.data_type)
                        
                        tensor_info = TensorInfo(
                            name=name,
                            shape=list(numpy_array.shape),
                            data_type=data_type,
                            is_constant=True,
                            constant_value=numpy_array
                        )
                        
                        constant_tensors[name] = tensor_info
                        
        return constant_tensors

    def _build_operation_nodes(self, model: onnx.ModelProto) -> List[OperationNode]:
        """Build operation nodes from ONNX model nodes
        
        Args:
            model: ONNX model
            
        Returns:
            List of OperationNode objects representing the computational graph
        """
        operation_nodes = []
        
        for node in model.graph.node:
            attributes = []
            for attr in node.attribute:
                attr_info = self._extract_attribute_info(attr)
                attributes.append(attr_info)
            
            # Generate a name for the node if it doesn't have one
            node_name = node.name
            if not node_name:
                node_name = f"{node.op_type}_{len(operation_nodes)}"
            
            op_node = OperationNode(
                name=node_name,
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attributes,
                domain=node.domain,
                doc_string=node.doc_string
            )
            
            operation_nodes.append(op_node)
            
        return operation_nodes

    def _build_tensor_map(self, model: onnx.ModelProto, constant_tensors: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        """Build a map of all tensors in the model
        
        Args:
            model: ONNX model
            constant_tensors: Dictionary of already extracted constant tensors
            
        Returns:
            Dictionary mapping tensor names to TensorInfo objects
        """
        tensor_map = dict(constant_tensors)  # Start with constant tensors
        
        # Add input tensors
        for input_info in model.graph.input:
            # Skip inputs that are actually initializers (constants)
            if input_info.name not in tensor_map:
                tensor_info = self._extract_tensor_info(input_info)
                tensor_map[input_info.name] = tensor_info
        
        # Add output tensors
        for output_info in model.graph.output:
            tensor_info = self._extract_tensor_info(output_info)
            tensor_map[output_info.name] = tensor_info
            
        # Add intermediate value_info tensors
        for value_info in model.graph.value_info:
            tensor_info = self._extract_tensor_info(value_info)
            tensor_map[value_info.name] = tensor_info
            
        # Infer additional tensors from node outputs (might have incomplete information)
        for node in model.graph.node:
            for output_name in node.output:
                if output_name not in tensor_map:
                    tensor_map[output_name] = TensorInfo(
                        name=output_name,
                        shape=[],  # Unknown shape
                        data_type=DataType.UNKNOWN
                    )
        
        return tensor_map

    def load(self, model_path: str) -> GraphInfo:
        """Load an ONNX model and convert it to Aurora's intermediate representation
        
        Args:
            model_path: Path to the ONNX model file
            
        Returns:
            GraphInfo object containing the parsed model information
        """
        logger.info(f"Loading ONNX model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        try:
            # Load the model
            model = onnx.load(model_path)
            
            # Basic model validation
            onnx.checker.check_model(model)
            logger.debug("ONNX model validation passed")
            
            # Extract model information
            model_opset = model.opset_import[0].version if model.opset_import else 0
            logger.info(f"Model opset version: {model_opset}")
            
            # Extract constant tensors
            constant_tensors = self._extract_constant_tensors(model)
            logger.debug(f"Extracted {len(constant_tensors)} constant tensors")
            
            # Build operation nodes
            operations = self._build_operation_nodes(model)
            logger.debug(f"Extracted {len(operations)} operation nodes")
            
            # Build complete tensor map
            tensor_map = self._build_tensor_map(model, constant_tensors)
            
            # Get input and output tensors
            input_tensors = [
                tensor_map[input_info.name] 
                for input_info in model.graph.input
                if input_info.name not in constant_tensors  # Skip initializers
            ]
            
            output_tensors = [
                tensor_map[output_info.name] 
                for output_info in model.graph.output
            ]
            
            # Create graph info
            graph_name = model.graph.name or os.path.basename(model_path)
            graph_info = GraphInfo(
                name=graph_name,
                operations=operations,
                inputs=input_tensors,
                outputs=output_tensors,
                tensors=tensor_map,
                model_path=model_path,
                opset_version=model_opset
            )
            
            logger.info(f"Successfully loaded model with {len(operations)} operations, "
                       f"{len(input_tensors)} inputs, and {len(output_tensors)} outputs")
            
            return graph_info
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def display_graph_summary(self, graph: GraphInfo):
        """Display a summary of the graph structure
        
        Args:
            graph: GraphInfo object to display
        """
        print(f"\n==== ONNX Model: {graph.name} ====")
        print(f"Opset version: {graph.opset_version}")
        print(f"Source: {graph.model_path}")
        
        print("\n-- Model Inputs --")
        for inp in graph.inputs:
            print(f"  • {inp.name}: shape={inp.shape}, type={inp.data_type.value}")
        
        print("\n-- Model Outputs --")
        for out in graph.outputs:
            print(f"  • {out.name}: shape={out.shape}, type={out.data_type.value}")
        
        print("\n-- Operations --")
        op_types = {}
        for op in graph.operations:
            op_types[op.op_type] = op_types.get(op.op_type, 0) + 1
        
        for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {op_type}: {count}")
        
        print("\n-- Constants --")
        constant_count = sum(1 for tensor in graph.tensors.values() if tensor.is_constant)
        print(f"  • {constant_count} constant tensors")
        
        print("\n-- Graph Structure --")
        for i, op in enumerate(graph.operations[:20]):  # Limit to first 20 ops for brevity
            inputs_str = ", ".join(op.inputs)
            outputs_str = ", ".join(op.outputs)
            attrs_str = ", ".join(f"{attr.name}={attr.value}" for attr in op.attributes[:3])  # Limited attributes
            
            print(f"  {i:3d}. {op.op_type} - Inputs: {inputs_str[:60]}{'...' if len(inputs_str) > 60 else ''}")
            print(f"       Outputs: {outputs_str[:60]}{'...' if len(outputs_str) > 60 else ''}")
            if attrs_str:
                print(f"       Attrs: {attrs_str[:60]}{'...' if len(attrs_str) > 60 else ''}")
        
        if len(graph.operations) > 20:
            print(f"  ... {len(graph.operations) - 20} more operations ...")


def test_resnet50():
    """Test function to load and analyze a ResNet50 ONNX model"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ONNX loader with ResNet50")
    parser.add_argument(
        "--model", 
        default="resnet50.onnx", 
        help="Path to ResNet50 ONNX model file (default: resnet50.onnx)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Testing ONNX loader with {args.model}")
        
        loader = ONNXLoader(verbose=args.verbose)
        graph = loader.load(args.model)
        
        loader.display_graph_summary(graph)
        
        logger.info("Test completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(test_resnet50())
