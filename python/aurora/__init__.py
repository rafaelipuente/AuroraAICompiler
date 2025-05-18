"""
Aurora AI Compiler Python Package

This package provides Python bindings and utilities for working with the 
AuroraAICompiler, enabling model import, compilation, and execution.
"""

__version__ = "0.1.0"

from .compiler import compile_model, optimize_model, save_model
from .runtime import load_model, execute_model, benchmark_model

__all__ = [
    "compile_model",
    "optimize_model",
    "save_model",
    "load_model",
    "execute_model",
    "benchmark_model",
]
