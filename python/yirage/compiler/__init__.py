"""
YICA Compilation Module

Automatic compilation and operator registry system.
"""

from .auto_compiler import *
from .operator_registry import *

__all__ = [
    # Auto Compiler
    "YICAAutoCompiler",
    "CompilationConfig",
    "CompilationResult",
    "get_auto_compiler",
    "compile_yica_extensions",
    "compile_extension",
    "get_compilation_status",
    "install_build_dependencies",
    "clear_compilation_cache",
    
    # Operator Registry
    "YICAOperatorRegistry",
    "OperatorInfo",
    "get_global_registry",
    "register_operator",
    "create_operator",
    "list_operators",
    "get_operator_info",
    "discover_operators",
    "get_registry_stats",
    "yica_operator",
]
