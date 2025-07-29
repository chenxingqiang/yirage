"""
YICA-Yirage: AI Computing Optimization Framework for In-Memory Computing Architecture
"""

__version__ = "1.0.1"

# Try to import optional dependencies gracefully
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Core Python modules (always available)
from .version import __version__
from .global_config import global_config
from .graph_dataset import graph_dataset
from .utils import *

# Import main modules with error handling
try:
    from . import yica_advanced
    YICA_ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: yica_advanced not available: {e}")
    YICA_ADVANCED_AVAILABLE = False

try:
    from . import yica_performance_monitor
    YICA_MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: yica_performance_monitor not available: {e}")
    YICA_MONITOR_AVAILABLE = False

try:
    from . import yica_optimizer
    YICA_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: yica_optimizer not available: {e}")
    YICA_OPTIMIZER_AVAILABLE = False

# Import other optional modules
optional_modules = [
    'yica_auto_tuner', 'yica_distributed', 'yica_llama_optimizer',
    'yica_pytorch_backend', 'visualizer', 'profiler', 'triton_profiler'
]

for module_name in optional_modules:
    try:
        __import__(f'{__name__}.{module_name}')
    except ImportError:
        pass  # Silently skip unavailable modules

# Main API functions
def create_yica_optimizer(config=None):
    """Create a YICA optimizer instance"""
    if not YICA_OPTIMIZER_AVAILABLE:
        raise ImportError("yica_optimizer module is not available")
    return yica_optimizer.create_yica_optimizer(config)

def quick_analyze(model_path, optimization_level="O2"):
    """Quick analysis of a model"""
    if not YICA_ADVANCED_AVAILABLE:
        raise ImportError("yica_advanced module is not available")
    return yica_advanced.quick_analyze(model_path, optimization_level)

def create_performance_monitor(config=None):
    """Create a performance monitor instance"""
    if not YICA_MONITOR_AVAILABLE:
        raise ImportError("yica_performance_monitor module is not available")
    return yica_performance_monitor.YICAPerformanceMonitor(config or {})

# Configuration
def set_gpu_device_id(device_id: int):
    """Set GPU device ID"""
    global_config.gpu_device_id = device_id

def bypass_compile_errors(value: bool = True):
    """Bypass compile errors for testing"""
    global_config.bypass_compile_errors = value

# Version and availability info
def get_version_info():
    """Get version and availability information"""
    return {
        "version": __version__,
        "z3_available": Z3_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "numpy_available": NUMPY_AVAILABLE,
        "yica_optimizer_available": YICA_OPTIMIZER_AVAILABLE,
        "yica_monitor_available": YICA_MONITOR_AVAILABLE,
        "yica_advanced_available": YICA_ADVANCED_AVAILABLE,
    }

# Aliases for backward compatibility
__all__ = [
    "__version__",
    "create_yica_optimizer",
    "quick_analyze", 
    "create_performance_monitor",
    "set_gpu_device_id",
    "bypass_compile_errors",
    "get_version_info",
    "global_config",
    "graph_dataset",
]
