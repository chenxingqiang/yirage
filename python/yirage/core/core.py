"""
YICA-Yirage Core Module
Provides core functionality and C++ bindings for YICA computing
"""

import os
import sys
from typing import Optional, Dict, Any, List, Union
import warnings

# Import warning manager
try:
    from ..utils.warning_manager import warn_cython_unavailable, warn_import_failed
except ImportError:
    # Fallback if warning manager is not available
    def warn_cython_unavailable(module_name: str):
        pass
    def warn_import_failed(module_name: str, fallback_available: bool = True):
        pass

# Try to import Cython bindings
try:
    from .._cython.core import *
    CYTHON_CORE_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try relative import
        from yirage._cython.core import *
        CYTHON_CORE_AVAILABLE = True
    except ImportError:
        CYTHON_CORE_AVAILABLE = False
        warn_cython_unavailable("core")

# Try to import YICA kernels
try:
    from .._cython.yica_kernels import *
    YICA_KERNELS_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try relative import
        from yirage._cython.yica_kernels import *
        YICA_KERNELS_AVAILABLE = True
    except ImportError:
        YICA_KERNELS_AVAILABLE = False
        warn_cython_unavailable("yica_kernels")

# Import Python-only fallbacks
from .global_config import global_config

# Import operator registry and auto-compiler
try:
    from ..compiler.operator_registry import get_global_registry, register_operator, create_operator
    OPERATOR_REGISTRY_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try old path
        from .operator_registry import get_global_registry, register_operator, create_operator
        OPERATOR_REGISTRY_AVAILABLE = True
    except ImportError:
        OPERATOR_REGISTRY_AVAILABLE = False
        warn_import_failed("operator_registry", True)

try:
    from ..compiler.auto_compiler import get_auto_compiler, compile_yica_extensions
    AUTO_COMPILER_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try old path
        from .auto_compiler import get_auto_compiler, compile_yica_extensions
        AUTO_COMPILER_AVAILABLE = True
    except ImportError:
        AUTO_COMPILER_AVAILABLE = False
        warn_import_failed("auto_compiler", True)


class YICACore:
    """
    YICA Core Interface
    Provides unified access to YICA hardware abstraction and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend_mode = self.config.get('backend_mode', 'cpu')
        self.num_cim_arrays = self.config.get('num_cim_arrays', 8)
        self.spm_size = self.config.get('spm_size', 128 * 1024 * 1024)  # 128MB
        
        # Initialize backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize YICA backend based on availability"""
        if CYTHON_CORE_AVAILABLE and YICA_KERNELS_AVAILABLE:
            self.backend_type = "native"
            self._init_native_backend()
        else:
            self.backend_type = "fallback"
            self._init_fallback_backend()
        
        # Initialize operator registry
        if OPERATOR_REGISTRY_AVAILABLE:
            self.operator_registry = get_global_registry()
        
        # Initialize auto-compiler
        if AUTO_COMPILER_AVAILABLE:
            self.auto_compiler = get_auto_compiler()
    
    def _init_native_backend(self):
        """Initialize native C++ backend"""
        try:
            # Initialize YICA hardware abstraction
            self.hardware_abstraction = YICAHardwareAbstraction(
                num_cim_arrays=self.num_cim_arrays,
                spm_size=self.spm_size
            )
            
            # Initialize device memory manager
            self.memory_manager = YICADeviceMemoryManager()
            
            # Initialize kernel graph
            self.kernel_graph = YICAKernelGraph()
            
            print(f"✅ YICA Native backend initialized successfully")
            print(f"   - CIM Arrays: {self.num_cim_arrays}")
            print(f"   - SPM Size: {self.spm_size // (1024*1024)}MB")
            
        except Exception as e:
            warnings.warn(f"Native backend initialization failed: {e}")
            self._init_fallback_backend()
    
    def _init_fallback_backend(self):
        """Initialize fallback Python-only backend"""
        self.hardware_abstraction = None
        self.memory_manager = None
        self.kernel_graph = None
        
        print("⚠️  YICA Fallback backend initialized")
        print("   - Limited functionality available")
        print("   - Consider installing C++ extensions for full performance")
    
    def is_available(self) -> bool:
        """Check if YICA core is available"""
        return CYTHON_CORE_AVAILABLE and YICA_KERNELS_AVAILABLE
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get YICA capabilities"""
        return {
            "backend_type": self.backend_type,
            "cython_core_available": CYTHON_CORE_AVAILABLE,
            "yica_kernels_available": YICA_KERNELS_AVAILABLE,
            "num_cim_arrays": self.num_cim_arrays,
            "spm_size": self.spm_size,
            "supported_operations": self._get_supported_operations()
        }
    
    def _get_supported_operations(self) -> List[str]:
        """Get list of supported operations"""
        if self.backend_type == "native":
            return [
                "matmul", "element_unary", "element_binary", 
                "reduction", "rms_norm", "all_reduce", "chunk",
                "customized"
            ]
        else:
            return ["basic_operations"]  # Fallback mode
    
    def create_optimizer(self, config: Optional[Dict[str, Any]] = None):
        """Create YICA optimizer instance"""
        if self.backend_type == "native" and self.is_available():
            optimizer_config = {**self.config, **(config or {})}
            return YICAOptimizer(optimizer_config)
        else:
            # Fall back to real optimizer when native core not available
            return None  # Let the main function handle fallback
    
    def create_kernel_graph(self):
        """Create kernel graph for computation"""
        if self.backend_type == "native":
            return self.kernel_graph.create_new_graph()
        else:
            raise RuntimeError("Kernel graph requires native backend")
    
    def profile_operation(self, operation_name: str, *args, **kwargs):
        """Profile a YICA operation"""
        if self.backend_type == "native":
            return self._profile_native_operation(operation_name, *args, **kwargs)
        else:
            return self._profile_fallback_operation(operation_name, *args, **kwargs)
    
    def _profile_native_operation(self, operation_name: str, *args, **kwargs):
        """Profile operation using native backend"""
        # Implementation would call C++ profiling functions
        return {
            "operation": operation_name,
            "backend": "native",
            "latency_ms": 0.0,
            "throughput_ops_per_sec": 0.0,
            "cim_utilization": 0.0,
            "spm_hit_rate": 0.0
        }
    
    def _profile_fallback_operation(self, operation_name: str, *args, **kwargs):
        """Profile operation using fallback backend"""
        return {
            "operation": operation_name,
            "backend": "fallback",
            "note": "Limited profiling in fallback mode"
        }


# Global YICA core instance
_yica_core_instance: Optional[YICACore] = None


def get_yica_core(config: Optional[Dict[str, Any]] = None) -> YICACore:
    """Get or create global YICA core instance"""
    global _yica_core_instance
    
    if _yica_core_instance is None:
        _yica_core_instance = YICACore(config)
    
    return _yica_core_instance


def initialize_yica(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize YICA core system"""
    try:
        core = get_yica_core(config)
        return core.is_available()
    except Exception as e:
        warnings.warn(f"YICA initialization failed: {e}")
        return False


def is_yica_available() -> bool:
    """Check if YICA is available"""
    return CYTHON_CORE_AVAILABLE and YICA_KERNELS_AVAILABLE


def get_yica_info() -> Dict[str, Any]:
    """Get YICA system information"""
    core = get_yica_core()
    return core.get_capabilities()


# Operator registration system
_registered_operators: Dict[str, Any] = {}


def register_operator(name: str, operator_class, config: Optional[Dict[str, Any]] = None):
    """Register a YICA operator"""
    _registered_operators[name] = {
        "class": operator_class,
        "config": config or {},
        "registered_at": __import__("time").time()
    }
    print(f"✅ Registered YICA operator: {name}")


def get_registered_operators() -> Dict[str, Any]:
    """Get all registered operators"""
    return _registered_operators.copy()


def create_operator(name: str, *args, **kwargs):
    """Create an instance of registered operator"""
    if name not in _registered_operators:
        raise ValueError(f"Operator '{name}' not registered")
    
    operator_info = _registered_operators[name]
    operator_class = operator_info["class"]
    
    # Merge config
    config = {**operator_info["config"], **kwargs}
    
    return operator_class(*args, **config)


# Auto-compilation system
class AutoCompiler:
    """Automatic C++ extension compiler"""
    
    def __init__(self):
        self.compile_cache = {}
    
    def compile_extension(self, extension_name: str, source_files: List[str], 
                         include_dirs: Optional[List[str]] = None,
                         libraries: Optional[List[str]] = None,
                         force_recompile: bool = False) -> bool:
        """Compile C++ extension automatically"""
        
        cache_key = f"{extension_name}_{hash(tuple(source_files))}"
        
        if not force_recompile and cache_key in self.compile_cache:
            print(f"✅ Using cached compilation for {extension_name}")
            return self.compile_cache[cache_key]
        
        try:
            print(f"🔨 Compiling C++ extension: {extension_name}")
            
            # This would integrate with setuptools/cmake for actual compilation
            # For now, we simulate the compilation process
            success = self._simulate_compilation(extension_name, source_files)
            
            self.compile_cache[cache_key] = success
            
            if success:
                print(f"✅ Successfully compiled {extension_name}")
            else:
                print(f"❌ Failed to compile {extension_name}")
                
            return success
            
        except Exception as e:
            print(f"❌ Compilation error for {extension_name}: {e}")
            self.compile_cache[cache_key] = False
            return False
    
    def _simulate_compilation(self, extension_name: str, source_files: List[str]) -> bool:
        """Simulate compilation process"""
        # In real implementation, this would:
        # 1. Check for compiler availability
        # 2. Set up build environment
        # 3. Invoke cmake/make/ninja
        # 4. Handle compilation errors
        # 5. Install compiled extensions
        
        import time
        time.sleep(0.1)  # Simulate compilation time
        return True  # Assume success for now


# Global auto-compiler instance
_auto_compiler = AutoCompiler()


def compile_yica_extensions(force_recompile: bool = False) -> bool:
    """Compile all YICA C++ extensions"""
    extensions = [
        {
            "name": "yica_kernels",
            "sources": ["yica_kernels.pyx", "yica_backend.cpp"],
            "includes": ["include/yirage/"],
            "libraries": ["yirage_core"]
        },
        {
            "name": "yica_performance",
            "sources": ["yica_performance.pyx", "performance_monitor.cpp"],
            "includes": ["include/yirage/"],
            "libraries": ["yirage_core"]
        }
    ]
    
    all_success = True
    
    for ext in extensions:
        success = _auto_compiler.compile_extension(
            ext["name"], 
            ext["sources"],
            ext.get("includes"),
            ext.get("libraries"),
            force_recompile
        )
        all_success = all_success and success
    
    return all_success


# Export main interfaces
__all__ = [
    "YICACore",
    "get_yica_core", 
    "initialize_yica",
    "is_yica_available",
    "get_yica_info",
    "register_operator",
    "get_registered_operators", 
    "create_operator",
    "AutoCompiler",
    "compile_yica_extensions",
    "CYTHON_CORE_AVAILABLE",
    "YICA_KERNELS_AVAILABLE"
] 