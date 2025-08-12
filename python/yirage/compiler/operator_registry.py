"""
YICA Operator Registry
Automatic operator discovery and registration system
"""

import inspect
import importlib
import pkgutil
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass
import warnings
import time
import threading


@dataclass
class OperatorInfo:
    """Operator information"""
    name: str
    operator_class: Type
    config: Dict[str, Any]
    module_name: str
    description: str
    version: str
    registered_at: float
    dependencies: List[str]
    is_yica_native: bool


class YICAOperatorRegistry:
    """YICA Operator Registry for automatic operator discovery and registration"""
    
    def __init__(self):
        self._operators: Dict[str, OperatorInfo] = {}
        self._lock = threading.RLock()
        self._auto_discovery_enabled = True
        self._discovery_paths = [
            'yirage._cython.yica_kernels',
            'yirage.yica_advanced',
            'yirage.yica_optimizer',
            'yirage.operators',  # Future operator modules
        ]
    
    def register_operator(self, 
                         name: str, 
                         operator_class: Type,
                         config: Optional[Dict[str, Any]] = None,
                         description: str = "",
                         version: str = "1.0.0",
                         dependencies: Optional[List[str]] = None,
                         is_yica_native: bool = False,
                         force_overwrite: bool = False) -> bool:
        """Register an operator"""
        
        with self._lock:
            if name in self._operators and not force_overwrite:
                warnings.warn(f"Operator '{name}' already registered. Use force_overwrite=True to replace.")
                return False
            
            # Validate operator class
            if not self._validate_operator(operator_class):
                warnings.warn(f"Operator class {operator_class} failed validation")
                return False
            
            operator_info = OperatorInfo(
                name=name,
                operator_class=operator_class,
                config=config or {},
                module_name=operator_class.__module__,
                description=description,
                version=version,
                registered_at=time.time(),
                dependencies=dependencies or [],
                is_yica_native=is_yica_native
            )
            
            self._operators[name] = operator_info
            print(f"✅ Registered YICA operator: {name} (v{version})")
            return True
    
    def _validate_operator(self, operator_class: Type) -> bool:
        """Validate that operator class meets requirements"""
        
        # Check if it's a class
        if not inspect.isclass(operator_class):
            return False
        
        # Check for required methods (flexible validation)
        required_methods = ['__init__']
        optional_methods = ['forward', 'profile', 'optimize', 'configure']
        
        for method in required_methods:
            if not hasattr(operator_class, method):
                return False
        
        return True
    
    def unregister_operator(self, name: str) -> bool:
        """Unregister an operator"""
        with self._lock:
            if name in self._operators:
                del self._operators[name]
                print(f"🗑️  Unregistered operator: {name}")
                return True
            return False
    
    def get_operator(self, name: str) -> Optional[OperatorInfo]:
        """Get operator information"""
        return self._operators.get(name)
    
    def list_operators(self, filter_yica_native: Optional[bool] = None) -> Dict[str, OperatorInfo]:
        """List all registered operators"""
        if filter_yica_native is None:
            return self._operators.copy()
        
        return {
            name: info for name, info in self._operators.items()
            if info.is_yica_native == filter_yica_native
        }
    
    def create_operator(self, name: str, *args, **kwargs):
        """Create an instance of registered operator"""
        operator_info = self.get_operator(name)
        if not operator_info:
            raise ValueError(f"Operator '{name}' not registered")
        
        # Check dependencies
        if not self._check_dependencies(operator_info.dependencies):
            raise ImportError(f"Dependencies not met for operator '{name}': {operator_info.dependencies}")
        
        # Merge configuration
        config = {**operator_info.config, **kwargs}
        
        try:
            return operator_info.operator_class(*args, **config)
        except Exception as e:
            raise RuntimeError(f"Failed to create operator '{name}': {e}")
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if dependencies are available"""
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                return False
        return True
    
    def auto_discover_operators(self, search_paths: Optional[List[str]] = None) -> int:
        """Automatically discover and register operators"""
        if not self._auto_discovery_enabled:
            return 0
        
        search_paths = search_paths or self._discovery_paths
        discovered_count = 0
        
        for path in search_paths:
            try:
                discovered_count += self._discover_operators_in_module(path)
            except Exception as e:
                warnings.warn(f"Failed to discover operators in {path}: {e}")
        
        return discovered_count
    
    def _discover_operators_in_module(self, module_path: str) -> int:
        """Discover operators in a specific module"""
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            return 0
        
        discovered_count = 0
        
        # Look for classes that might be operators
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if self._is_potential_operator(name, obj):
                operator_name = self._generate_operator_name(name)
                
                # Auto-detect if it's YICA native
                is_yica_native = self._is_yica_native_operator(obj)
                
                success = self.register_operator(
                    name=operator_name,
                    operator_class=obj,
                    description=f"Auto-discovered from {module_path}",
                    is_yica_native=is_yica_native,
                    force_overwrite=False
                )
                
                if success:
                    discovered_count += 1
        
        return discovered_count
    
    def _is_potential_operator(self, name: str, obj: Type) -> bool:
        """Check if a class is potentially an operator"""
        # Look for operator naming patterns
        operator_patterns = [
            'Op', 'Operator', 'Kernel', 'Layer', 'Function'
        ]
        
        # Check name patterns
        for pattern in operator_patterns:
            if name.endswith(pattern):
                return True
        
        # Check if it has operator-like methods
        operator_methods = ['forward', 'execute', 'apply', 'compute']
        for method in operator_methods:
            if hasattr(obj, method):
                return True
        
        return False
    
    def _is_yica_native_operator(self, obj: Type) -> bool:
        """Check if operator is YICA native"""
        # Check module path
        if 'yica' in obj.__module__.lower():
            return True
        
        # Check class name
        if 'YICA' in obj.__name__:
            return True
        
        # Check for YICA-specific methods
        yica_methods = [
            'optimize_for_cim_arrays',
            'enable_spm_staging',
            'configure_yica_hardware'
        ]
        
        for method in yica_methods:
            if hasattr(obj, method):
                return True
        
        return False
    
    def _generate_operator_name(self, class_name: str) -> str:
        """Generate operator name from class name"""
        # Convert CamelCase to snake_case
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        
        # Remove common suffixes
        suffixes = ['_op', '_operator', '_kernel', '_layer']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        return name
    
    def get_operator_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            total_operators = len(self._operators)
            yica_native_count = sum(1 for op in self._operators.values() if op.is_yica_native)
            
            # Group by module
            module_stats = {}
            for op in self._operators.values():
                module = op.module_name
                if module not in module_stats:
                    module_stats[module] = 0
                module_stats[module] += 1
            
            return {
                "total_operators": total_operators,
                "yica_native_operators": yica_native_count,
                "python_operators": total_operators - yica_native_count,
                "operators_by_module": module_stats,
                "auto_discovery_enabled": self._auto_discovery_enabled
            }
    
    def enable_auto_discovery(self, enabled: bool = True):
        """Enable or disable automatic operator discovery"""
        self._auto_discovery_enabled = enabled
        print(f"Auto-discovery {'enabled' if enabled else 'disabled'}")
    
    def add_discovery_path(self, path: str):
        """Add a path for automatic discovery"""
        if path not in self._discovery_paths:
            self._discovery_paths.append(path)
            print(f"Added discovery path: {path}")
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry for serialization"""
        exported = {}
        for name, info in self._operators.items():
            exported[name] = {
                "module_name": info.module_name,
                "class_name": info.operator_class.__name__,
                "config": info.config,
                "description": info.description,
                "version": info.version,
                "dependencies": info.dependencies,
                "is_yica_native": info.is_yica_native,
                "registered_at": info.registered_at
            }
        return exported


# Global registry instance
_global_registry: Optional[YICAOperatorRegistry] = None
_registry_lock = threading.Lock()


def get_global_registry() -> YICAOperatorRegistry:
    """Get or create the global operator registry"""
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = YICAOperatorRegistry()
                # Auto-discover operators on first access
                try:
                    discovered = _global_registry.auto_discover_operators()
                    if discovered > 0:
                        print(f"🔍 Auto-discovered {discovered} operators")
                except Exception as e:
                    warnings.warn(f"Auto-discovery failed: {e}")
    
    return _global_registry


# Convenience functions
def register_operator(name: str, operator_class: Type, **kwargs) -> bool:
    """Register an operator in the global registry"""
    return get_global_registry().register_operator(name, operator_class, **kwargs)


def create_operator(name: str, *args, **kwargs):
    """Create an operator from the global registry"""
    return get_global_registry().create_operator(name, *args, **kwargs)


def list_operators(**kwargs) -> Dict[str, OperatorInfo]:
    """List operators from the global registry"""
    return get_global_registry().list_operators(**kwargs)


def get_operator_info(name: str) -> Optional[OperatorInfo]:
    """Get operator information from the global registry"""
    return get_global_registry().get_operator(name)


def discover_operators(search_paths: Optional[List[str]] = None) -> int:
    """Discover operators and register them"""
    return get_global_registry().auto_discover_operators(search_paths)


def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics"""
    return get_global_registry().get_operator_stats()


# Decorator for automatic registration
def yica_operator(name: Optional[str] = None, 
                  config: Optional[Dict[str, Any]] = None,
                  description: str = "",
                  version: str = "1.0.0",
                  dependencies: Optional[List[str]] = None,
                  is_yica_native: bool = False):
    """Decorator for automatic operator registration"""
    
    def decorator(cls):
        operator_name = name or get_global_registry()._generate_operator_name(cls.__name__)
        
        register_operator(
            name=operator_name,
            operator_class=cls,
            config=config,
            description=description,
            version=version,
            dependencies=dependencies,
            is_yica_native=is_yica_native
        )
        
        return cls
    
    return decorator


# Export main interfaces
__all__ = [
    "YICAOperatorRegistry",
    "OperatorInfo", 
    "get_global_registry",
    "register_operator",
    "create_operator",
    "list_operators",
    "get_operator_info",
    "discover_operators",
    "get_registry_stats",
    "yica_operator"
] 