"""
YICA Auto Compiler
Automatic C++ extension compilation system
"""

import os
import sys
import subprocess
import tempfile
import shutil
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
import threading
from dataclasses import dataclass, asdict


@dataclass
class CompilationConfig:
    """Configuration for compilation"""
    extension_name: str
    source_files: List[str]
    include_dirs: List[str]
    library_dirs: List[str]
    libraries: List[str]
    extra_compile_args: List[str]
    extra_link_args: List[str]
    language: str = "c++"
    std: str = "c++17"
    optimization: str = "O2"
    debug: bool = False
    parallel_jobs: int = 0  # 0 = auto-detect


@dataclass 
class CompilationResult:
    """Result of compilation"""
    success: bool
    extension_name: str
    output_path: Optional[str]
    compilation_time: float
    error_message: Optional[str]
    warnings: List[str]
    cache_hit: bool


class YICAAutoCompiler:
    """Automatic C++ extension compiler for YICA"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or self._get_default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.compilation_cache: Dict[str, CompilationResult] = {}
        self.lock = threading.RLock()
        
        # Load cache from disk
        self._load_cache()
        
        # Detect build tools
        self.build_tools = self._detect_build_tools()
        
    def _get_default_cache_dir(self) -> str:
        """Get default cache directory"""
        import tempfile
        return os.path.join(tempfile.gettempdir(), "yica_compilation_cache")
    
    def _load_cache(self):
        """Load compilation cache from disk"""
        cache_file = self.cache_dir / "compilation_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Convert dict back to CompilationResult objects
                    for key, data in cache_data.items():
                        self.compilation_cache[key] = CompilationResult(**data)
            except Exception as e:
                warnings.warn(f"Failed to load compilation cache: {e}")
    
    def _save_cache(self):
        """Save compilation cache to disk"""
        cache_file = self.cache_dir / "compilation_cache.json"
        try:
            cache_data = {
                key: asdict(result) for key, result in self.compilation_cache.items()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save compilation cache: {e}")
    
    def _detect_build_tools(self) -> Dict[str, Any]:
        """Detect available build tools"""
        tools = {
            "cmake": self._check_command("cmake"),
            "make": self._check_command("make"),
            "ninja": self._check_command("ninja"),
            "gcc": self._check_command("gcc"),
            "g++": self._check_command("g++"),
            "clang": self._check_command("clang"),
            "clang++": self._check_command("clang++"),
            "nvcc": self._check_command("nvcc"),
            "python": sys.executable
        }
        
        # Detect Python development headers
        try:
            import sysconfig
            tools["python_include"] = sysconfig.get_path('include')
            tools["python_lib"] = sysconfig.get_config_var('LIBDIR')
        except:
            tools["python_include"] = None
            tools["python_lib"] = None
        
        return tools
    
    def _check_command(self, command: str) -> Optional[str]:
        """Check if a command is available"""
        try:
            result = subprocess.run([command, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return shutil.which(command)
        except:
            pass
        return None
    
    def _generate_cache_key(self, config: CompilationConfig) -> str:
        """Generate cache key for compilation configuration"""
        # Create hash from configuration
        config_str = json.dumps(asdict(config), sort_keys=True)
        
        # Add file modification times
        file_times = []
        for source_file in config.source_files:
            if os.path.exists(source_file):
                file_times.append(str(os.path.getmtime(source_file)))
        
        combined = config_str + "".join(file_times)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def compile_extension(self, config: CompilationConfig, 
                         force_recompile: bool = False) -> CompilationResult:
        """Compile C++ extension"""
        
        with self.lock:
            cache_key = self._generate_cache_key(config)
            
            # Check cache
            if not force_recompile and cache_key in self.compilation_cache:
                cached_result = self.compilation_cache[cache_key]
                if cached_result.success and os.path.exists(cached_result.output_path or ""):
                    print(f"✅ Using cached compilation for {config.extension_name}")
                    cached_result.cache_hit = True
                    return cached_result
            
            print(f"🔨 Compiling C++ extension: {config.extension_name}")
            start_time = time.time()
            
            try:
                result = self._compile_with_setuptools(config)
                result.compilation_time = time.time() - start_time
                result.cache_hit = False
                
                # Cache the result
                self.compilation_cache[cache_key] = result
                self._save_cache()
                
                if result.success:
                    print(f"✅ Successfully compiled {config.extension_name} in {result.compilation_time:.2f}s")
                else:
                    print(f"❌ Failed to compile {config.extension_name}: {result.error_message}")
                
                return result
                
            except Exception as e:
                error_result = CompilationResult(
                    success=False,
                    extension_name=config.extension_name,
                    output_path=None,
                    compilation_time=time.time() - start_time,
                    error_message=str(e),
                    warnings=[],
                    cache_hit=False
                )
                
                print(f"❌ Compilation error for {config.extension_name}: {e}")
                return error_result
    
    def _compile_with_setuptools(self, config: CompilationConfig) -> CompilationResult:
        """Compile using setuptools/distutils"""
        
        # Create temporary setup.py
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_py_content = self._generate_setup_py(config, temp_dir)
            setup_py_path = os.path.join(temp_dir, "setup.py")
            
            with open(setup_py_path, 'w') as f:
                f.write(setup_py_content)
            
            # Copy source files to temp directory
            for source_file in config.source_files:
                if os.path.exists(source_file):
                    shutil.copy2(source_file, temp_dir)
            
            # Run compilation
            cmd = [
                sys.executable, "setup.py", 
                "build_ext", "--inplace"
            ]
            
            if config.parallel_jobs > 0:
                cmd.extend(["-j", str(config.parallel_jobs)])
            
            try:
                result = subprocess.run(
                    cmd, 
                    cwd=temp_dir,
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minutes timeout
                )
                
                warnings_list = []
                if result.stderr:
                    warnings_list = [line.strip() for line in result.stderr.split('\n') 
                                   if line.strip() and 'warning' in line.lower()]
                
                if result.returncode == 0:
                    # Find compiled extension
                    output_path = self._find_compiled_extension(temp_dir, config.extension_name)
                    
                    if output_path:
                        # Move to final location
                        final_path = self._get_final_extension_path(config.extension_name)
                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                        shutil.copy2(output_path, final_path)
                        
                        return CompilationResult(
                            success=True,
                            extension_name=config.extension_name,
                            output_path=final_path,
                            compilation_time=0,  # Will be set by caller
                            error_message=None,
                            warnings=warnings_list,
                            cache_hit=False
                        )
                    else:
                        return CompilationResult(
                            success=False,
                            extension_name=config.extension_name,
                            output_path=None,
                            compilation_time=0,
                            error_message="Compiled extension not found",
                            warnings=warnings_list,
                            cache_hit=False
                        )
                else:
                    return CompilationResult(
                        success=False,
                        extension_name=config.extension_name,
                        output_path=None,
                        compilation_time=0,
                        error_message=result.stderr or "Compilation failed",
                        warnings=warnings_list,
                        cache_hit=False
                    )
                    
            except subprocess.TimeoutExpired:
                return CompilationResult(
                    success=False,
                    extension_name=config.extension_name,
                    output_path=None,
                    compilation_time=0,
                    error_message="Compilation timeout (5 minutes)",
                    warnings=[],
                    cache_hit=False
                )
    
    def _generate_setup_py(self, config: CompilationConfig, temp_dir: str) -> str:
        """Generate setup.py content for compilation"""
        
        # Get relative paths for source files
        source_files = []
        for source_file in config.source_files:
            if os.path.exists(source_file):
                filename = os.path.basename(source_file)
                source_files.append(filename)
        
        setup_py = f'''
import os
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Extension configuration
ext_modules = [
    Pybind11Extension(
        "{config.extension_name}",
        {source_files},
        include_dirs=[
            pybind11.get_include(),
            {config.include_dirs},
        ],
        library_dirs={config.library_dirs},
        libraries={config.libraries},
        language="{config.language}",
        cxx_std={config.std.replace("c++", "")},
        extra_compile_args={config.extra_compile_args + [f"-{config.optimization}"]},
        extra_link_args={config.extra_link_args},
    ),
]

setup(
    name="{config.extension_name}",
    ext_modules=ext_modules,
    cmdclass={{"build_ext": build_ext}},
    zip_safe=False,
    python_requires=">=3.7",
)
'''
        return setup_py
    
    def _find_compiled_extension(self, temp_dir: str, extension_name: str) -> Optional[str]:
        """Find compiled extension in temporary directory"""
        
        # Common extension suffixes
        suffixes = ['.so', '.pyd', '.dll']
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.startswith(extension_name):
                    for suffix in suffixes:
                        if file.endswith(suffix):
                            return os.path.join(root, file)
        
        return None
    
    def _get_final_extension_path(self, extension_name: str) -> str:
        """Get final path for compiled extension"""
        
        # Get site-packages directory
        import site
        site_packages = site.getsitepackages()[0]
        
        # Use .so extension on Unix, .pyd on Windows
        if sys.platform.startswith('win'):
            ext_suffix = '.pyd'
        else:
            ext_suffix = '.so'
        
        return os.path.join(site_packages, f"{extension_name}{ext_suffix}")
    
    def compile_yica_extensions(self, force_recompile: bool = False) -> Dict[str, CompilationResult]:
        """Compile all YICA extensions"""
        
        extensions_config = self._get_yica_extensions_config()
        results = {}
        
        for ext_name, config in extensions_config.items():
            result = self.compile_extension(config, force_recompile)
            results[ext_name] = result
        
        return results
    
    def _get_yica_extensions_config(self) -> Dict[str, CompilationConfig]:
        """Get configuration for YICA extensions"""
        
        # Base paths
        yirage_root = Path(__file__).parent.parent.parent
        include_dirs = [
            str(yirage_root / "include"),
            str(yirage_root / "include" / "yirage"),
        ]
        
        library_dirs = [
            str(yirage_root / "build" / "lib"),
        ]
        
        libraries = ["yirage_core"]
        
        extra_compile_args = [
            "-std=c++17",
            "-fPIC",
            "-DWITH_YICA",
        ]
        
        if self.build_tools.get("nvcc"):
            extra_compile_args.append("-DWITH_CUDA")
        
        extensions = {
            "yica_kernels": CompilationConfig(
                extension_name="yica_kernels",
                source_files=[
                    str(yirage_root / "python" / "yirage" / "_cython" / "yica_kernels.pyx"),
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_link_args=[],
            ),
            
            "yica_performance": CompilationConfig(
                extension_name="yica_performance",
                source_files=[
                    str(yirage_root / "src" / "yica" / "yica_performance_monitor.cc"),
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_link_args=[],
            ),
        }
        
        return extensions
    
    def get_compilation_status(self) -> Dict[str, Any]:
        """Get compilation status"""
        
        with self.lock:
            total_extensions = len(self.compilation_cache)
            successful_compilations = sum(1 for result in self.compilation_cache.values() 
                                        if result.success)
            
            return {
                "total_extensions": total_extensions,
                "successful_compilations": successful_compilations,
                "failed_compilations": total_extensions - successful_compilations,
                "cache_size": len(self.compilation_cache),
                "build_tools": self.build_tools,
                "cache_dir": str(self.cache_dir)
            }
    
    def clear_cache(self):
        """Clear compilation cache"""
        with self.lock:
            self.compilation_cache.clear()
            cache_file = self.cache_dir / "compilation_cache.json"
            if cache_file.exists():
                cache_file.unlink()
            print("🗑️  Cleared compilation cache")
    
    def install_dependencies(self) -> bool:
        """Install compilation dependencies"""
        
        dependencies = [
            "setuptools",
            "wheel", 
            "pybind11",
            "cmake",
            "ninja"
        ]
        
        print("📦 Installing compilation dependencies...")
        
        for dep in dependencies:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"✅ Installed {dep}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {dep}: {e}")
                return False
        
        # Update build tools detection
        self.build_tools = self._detect_build_tools()
        return True


# Global auto-compiler instance
_global_compiler: Optional[YICAAutoCompiler] = None
_compiler_lock = threading.Lock()


def get_auto_compiler() -> YICAAutoCompiler:
    """Get or create global auto-compiler instance"""
    global _global_compiler
    
    if _global_compiler is None:
        with _compiler_lock:
            if _global_compiler is None:
                _global_compiler = YICAAutoCompiler()
    
    return _global_compiler


# Convenience functions
def compile_yica_extensions(force_recompile: bool = False) -> Dict[str, CompilationResult]:
    """Compile all YICA extensions"""
    return get_auto_compiler().compile_yica_extensions(force_recompile)


def compile_extension(config: CompilationConfig, force_recompile: bool = False) -> CompilationResult:
    """Compile a single extension"""
    return get_auto_compiler().compile_extension(config, force_recompile)


def get_compilation_status() -> Dict[str, Any]:
    """Get compilation status"""
    return get_auto_compiler().get_compilation_status()


def install_build_dependencies() -> bool:
    """Install build dependencies"""
    return get_auto_compiler().install_dependencies()


def clear_compilation_cache():
    """Clear compilation cache"""
    get_auto_compiler().clear_cache()


# Export main interfaces
__all__ = [
    "YICAAutoCompiler",
    "CompilationConfig",
    "CompilationResult",
    "get_auto_compiler",
    "compile_yica_extensions",
    "compile_extension", 
    "get_compilation_status",
    "install_build_dependencies",
    "clear_compilation_cache"
] 