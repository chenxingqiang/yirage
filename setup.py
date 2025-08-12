#!/usr/bin/env python3
"""
Enhanced YIRAGE Installation Script
Handles OpenMP, CUTLASS and other hard dependencies
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys
import platform

# Read version information - dynamically from version file
def get_version():
    version_file = os.path.join('python', 'yirage', 'version.py')
    
    # Method 1: Directly execute version file to get version
    try:
        version_globals = {}
        with open(version_file, 'r') as f:
            exec(f.read(), version_globals)
        
        if '__version__' in version_globals:
            version = version_globals['__version__']
            print(f"✅ Dynamically read version: {version} (source: {version_file})")
            return version
    except Exception as e:
        print(f"⚠️  Method 1 failed: {e}")
    
    # Method 2: Text parsing (backup)
    try:
        with open(version_file, 'r') as f:
            content = f.read()
            import re
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                version = match.group(1)
                print(f"✅ Parsed version: {version} (source: {version_file})")
                return version
    except Exception as e:
        print(f"⚠️  Method 2 failed: {e}")
    
    # Method 3: Try import (if in correct path)
    try:
        import sys
        sys.path.insert(0, 'python')
        from yirage.version import __version__
        print(f"✅ Imported version: {__version__} (source: module import)")
        return __version__
    except Exception as e:
        print(f"⚠️  Method 3 failed: {e}")
    
    print(f"❌ Unable to get version information, using default value")
    return "dev-unknown"

# Detect compilation environment
def detect_compile_env():
    env = {
        'has_cuda': False,
        'has_openmp': False,
        'cutlass_path': None,
        'json_path': None,
        'z3_path': None,
        'is_macos': platform.system() == 'Darwin',
        'is_linux': platform.system() == 'Linux',
    }
    
    # Check CUDA
    if os.path.exists('/usr/local/cuda') or os.environ.get('CUDA_HOME'):
        env['has_cuda'] = True
        print("✅ Detected CUDA environment")
    
    # Check dependency paths
    deps_dir = os.path.join(os.getcwd(), 'deps')
    
    if os.path.exists(os.path.join(deps_dir, 'cutlass', 'include')):
        env['cutlass_path'] = os.path.join(deps_dir, 'cutlass')
        print(f"✅ Found CUTLASS: {env['cutlass_path']}")
    
    if os.path.exists(os.path.join(deps_dir, 'json', 'include')):
        env['json_path'] = os.path.join(deps_dir, 'json')
        print(f"✅ Found nlohmann/json: {env['json_path']}")
    
    # Prioritize pip-installed Z3
    try:
        import z3
        print(f"✅ Found Z3 (pip): {z3.get_version_string()}")
        env['z3_pip'] = True
    except ImportError:
        env['z3_pip'] = False
        # Then check locally compiled Z3
        if os.path.exists(os.path.join(deps_dir, 'z3', 'install')):
            env['z3_path'] = os.path.join(deps_dir, 'z3', 'install')
            print(f"✅ Found Z3 (source): {env['z3_path']}")
        else:
            print("⚠️  Z3 not found, recommend running: pip install z3-solver")
    
    # Check OpenMP
    if env['is_macos']:
        # macOS uses libomp
        try:
            import subprocess
            result = subprocess.run(['brew', '--prefix', 'libomp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                env['has_openmp'] = True
                env['openmp_path'] = result.stdout.strip()
                print(f"✅ Found OpenMP (libomp): {env['openmp_path']}")
        except:
            pass
    else:
        # Linux usually has system OpenMP
        env['has_openmp'] = True
        print("✅ Assuming Linux system has OpenMP support")
    
    return env

# Build extension modules
def create_extensions(env):
    extensions = []
    
    # Basic include paths
    include_dirs = [
        'include',
        'python',
        pybind11.get_include(),
        '/opt/homebrew/include',  # Add homebrew include path for Z3
    ]
    
    # Add dependency include paths
    if env['cutlass_path']:
        include_dirs.append(os.path.join(env['cutlass_path'], 'include'))
    
    if env['json_path']:
        include_dirs.append(os.path.join(env['json_path'], 'include'))
    
    if env['z3_path']:
        include_dirs.extend([
            os.path.join(env['z3_path'], 'include'),
        ])
    
    # Compilation flags
    compile_args = ['-std=c++17', '-O3']
    link_args = []
    libraries = []
    library_dirs = []
    
    # OpenMP support
    if env['has_openmp']:
        if env['is_macos'] and 'openmp_path' in env:
            # macOS libomp
            compile_args.extend(['-Xpreprocessor', '-fopenmp'])
            include_dirs.append(os.path.join(env['openmp_path'], 'include'))
            library_dirs.append(os.path.join(env['openmp_path'], 'lib'))
            libraries.append('omp')
        else:
            # Linux OpenMP
            compile_args.append('-fopenmp')
            link_args.append('-fopenmp')
    
    # Z3 library (prioritize pip version, no manual linking needed)
    if env.get('z3_pip'):
        # pip-installed Z3 handles linking automatically
        print("✅ Using pip-installed Z3, no manual linking needed")
    elif env.get('z3_path'):
        # Use locally compiled Z3
        library_dirs.append(os.path.join(env['z3_path'], 'lib'))
        libraries.append('z3')
        include_dirs.append(os.path.join(env['z3_path'], 'include'))
        print("✅ Using locally compiled Z3")
    else:
        print("⚠️  Z3 not found, some features may not be available")
    
    # CUDA support (optional)
    if env['has_cuda']:
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        include_dirs.append(os.path.join(cuda_home, 'include'))
        library_dirs.append(os.path.join(cuda_home, 'lib64'))
        libraries.extend(['cuda', 'cudart', 'cublas'])
        compile_args.append('-DYICA_ENABLE_CUDA')
    else:
        compile_args.append('-DYICA_CPU_ONLY')
    
    # Create core extension
    try:
        core_extension = Pybind11Extension(
            "yirage._core",
            sources=[
                # Add key source files
                "src/base/layout.cc",
                "src/search/config.cc",
                "src/search/search.cc",
                # Can add more source files as needed
            ],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            cxx_std=17,
        )
        
        # Set compilation and linking parameters
        core_extension.extra_compile_args = compile_args
        core_extension.extra_link_args = link_args
        
        extensions.append(core_extension)
        print(f"✅ Created core extension module")
        
    except Exception as e:
        print(f"⚠️  Skipping C++ extension module: {e}")
    
    return extensions

# Main installation configuration
def main():
    print("🔧 Detecting compilation environment...")
    env = detect_compile_env()
    
    print("🔨 Creating extension modules...")
    extensions = create_extensions(env)
    
    # Basic dependencies
    install_requires = [
        "numpy>=1.19.0",
        "z3-solver>=4.8.0",
    ]
    
    # Z3 dependency handling
    if env.get('z3_pip'):
        # Already satisfied through pip installation, no need to add again
        print("✅ Z3 dependency already satisfied through pip")
    elif env.get('z3_path'):
        # Has locally compiled version, no need for pip version
        print("✅ Z3 dependency satisfied through local compilation")
    else:
        # Ensure Z3 dependency
        print("📦 Will install Z3 through pip")
    
    # PyTorch dependency (optional)
    try:
        import torch
        print(f"✅ Detected PyTorch {torch.__version__}")
    except ImportError:
        install_requires.append("torch>=1.12.0")
        print("📦 Will install PyTorch")
    
    setup(
        name="yica-yirage",
        version=get_version(),
        description="YICA-Yirage: AI Computing Optimization Framework (Enhanced Build)",
        long_description="YICA-Yirage with OpenMP, CUTLASS, and Z3 support",
        long_description_content_type="text/plain",
        author="YICA Team",
        author_email="contact@yica.ai",
        
        # Package configuration
        package_dir={"": "python"},
        packages=find_packages(where="python"),
        
        # C++ extensions
        ext_modules=extensions,
        cmdclass={"build_ext": build_ext},
        
        # Dependencies
        install_requires=install_requires,
        
        extras_require={
            "dev": [
                "pytest>=6.0",
                "pytest-cov>=3.0",
                "black>=21.0",
                "flake8>=3.8",
            ],
            "triton": [
                "triton>=2.0.0; sys_platform=='linux'",
            ],
            "full": [
                "torch>=1.12.0",
                "triton>=2.0.0; sys_platform=='linux'",
                "matplotlib>=3.0.0",
                "tqdm>=4.0.0",
            ],
        },
        
        python_requires=">=3.8",
        zip_safe=False,
        
        # Classifiers
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )

if __name__ == "__main__":
    main()
