# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from os import path
import sys
import sysconfig
from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:                                                
    from distutils.core import setup
    from distutils.extension import Extension                              
else:
    from setuptools import setup
    from setuptools.extension import Extension

def config_cython():
    """Configure Cython extensions with YICA hardware acceleration support"""
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        cython_path = path.join(path.dirname(__file__), "yirage/_cython")
        yirage_path = path.join(path.dirname(__file__), "..")
        
        # Enhanced include directories for YICA support
        include_dirs = [
            path.join(yirage_path, "include"),
            path.join(yirage_path, "include", "yirage", "yica"),  # YICA headers
            path.join(yirage_path, "deps", "json", "include"),
            path.join(yirage_path, "deps", "cutlass", "include"),
            "/usr/local/cuda/include"
        ]
        
        # Libraries for basic functionality (YICA disabled)
        libraries = [
            "yirage_runtime", 
            "z3"
        ]
        
        # Library directories
        library_dirs = [
            path.join(yirage_path, "build"),
            "/opt/homebrew/lib"
        ]
        
        # Compile flags for basic functionality
        extra_compile_args = [
            "-std=c++17",
            "-O3",                            # Optimization
            "-fPIC"
        ]
        
        # Enhanced link flags (macOS compatible)
        if sys.platform == 'darwin':
            extra_link_args = ["-fPIC"]
        else:
            extra_link_args = [
                "-fPIC",
                "-Wl,--no-undefined",  # Catch undefined symbols early
                "-Wl,--as-needed"      # Only link needed libraries
            ]
        
        # Process all .pyx files (with existence check for Phase 1)
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            
            # Skip problematic files in Phase 1
            if fn in ["core.pyx", "yica_kernels.pyx"]:
                print(f"⏳ Skipping {fn} (Phase 1 - will be implemented in Phase 2)")
                continue
            
            module_name = fn[:-4]
            print(f"🐍 Configuring Cython module: {module_name}")
            
            ret.append(Extension(
                f"yirage.{module_name}",
                [f"{cython_path}/{fn}"],
                include_dirs=include_dirs,
                libraries=libraries,
                library_dirs=library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++"
            ))
        
        # Enhanced Cython compiler directives
        compiler_directives = {
            "language_level": 3,
            "boundscheck": False,      # Disable bounds checking for performance
            "wraparound": False,       # Disable wraparound for performance  
            "initializedcheck": False, # Disable initialization checking
            "cdivision": True,         # Use C division semantics
            "embedsignature": True,    # Embed function signatures in docstrings
        }
        
        print(f"✅ Configured {len(ret)} Cython extensions")
        return cythonize(ret, compiler_directives=compiler_directives)
        
    except ImportError as e:
        print(f"❌ ERROR: Cython is not installed or not found: {e}")
        print("Please install Cython >= 0.29.32:")
        print("  pip install 'cython>=0.29.32'")
        print("  or")
        print("  conda install cython")
        raise SystemExit(1)

setup_args = {}

#if not os.getenv('CONDA_BUILD'):
#    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#    for i, path in enumerate(LIB_LIST):
#    LIB_LIST[i] = os.path.relpath(path, curr_path)
#    setup_args = {
#        "include_package_data": True,
#        "data_files": [('yirage', LIB_LIST)]
#    }

setup(name='yirage',
      version="1.0.5",
      description="Yirage: A Multi-Level Superoptimizer for Tensor Algebra",
      zip_safe=False,
      install_requires=[],
      packages=find_packages(),
      url='https://github.com/yirage-project/yica-yirage',
      ext_modules=config_cython(),
      #**setup_args,
      )
