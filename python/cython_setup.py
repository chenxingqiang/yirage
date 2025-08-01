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
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        cython_path = path.join(path.dirname(__file__), "yirage/_cython")
        yirage_path = path.join(path.dirname(__file__), "..")
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "yirage.%s" % fn[:-4],
                ["%s/%s" % (cython_path, fn)],
                include_dirs=[path.join(yirage_path, "include"),
                              path.join(yirage_path, "deps", "json", "include"),
                              path.join(yirage_path, "deps", "cutlass", "include"),
                              "/usr/local/cuda/include"],
                libraries=["yirage_runtime", "cudadevrt", "cudart_static", "cudnn", "cublas", "cudart", "cuda", "z3", "gomp", "rt"],
                library_dirs=[path.join(yirage_path, "build"),
                              path.join(yirage_path, "deps", "z3", "build"),
                              "/usr/local/cuda/lib",
                              "/usr/local/cuda/lib64",
                              "/usr/local/cuda/lib64/stubs"],
                extra_compile_args=["-std=c++17", "-fopenmp"],
                extra_link_args=["-fPIC", "-fopenmp"],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)
        return []

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
      version="0.2.4",
      description="Yirage: A Multi-Level Superoptimizer for Tensor Algebra",
      zip_safe=False,
      install_requires=[],
      packages=find_packages(),
      url='https://github.com/yirage-project/yirage',
      ext_modules=config_cython(),
      #**setup_args,
      )
