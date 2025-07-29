#!/usr/bin/env python3
"""
增强版YIRAGE安装脚本
处理OpenMP、CUTLASS等硬性依赖
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys
import platform

# 读取版本信息
def get_version():
    version_file = os.path.join('python', 'yirage', 'version.py')
    with open(version_file, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.1"

# 检测编译环境
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
    
    # 检查CUDA
    if os.path.exists('/usr/local/cuda') or os.environ.get('CUDA_HOME'):
        env['has_cuda'] = True
        print("✅ 检测到CUDA环境")
    
    # 检查依赖路径
    deps_dir = os.path.join(os.getcwd(), 'deps')
    
    if os.path.exists(os.path.join(deps_dir, 'cutlass', 'include')):
        env['cutlass_path'] = os.path.join(deps_dir, 'cutlass')
        print(f"✅ 找到CUTLASS: {env['cutlass_path']}")
    
    if os.path.exists(os.path.join(deps_dir, 'json', 'include')):
        env['json_path'] = os.path.join(deps_dir, 'json')
        print(f"✅ 找到nlohmann/json: {env['json_path']}")
    
    # 优先检查pip安装的Z3
    try:
        import z3
        print(f"✅ 找到Z3 (pip): {z3.get_version_string()}")
        env['z3_pip'] = True
    except ImportError:
        env['z3_pip'] = False
        # 然后检查本地编译的Z3
        if os.path.exists(os.path.join(deps_dir, 'z3', 'install')):
            env['z3_path'] = os.path.join(deps_dir, 'z3', 'install')
            print(f"✅ 找到Z3 (源码): {env['z3_path']}")
        else:
            print("⚠️  未找到Z3，建议运行: pip install z3-solver")
    
    # 检查OpenMP
    if env['is_macos']:
        # macOS使用libomp
        try:
            import subprocess
            result = subprocess.run(['brew', '--prefix', 'libomp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                env['has_openmp'] = True
                env['openmp_path'] = result.stdout.strip()
                print(f"✅ 找到OpenMP (libomp): {env['openmp_path']}")
        except:
            pass
    else:
        # Linux通常有系统OpenMP
        env['has_openmp'] = True
        print("✅ 假设Linux系统有OpenMP支持")
    
    return env

# 构建扩展模块
def create_extensions(env):
    extensions = []
    
    # 基础包含路径
    include_dirs = [
        'include',
        'python',
        pybind11.get_include(),
    ]
    
    # 添加依赖包含路径
    if env['cutlass_path']:
        include_dirs.append(os.path.join(env['cutlass_path'], 'include'))
    
    if env['json_path']:
        include_dirs.append(os.path.join(env['json_path'], 'include'))
    
    if env['z3_path']:
        include_dirs.extend([
            os.path.join(env['z3_path'], 'include'),
        ])
    
    # 编译标志
    compile_args = ['-std=c++17', '-O3']
    link_args = []
    libraries = []
    library_dirs = []
    
    # OpenMP支持
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
    
    # Z3库 (优先使用pip版本，无需手动链接)
    if env.get('z3_pip'):
        # pip安装的Z3会自动处理链接
        print("✅ 使用pip安装的Z3，无需手动链接")
    elif env.get('z3_path'):
        # 使用本地编译的Z3
        library_dirs.append(os.path.join(env['z3_path'], 'lib'))
        libraries.append('z3')
        include_dirs.append(os.path.join(env['z3_path'], 'include'))
        print("✅ 使用本地编译的Z3")
    else:
        print("⚠️  未找到Z3，某些功能可能不可用")
    
    # CUDA支持 (可选)
    if env['has_cuda']:
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        include_dirs.append(os.path.join(cuda_home, 'include'))
        library_dirs.append(os.path.join(cuda_home, 'lib64'))
        libraries.extend(['cuda', 'cudart', 'cublas'])
        compile_args.append('-DYICA_ENABLE_CUDA')
    else:
        compile_args.append('-DYICA_CPU_ONLY')
    
    # 创建核心扩展
    try:
        core_extension = Pybind11Extension(
            "yirage._core",
            sources=[
                # 添加关键源文件
                "src/base/layout.cc",
                "src/search/config.cc",
                "src/search/search.cc",
                # 可以根据需要添加更多源文件
            ],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            cxx_std=17,
        )
        
        # 设置编译和链接参数
        core_extension.extra_compile_args = compile_args
        core_extension.extra_link_args = link_args
        
        extensions.append(core_extension)
        print(f"✅ 创建核心扩展模块")
        
    except Exception as e:
        print(f"⚠️  跳过C++扩展模块: {e}")
    
    return extensions

# 主安装配置
def main():
    print("🔧 检测编译环境...")
    env = detect_compile_env()
    
    print("🔨 创建扩展模块...")
    extensions = create_extensions(env)
    
    # 基础依赖
    install_requires = [
        "numpy>=1.19.0",
        "z3-solver>=4.8.0",
    ]
    
    # Z3依赖处理
    if env.get('z3_pip'):
        # 已经通过pip安装，无需重复添加
        print("✅ Z3依赖已通过pip满足")
    elif env.get('z3_path'):
        # 有本地编译版本，无需pip版本
        print("✅ Z3依赖通过本地编译满足")
    else:
        # 确保有Z3依赖
        print("📦 将通过pip安装Z3")
    
    # PyTorch依赖 (可选)
    try:
        import torch
        print(f"✅ 检测到PyTorch {torch.__version__}")
    except ImportError:
        install_requires.append("torch>=1.12.0")
        print("📦 将安装PyTorch")
    
    setup(
        name="yica-yirage",
        version=get_version(),
        description="YICA-Yirage: AI Computing Optimization Framework (Enhanced Build)",
        long_description="YICA-Yirage with OpenMP, CUTLASS, and Z3 support",
        long_description_content_type="text/plain",
        author="YICA Team",
        author_email="contact@yica.ai",
        
        # 包配置
        package_dir={"": "python"},
        packages=find_packages(where="python"),
        
        # C++扩展
        ext_modules=extensions,
        cmdclass={"build_ext": build_ext},
        
        # 依赖
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
        
        # 分类
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
