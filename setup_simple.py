#!/usr/bin/env python3
"""
纯Python版YIRAGE安装脚本 (回退版本)
"""

from setuptools import setup, find_packages
import os

def get_version():
    version_file = os.path.join('python', 'yirage', 'version.py')
    with open(version_file, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.2"

setup(
    name="yica-yirage",
    version=get_version(),
    description="YICA-Yirage: AI Computing Optimization Framework (Pure Python)",
    long_description="YICA-Yirage Pure Python version without C++ extensions",
    long_description_content_type="text/plain",
    author="YICA Team",
    author_email="contact@yica.ai",
    
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.12.0",
        "z3-solver>=4.8.0",
    ],
    
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"],
        "triton": ["triton>=2.0.0; sys_platform=='linux'"],
        "full": ["torch>=1.12.0", "matplotlib>=3.0.0", "tqdm>=4.0.0"],
    },
    
    python_requires=">=3.8",
    zip_safe=False,
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 