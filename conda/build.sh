#!/bin/bash

# Exit on any error
set -e

echo "=== YICA-Yirage Conda Build Script ==="

# Set environment variables
export CMAKE_BUILD_TYPE=Release
export PYTHONPATH="${SRC_DIR}/python:${PYTHONPATH}"

# Install build dependencies
echo "Installing build dependencies..."
pip install --no-deps --no-build-isolation pybind11>=2.10.0 cython>=0.29.32

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DPYTHON_EXECUTABLE=${PYTHON} \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DENABLE_CUDA=OFF \
    -DENABLE_OPENMP=ON \
    -DCMAKE_CXX_FLAGS="-std=c++17 -O3 -fPIC" \
    -DCMAKE_C_FLAGS="-O3 -fPIC"

# Build
echo "Building..."
make -j${CPU_COUNT}

# Install
echo "Installing..."
make install

# Install Python package with Cython extensions
echo "Building and installing Python package with Cython extensions..."
cd ${SRC_DIR}
cp -f ${SRC_DIR}/python/cython_setup.py ${SRC_DIR}/setup.py
python -m pip install --no-deps --no-build-isolation .

echo "=== Build completed successfully ==="
