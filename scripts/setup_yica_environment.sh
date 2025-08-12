#!/bin/bash
# YICA Hardware Environment Setup Script
# Sets up the development environment for yz-g100 hardware

set -e

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YIRAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

info "🎯 YICA Hardware Environment Setup"
info "=================================="
info "Yirage Root: $YIRAGE_ROOT"
echo

# Function: Setup conda environment
setup_conda_environment() {
    info "🐍 Setting up conda environment for YICA..."
    
    CONDA_ENV_FILE="$YIRAGE_ROOT/conda/yirage.yml"
    
    if [ ! -f "$CONDA_ENV_FILE" ]; then
        error "Conda environment file not found: $CONDA_ENV_FILE"
        exit 1
    fi
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        error "Conda not found. Please install Anaconda or Miniconda"
        exit 1
    fi
    
    # Create or update environment
    if conda env list | grep -q "yirage-yica"; then
        info "Updating existing yirage-yica environment..."
        conda env update -f "$CONDA_ENV_FILE"
    else
        info "Creating new yirage-yica environment..."
        conda env create -f "$CONDA_ENV_FILE"
    fi
    
    success "Conda environment setup completed"
    echo
}

# Function: Install additional dependencies
install_additional_deps() {
    info "📦 Installing additional YICA dependencies..."
    
    # Activate the conda environment
    eval "$(conda shell.bash hook)"
    conda activate yirage-yica
    
    # Install YICA-specific requirements
    if [ -f "$YIRAGE_ROOT/requirements-yica.txt" ]; then
        pip install -r "$YIRAGE_ROOT/requirements-yica.txt"
        success "YICA requirements installed"
    fi
    
    # Verify key dependencies
    info "🔍 Verifying key dependencies..."
    
    python -c "import Cython; print('✅ Cython version:', Cython.__version__)"
    python -c "import torch; print('✅ PyTorch version:', torch.__version__)"
    python -c "import numpy; print('✅ NumPy version:', numpy.__version__)"
    
    # Check for optional dependencies
    if python -c "import pybind11" 2>/dev/null; then
        python -c "import pybind11; print('✅ PyBind11 available (backup option)')"
    else
        warning "PyBind11 not found (backup option not available)"
    fi
    
    success "Dependency verification completed"
    echo
}

# Function: Setup environment variables
setup_environment_variables() {
    info "⚙️  Setting up YICA environment variables..."
    
    # Create environment setup file
    ENV_SETUP_FILE="$YIRAGE_ROOT/scripts/yica_env.sh"
    
    cat > "$ENV_SETUP_FILE" << 'EOF'
#!/bin/bash
# YICA Environment Variables
# Source this file to set up YICA development environment

# YICA Configuration
export YICA_BACKEND=yica
export YICA_DEVICE=yz-g100
export YICA_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Hardware Configuration
export YICA_CIM_ARRAYS=4
export YICA_SPM_SIZE_MB=128
export YICA_HARDWARE_TARGET=yz-g100

# Build Configuration
export YIRAGE_BUILD_DIR="$YICA_HOME/build_cpp"
export YIRAGE_ENABLE_YICA=1
export YICA_HARDWARE_ACCELERATION=1

# Development Configuration
export PYTHONPATH="$YICA_HOME/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$YIRAGE_BUILD_DIR:$LD_LIBRARY_PATH"

# Debugging
export YICA_DEBUG=${YICA_DEBUG:-0}
export YICA_VERBOSE=${YICA_VERBOSE:-0}

echo "🎯 YICA environment configured:"
echo "  YICA_BACKEND: $YICA_BACKEND"
echo "  YICA_DEVICE: $YICA_DEVICE"
echo "  YICA_HOME: $YICA_HOME"
echo "  Hardware Target: $YICA_HARDWARE_TARGET"
echo "  CIM Arrays: $YICA_CIM_ARRAYS"
echo "  SPM Size: ${YICA_SPM_SIZE_MB}MB"
EOF

    chmod +x "$ENV_SETUP_FILE"
    success "Environment setup file created: $ENV_SETUP_FILE"
    echo
}

# Function: Create development aliases
create_dev_aliases() {
    info "🔧 Creating development aliases..."
    
    ALIAS_FILE="$YIRAGE_ROOT/scripts/yica_aliases.sh"
    
    cat > "$ALIAS_FILE" << EOF
#!/bin/bash
# YICA Development Aliases

# Build aliases
alias yica-build='$YIRAGE_ROOT/scripts/build_yica_hardware.sh'
alias yica-clean='$YIRAGE_ROOT/scripts/build_yica_hardware.sh --clean'
alias yica-debug='BUILD_TYPE=Debug $YIRAGE_ROOT/scripts/build_yica_hardware.sh'

# Test aliases
alias yica-test='cd $YIRAGE_ROOT && python -m pytest tests/ -v'
alias yica-test-hardware='cd $YIRAGE_ROOT && python -m pytest tests/hardware/ -v -s'
alias yica-test-cython='cd $YIRAGE_ROOT && python -m pytest tests/python/ -v'

# Development aliases
alias yica-env='source $YIRAGE_ROOT/scripts/yica_env.sh'
alias yica-python='cd $YIRAGE_ROOT/python && python'
alias yica-root='cd $YIRAGE_ROOT'

# Monitoring aliases
alias yica-build-log='tail -f $YIRAGE_ROOT/build_cpp/build.log'
alias yica-build-report='cat $YIRAGE_ROOT/build_cpp/yica_build_report.txt'

echo "🔧 YICA development aliases loaded"
echo "Available commands:"
echo "  yica-build      - Build YICA with hardware acceleration"
echo "  yica-clean      - Clean build artifacts"
echo "  yica-debug      - Build in debug mode"
echo "  yica-test       - Run all tests"
echo "  yica-env        - Load YICA environment"
echo "  yica-python     - Start Python in yirage directory"
EOF

    chmod +x "$ALIAS_FILE"
    success "Development aliases created: $ALIAS_FILE"
    echo
}

# Function: Setup VS Code configuration
setup_vscode_config() {
    info "💻 Setting up VS Code configuration..."
    
    VSCODE_DIR="$YIRAGE_ROOT/.vscode"
    mkdir -p "$VSCODE_DIR"
    
    # C++ configuration
    cat > "$VSCODE_DIR/c_cpp_properties.json" << EOF
{
    "configurations": [
        {
            "name": "YICA Development",
            "includePath": [
                "\${workspaceFolder}/yirage/include",
                "\${workspaceFolder}/yirage/include/yirage/yica",
                "\${workspaceFolder}/yirage/deps/*/include",
                "/usr/local/cuda/include",
                "/usr/include/python3.11"
            ],
            "defines": [
                "YIRAGE_ENABLE_YICA",
                "YICA_HARDWARE_ACCELERATION",
                "YICA_TARGET_YZ_G100",
                "YICA_CIM_ARRAYS=4"
            ],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64",
            "compileCommands": "\${workspaceFolder}/yirage/build_cpp/compile_commands.json"
        }
    ],
    "version": 4
}
EOF

    # Python configuration
    cat > "$VSCODE_DIR/settings.json" << EOF
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/yirage-yica/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.associations": {
        "*.pyx": "python",
        "*.pxd": "python"
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "cmake.buildDirectory": "\${workspaceFolder}/yirage/build_cpp"
}
EOF

    # Launch configuration for debugging
    cat > "$VSCODE_DIR/launch.json" << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug YICA Python",
            "type": "python",
            "request": "launch",
            "program": "\${file}",
            "console": "integratedTerminal",
            "env": {
                "YICA_BACKEND": "yica",
                "YICA_DEVICE": "yz-g100",
                "YICA_DEBUG": "1",
                "PYTHONPATH": "\${workspaceFolder}/yirage/python"
            }
        },
        {
            "name": "Debug YICA C++",
            "type": "cppdbg",
            "request": "launch",
            "program": "\${workspaceFolder}/yirage/build_cpp/cpp_examples/dnn",
            "args": [],
            "stopAtEntry": false,
            "cwd": "\${workspaceFolder}/yirage",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb"
        }
    ]
}
EOF

    success "VS Code configuration created"
    echo
}

# Function: Display setup summary
display_setup_summary() {
    info "📋 Environment Setup Summary"
    echo
    success "YICA development environment is ready!"
    echo
    info "To get started:"
    echo "1. 🐍 Activate conda environment:"
    echo "   conda activate yirage-yica"
    echo
    echo "2. ⚙️  Load YICA environment:"
    echo "   source $YIRAGE_ROOT/scripts/yica_env.sh"
    echo
    echo "3. 🔧 Load development aliases:"
    echo "   source $YIRAGE_ROOT/scripts/yica_aliases.sh"
    echo
    echo "4. 🔨 Build YICA with hardware acceleration:"
    echo "   yica-build"
    echo
    echo "5. 🧪 Run tests:"
    echo "   yica-test"
    echo
    info "Configuration files created:"
    echo "  - $YIRAGE_ROOT/scripts/yica_env.sh (environment variables)"
    echo "  - $YIRAGE_ROOT/scripts/yica_aliases.sh (development aliases)"
    echo "  - $YIRAGE_ROOT/.vscode/ (VS Code configuration)"
    echo
    warning "Remember to activate the conda environment before development!"
}

# Main execution
main() {
    setup_conda_environment
    install_additional_deps
    setup_environment_variables
    create_dev_aliases
    setup_vscode_config
    display_setup_summary
    
    success "🎉 YICA environment setup completed!"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "YICA Hardware Environment Setup Script"
        echo "Usage: $0 [options]"
        echo ""
        echo "This script sets up the complete development environment for"
        echo "YICA hardware acceleration on yz-g100."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
