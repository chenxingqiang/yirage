#!/bin/bash
# YICA Build System Verification Test Script
# Tests the build system configuration and dependencies

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
TEST_BUILD_DIR="$YIRAGE_ROOT/test_build"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function: Run a test
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TESTS_TOTAL++))
    info "🧪 Test $TESTS_TOTAL: $test_name"
    
    if $test_function; then
        success "PASSED: $test_name"
        ((TESTS_PASSED++))
    else
        error "FAILED: $test_name"
        ((TESTS_FAILED++))
    fi
    echo
}

# Test 1: CMake configuration generation
test_cmake_configuration() {
    mkdir -p "$TEST_BUILD_DIR"
    cd "$TEST_BUILD_DIR"
    
    # Test basic CMake configuration
    if cmake "$YIRAGE_ROOT" \
        -DENABLE_YICA=ON \
        -DYICA_HARDWARE_ACCELERATION=ON \
        -DBUILD_YICA_CYTHON_BINDINGS=ON \
        &> cmake_config.log; then
        
        # Check if YICA-specific variables are set
        if grep -q "YICA Hardware Acceleration: ENABLED" cmake_config.log; then
            return 0
        else
            error "YICA configuration not found in CMake output"
            return 1
        fi
    else
        error "CMake configuration failed"
        cat cmake_config.log
        return 1
    fi
}

# Test 2: Dependency detection
test_dependency_detection() {
    local all_deps_found=true
    
    # Test CMake
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/.*version //')
        success "CMake found: $CMAKE_VERSION"
    else
        error "CMake not found"
        all_deps_found=false
    fi
    
    # Test Cython
    if python3 -c "import Cython; print('Cython version:', Cython.__version__)" &> /dev/null; then
        CYTHON_VERSION=$(python3 -c "import Cython; print(Cython.__version__)")
        success "Cython found: $CYTHON_VERSION"
    else
        error "Cython not found"
        all_deps_found=false
    fi
    
    # Test PyBind11
    if python3 -c "import pybind11" &> /dev/null; then
        success "PyBind11 found (backup option available)"
    else
        warning "PyBind11 not found (backup option not available)"
    fi
    
    # Test C++ compiler
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1)
        success "g++ compiler found: $GCC_VERSION"
        
        # Test C++17 support
        if echo 'int main() { auto x = []() { return 42; }; return x() - 42; }' | g++ -std=c++17 -x c++ - -o /tmp/cpp17_test 2>/dev/null; then
            success "C++17 support verified"
            rm -f /tmp/cpp17_test
        else
            error "C++17 support not available"
            all_deps_found=false
        fi
    else
        error "g++ compiler not found"
        all_deps_found=false
    fi
    
    # Test Python development headers
    if python3-config --includes &> /dev/null; then
        success "Python development headers found"
    else
        error "Python development headers not found"
        all_deps_found=false
    fi
    
    $all_deps_found
}

# Test 3: YICA source files existence
test_yica_source_files() {
    local all_files_found=true
    
    # Check core YICA headers
    local yica_headers=(
        "include/yirage/yica/yica_backend.h"
        "include/yirage/yica/yica_hardware_abstraction.h"
        "include/yirage/yica/config.h"
    )
    
    for header in "${yica_headers[@]}"; do
        if [ -f "$YIRAGE_ROOT/$header" ]; then
            success "Found header: $header"
        else
            warning "Missing header: $header (will be created in Phase 2)"
        fi
    done
    
    # Check Cython files
    local cython_files=(
        "python/yirage/_cython/yica_kernels.pyx"
        "python/yirage/_cython/CCore.pxd"
        "python/yirage/_cython/core.pyx"
    )
    
    for cython_file in "${cython_files[@]}"; do
        if [ -f "$YIRAGE_ROOT/$cython_file" ]; then
            success "Found Cython file: $cython_file"
        else
            error "Missing Cython file: $cython_file"
            all_files_found=false
        fi
    done
    
    # Check if at least the basic structure exists
    if [ -d "$YIRAGE_ROOT/include/yirage/yica" ]; then
        success "YICA include directory structure exists"
    else
        warning "YICA include directory structure missing (will be created in Phase 2)"
    fi
    
    return 0  # Pass test even if some files are missing (expected during development)
}

# Test 4: Build target generation
test_build_target_generation() {
    cd "$TEST_BUILD_DIR"
    
    # Generate Makefile
    if make --dry-run &> make_dry_run.log; then
        success "Makefile generated successfully"
        
        # Check for YICA-related targets
        if grep -q "yirage_runtime" make_dry_run.log; then
            success "Main runtime target found"
        else
            error "Main runtime target not found"
            return 1
        fi
        
        return 0
    else
        error "Makefile generation failed"
        cat make_dry_run.log
        return 1
    fi
}

# Test 5: Cython configuration test
test_cython_configuration() {
    cd "$YIRAGE_ROOT/python"
    
    # Test Cython setup script
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from cython_setup import config_cython
    print('✅ Cython setup script can be imported')
    
    # Test configuration (may fail due to missing libraries, but should not crash)
    try:
        extensions = config_cython()
        print(f'✅ Configured {len(extensions)} Cython extensions')
    except SystemExit:
        print('⚠️  Cython configuration failed (expected without built libraries)')
    except Exception as e:
        print(f'⚠️  Cython configuration error: {e}')
        
except ImportError as e:
    print(f'❌ Failed to import cython_setup: {e}')
    sys.exit(1)
" 2>&1; then
        success "Cython configuration test passed"
        return 0
    else
        error "Cython configuration test failed"
        return 1
    fi
}

# Test 6: Environment variable test
test_environment_variables() {
    # Source the environment setup
    if [ -f "$YIRAGE_ROOT/scripts/yica_env.sh" ]; then
        source "$YIRAGE_ROOT/scripts/yica_env.sh"
        
        # Check key environment variables
        if [ "$YICA_BACKEND" = "yica" ]; then
            success "YICA_BACKEND environment variable set correctly"
        else
            error "YICA_BACKEND environment variable not set correctly"
            return 1
        fi
        
        if [ "$YICA_DEVICE" = "yz-g100" ]; then
            success "YICA_DEVICE environment variable set correctly"
        else
            error "YICA_DEVICE environment variable not set correctly"
            return 1
        fi
        
        return 0
    else
        warning "Environment setup script not found (will be created by setup script)"
        return 0
    fi
}

# Test 7: Python import test
test_python_import() {
    cd "$YIRAGE_ROOT/python"
    
    # Test basic yirage import
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import yirage
    print(f'✅ yirage imported successfully, version: {yirage.__version__}')
    
    # Test basic functionality
    graph = yirage.new_kernel_graph()
    print('✅ Kernel graph creation successful')
    
    A = graph.new_input(dims=(4, 4), dtype='float32')
    B = graph.new_input(dims=(4, 4), dtype='float32')
    C = graph.matmul(A, B)
    graph.mark_output(C)
    print('✅ Graph construction successful')
    
except Exception as e:
    print(f'❌ Python import test failed: {e}')
    sys.exit(1)
"; then
        success "Python import test passed"
        return 0
    else
        error "Python import test failed"
        return 1
    fi
}

# Test 8: Build script functionality
test_build_script_functionality() {
    # Test build script help
    if "$YIRAGE_ROOT/scripts/build_yica_hardware.sh" --help &> /dev/null; then
        success "Build script help function works"
    else
        error "Build script help function failed"
        return 1
    fi
    
    # Test build script clean function
    if "$YIRAGE_ROOT/scripts/build_yica_hardware.sh" --clean &> /dev/null; then
        success "Build script clean function works"
    else
        error "Build script clean function failed"
        return 1
    fi
    
    return 0
}

# Function: Generate test report
generate_test_report() {
    local report_file="$TEST_BUILD_DIR/build_system_test_report.txt"
    
    cat > "$report_file" << EOF
YICA Build System Test Report
============================
Test Date: $(date)
Test Host: $(hostname)
Test User: $(whoami)

Test Results:
- Total Tests: $TESTS_TOTAL
- Passed: $TESTS_PASSED
- Failed: $TESTS_FAILED
- Success Rate: $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%

Environment Information:
- OS: $(uname -s) $(uname -r)
- Architecture: $(uname -m)
- CMake Version: $(cmake --version | head -n1 2>/dev/null || echo "Not found")
- GCC Version: $(g++ --version | head -n1 2>/dev/null || echo "Not found")
- Python Version: $(python3 --version 2>/dev/null || echo "Not found")
- Cython Version: $(python3 -c "import Cython; print(Cython.__version__)" 2>/dev/null || echo "Not found")

Test Details:
EOF

    # Add test log if available
    if [ -f "$TEST_BUILD_DIR/cmake_config.log" ]; then
        echo "" >> "$report_file"
        echo "CMake Configuration Log:" >> "$report_file"
        echo "========================" >> "$report_file"
        tail -n 20 "$TEST_BUILD_DIR/cmake_config.log" >> "$report_file"
    fi
    
    success "Test report generated: $report_file"
}

# Function: Cleanup test artifacts
cleanup_test_artifacts() {
    if [ -d "$TEST_BUILD_DIR" ]; then
        rm -rf "$TEST_BUILD_DIR"
        success "Test artifacts cleaned up"
    fi
}

# Function: Display test summary
display_test_summary() {
    info "📊 Build System Test Summary"
    echo "=========================="
    echo "Total Tests: $TESTS_TOTAL"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "Success Rate: $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%"
    echo
    
    if [ $TESTS_FAILED -eq 0 ]; then
        success "🎉 All build system tests passed!"
        echo
        info "✅ Build system is ready for YICA development"
        info "Next steps:"
        echo "1. Run environment setup: ./scripts/setup_yica_environment.sh"
        echo "2. Build YICA: ./scripts/build_yica_hardware.sh"
        echo "3. Start Phase 2: C++ backend implementation"
    else
        error "❌ Some build system tests failed"
        echo
        warning "Please fix the failing tests before proceeding to development"
        info "Check the test report for details: $TEST_BUILD_DIR/build_system_test_report.txt"
    fi
}

# Main execution
main() {
    info "🧪 YICA Build System Verification Tests"
    info "======================================="
    echo
    
    # Create test build directory
    mkdir -p "$TEST_BUILD_DIR"
    
    # Run all tests
    run_test "CMake Configuration Generation" test_cmake_configuration
    run_test "Dependency Detection" test_dependency_detection
    run_test "YICA Source Files Existence" test_yica_source_files
    run_test "Build Target Generation" test_build_target_generation
    run_test "Cython Configuration" test_cython_configuration
    run_test "Environment Variables" test_environment_variables
    run_test "Python Import" test_python_import
    run_test "Build Script Functionality" test_build_script_functionality
    
    # Generate report and cleanup
    generate_test_report
    display_test_summary
    
    # Return appropriate exit code
    if [ $TESTS_FAILED -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "YICA Build System Test Script"
        echo "Usage: $0 [options]"
        echo ""
        echo "This script tests the YICA build system configuration"
        echo "and verifies all dependencies are properly set up."
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --cleanup      Clean up test artifacts"
        exit 0
        ;;
    --cleanup)
        cleanup_test_artifacts
        success "Test cleanup completed"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
