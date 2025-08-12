# YICA Architecture Support for Yirage
# This cmake file configures YICA-specific optimizations for yz-g100 hardware

# YICA configuration options - Updated for hardware acceleration
option(ENABLE_YICA "Enable YICA architecture support" ON)
option(YICA_HARDWARE_ACCELERATION "Enable real yz-g100 hardware acceleration" ON)
option(YICA_SIMULATION_MODE "Enable YICA simulation mode" OFF)  # Disabled for real hardware
option(YICA_RUNTIME_PROFILING "Enable YICA runtime profiling" ON)
option(BUILD_YICA_CYTHON_BINDINGS "Build YICA Cython bindings" ON)

if(ENABLE_YICA)
    message(STATUS "🚀 Enabling YICA architecture support for yz-g100")
    
    # YICA specific definitions
    add_definitions(-DYIRAGE_ENABLE_YICA)
    
    # Hardware acceleration specific flags
    if(YICA_HARDWARE_ACCELERATION)
        add_definitions(-DYICA_HARDWARE_ACCELERATION)
        add_definitions(-DYICA_TARGET_YZ_G100)
        message(STATUS "✅ YICA hardware acceleration enabled")
    endif()
    
    if(YICA_SIMULATION_MODE)
        add_definitions(-DYICA_SIMULATION_MODE)
        message(STATUS "⚠️  YICA simulation mode enabled")
    else()
        message(STATUS "🎯 YICA real hardware mode enabled")
    endif()
    
    if(YICA_RUNTIME_PROFILING)
        add_definitions(-DYICA_RUNTIME_PROFILING)
        message(STATUS "📊 YICA runtime profiling enabled")
    endif()
    
    # Cython bindings configuration
    if(BUILD_YICA_CYTHON_BINDINGS)
        add_definitions(-DBUILD_YICA_CYTHON_BINDINGS)
        message(STATUS "🐍 YICA Cython bindings enabled")
    endif()
    
    # YICA specific include directories
    set(YICA_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/runtime
    )
    
    # YICA source files - Only include existing files for Phase 1
    set(YICA_SOURCES)
    
    # Check for existing YICA source files
    set(YICA_SOURCE_CANDIDATES
        # Core YICA backend (existing) - Disabled for minimal build
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/yica_backend.cc  # Complex Graph copy issues
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/yica_hardware_abstraction.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/yica_kernel_generator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/yccl_communicator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/yis_instruction_engine.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/cim_resource_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/spm_memory_manager.cc
        
        # YIS Instruction Engine (existing) - Disabled for now
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/engine/yis_instruction_engine.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/engine/cim_array_simulator.cc
        
            # YICA kernel operators (optimized and unified)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/yica/yica_kernel_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/yica/yica_all_reduce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/yica/yica_element_ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/yica/yica_matmul.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/yica/yica_rms_norm.cc
    
    # YICA KNOperator implementations
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/yica_matmul_op.cc
        
        # Search/YICA optimizers (existing) - Disabled for minimal build
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/yica_analyzer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/code_generator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/cpu_code_generator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/operator_generators.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/optimization_strategy.cc  # Needs fixes
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/runtime_types.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/search/yica/strategy_library.cc  # Needs fixes
        
        # Legacy files (if they exist)
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/architecture_analyzer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/search_space.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/cim_simulator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/spm_manager.cc
    )
    
    # Only include files that actually exist
    foreach(SOURCE_FILE ${YICA_SOURCE_CANDIDATES})
        if(EXISTS ${SOURCE_FILE})
            list(APPEND YICA_SOURCES ${SOURCE_FILE})
            message(STATUS "  ✅ Including YICA source: ${SOURCE_FILE}")
        else()
            message(STATUS "  ⏳ YICA source pending: ${SOURCE_FILE} (Phase 2)")
        endif()
    endforeach()
    
    # If no YICA sources found, create a placeholder
    if(NOT YICA_SOURCES)
        set(YICA_PLACEHOLDER_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/yica_placeholder.cc)
        file(WRITE ${YICA_PLACEHOLDER_SOURCE} 
            "// YICA Placeholder - Phase 1 Build System\n"
            "// This file ensures YICA library can be built during Phase 1\n"
            "// Real implementation will be added in Phase 2\n"
            "\n"
            "namespace yirage {\n"
            "namespace yica {\n"
            "\n"
            "// Placeholder function to ensure library builds\n"
            "bool yica_phase1_placeholder() {\n"
            "    return true;  // Phase 1 build system ready\n"
            "}\n"
            "\n"
            "}  // namespace yica\n"
            "}  // namespace yirage\n"
        )
        list(APPEND YICA_SOURCES ${YICA_PLACEHOLDER_SOURCE})
        message(STATUS "  📝 Created YICA placeholder source for Phase 1")
    endif()
    
    # YICA headers - Only include existing files for Phase 1
    set(YICA_HEADERS)
    
    # Check for existing YICA header files
    set(YICA_HEADER_CANDIDATES
        # Core YICA backend headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/yica_backend.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/yica_hardware_abstraction.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/config.h
        
        # YIS Instruction Set headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/yis_instruction_set.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/yis_instruction_types.h
        
        # Resource Management headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/cim_resource_manager.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/spm_memory_manager.h
        
        # Hardware Communication headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/yz_g100_communicator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/hardware_protocol.h
        
        # Legacy headers (for compatibility)
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/optimizer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/architecture_analyzer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/search_space.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/cim_simulator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/spm_manager.h
    )
    
    # Only include headers that actually exist
    foreach(HEADER_FILE ${YICA_HEADER_CANDIDATES})
        if(EXISTS ${HEADER_FILE})
            list(APPEND YICA_HEADERS ${HEADER_FILE})
            message(STATUS "  ✅ Including YICA header: ${HEADER_FILE}")
        else()
            message(STATUS "  ⏳ YICA header pending: ${HEADER_FILE} (Phase 2)")
        endif()
    endforeach()
    
    # Create YICA library target
    if(YICA_SOURCES)
        add_library(yirage_yica STATIC ${YICA_SOURCES} ${YICA_HEADERS})
        
        target_include_directories(yirage_yica PUBLIC
            ${YICA_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        # Link with Yirage core libraries
        target_link_libraries(yirage_yica PUBLIC
            yirage_search
            yirage_threadblock
            yirage_transpiler
        )
        
        # Set C++ standard
        target_compile_features(yirage_yica PUBLIC cxx_std_17)
        
        # Compiler specific options
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(yirage_yica PRIVATE
                -Wall -Wextra -O3
                $<$<CONFIG:Debug>:-g -O0>
            )
        endif()
        
        message(STATUS "YICA library target created")
    endif()
    
    # YICA Python bindings - Enhanced for hardware acceleration
    if(BUILD_YICA_CYTHON_BINDINGS)
        message(STATUS "🐍 Configuring YICA Cython bindings")
        
        # Find Cython
        find_program(CYTHON_EXECUTABLE cython)
        if(NOT CYTHON_EXECUTABLE)
            message(FATAL_ERROR "Cython not found. Please install Cython >= 0.29.32")
        endif()
        
        # Check Cython version
        execute_process(
            COMMAND ${CYTHON_EXECUTABLE} --version
            OUTPUT_VARIABLE CYTHON_VERSION_OUTPUT
            ERROR_VARIABLE CYTHON_VERSION_OUTPUT
        )
        message(STATUS "Found Cython: ${CYTHON_VERSION_OUTPUT}")
        
        # Check for existing Cython sources
        set(YICA_CYTHON_SOURCES)
        set(YICA_CYTHON_SOURCE_CANDIDATES
            ${CMAKE_CURRENT_SOURCE_DIR}/python/yirage/_cython/core_phase1.pyx
            ${CMAKE_CURRENT_SOURCE_DIR}/python/yirage/_cython/yica_operators.pyx
        )
        
        foreach(SOURCE_FILE ${YICA_CYTHON_SOURCE_CANDIDATES})
            if(EXISTS ${SOURCE_FILE})
                list(APPEND YICA_CYTHON_SOURCES ${SOURCE_FILE})
                message(STATUS "  ✅ Including Cython source: ${SOURCE_FILE}")
            else()
                message(STATUS "  ⏳ Cython source pending: ${SOURCE_FILE} (Phase 2)")
            endif()
        endforeach()
        
        # Check for existing Cython headers
        set(YICA_CYTHON_HEADERS)
        set(YICA_CYTHON_HEADER_CANDIDATES
            ${CMAKE_CURRENT_SOURCE_DIR}/python/yirage/_cython/CCore.pxd
            ${CMAKE_CURRENT_SOURCE_DIR}/python/yirage/_cython/yica_types.pxd
        )
        
        foreach(HEADER_FILE ${YICA_CYTHON_HEADER_CANDIDATES})
            if(EXISTS ${HEADER_FILE})
                list(APPEND YICA_CYTHON_HEADERS ${HEADER_FILE})
                message(STATUS "  ✅ Including Cython header: ${HEADER_FILE}")
            else()
                message(STATUS "  ⏳ Cython header pending: ${HEADER_FILE} (Phase 2)")
            endif()
        endforeach()
        
        # Configure Cython compilation flags
        set(CYTHON_FLAGS 
            --cplus 
            --fast-fail
            -3  # Python 3
            -X language_level=3
            -X boundscheck=False
            -X wraparound=False
        )
        
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CYTHON_FLAGS --gdb)
            message(STATUS "🐛 Cython debug symbols enabled")
        endif()
        
        # Add YICA Cython targets
        foreach(CYTHON_SOURCE ${YICA_CYTHON_SOURCES})
            get_filename_component(MODULE_NAME ${CYTHON_SOURCE} NAME_WE)
            set(CYTHON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/yirage/_cython/${MODULE_NAME}.cpp)
            
            add_custom_command(
                OUTPUT ${CYTHON_OUTPUT}
                COMMAND ${CYTHON_EXECUTABLE} ${CYTHON_FLAGS} -o ${CYTHON_OUTPUT} ${CYTHON_SOURCE}
                DEPENDS ${CYTHON_SOURCE} ${YICA_CYTHON_HEADERS}
                COMMENT "🐍 Compiling Cython module ${MODULE_NAME}"
            )
            
            list(APPEND YICA_CYTHON_OUTPUTS ${CYTHON_OUTPUT})
        endforeach()
        
        # Add YICA Cython library
        add_library(yirage_yica_cython SHARED ${YICA_CYTHON_OUTPUTS})
        add_dependencies(yirage_yica_cython yirage_yica)
        target_link_libraries(yirage_yica_cython yirage_yica)
        target_include_directories(yirage_yica_cython PRIVATE
            ${PYTHON_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        # Set Python extension properties
        set_target_properties(yirage_yica_cython PROPERTIES
            PREFIX ""
            SUFFIX ".so"
            CXX_STANDARD 17
        )
        
        message(STATUS "✅ YICA Cython bindings configured")
    endif()
    
    # YICA tests
    if(BUILD_TESTS)
        set(YICA_TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_yica_optimizer.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_cim_simulator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_spm_manager.cc
        )
        
        foreach(test_source ${YICA_TEST_SOURCES})
            get_filename_component(test_name ${test_source} NAME_WE)
            add_executable(${test_name} ${test_source})
            target_link_libraries(${test_name} yirage_yica gtest gtest_main)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
    endif()
    
    # Installation
    install(TARGETS yirage_yica
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(FILES ${YICA_HEADERS}
        DESTINATION include/yirage/yica
    )
    
else()
    message(STATUS "YICA architecture support disabled")
endif()

# YICA utility functions
function(add_yica_optimization target)
    if(ENABLE_YICA)
        target_link_libraries(${target} yirage_yica)
        target_compile_definitions(${target} PRIVATE YIRAGE_ENABLE_YICA)
    endif()
endfunction()

# Export YICA configuration
if(ENABLE_YICA)
    set(YIRAGE_YICA_ENABLED TRUE CACHE BOOL "YICA support enabled" FORCE)
    set(YIRAGE_YICA_INCLUDE_DIRS ${YICA_INCLUDE_DIRS} CACHE STRING "YICA include directories" FORCE)
else()
    set(YIRAGE_YICA_ENABLED FALSE CACHE BOOL "YICA support disabled" FORCE)
endif() 