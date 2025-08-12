# YICA 强化学习优化器 CMake 配置

# 设置强化学习优化器相关的编译选项
set(YICA_RL_ENABLED ON CACHE BOOL "Enable YICA Reinforcement Learning Optimizer")

if(YICA_RL_ENABLED)
    message(STATUS "YICA Reinforcement Learning Optimizer: ENABLED")
    
    # 添加强化学习优化器源文件
    set(YICA_RL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/yica_rl_optimizer_simple.cc
    )
    
    # 添加强化学习优化器头文件
    set(YICA_RL_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yirage/yica/yica_backend.h
    )
    
    # 创建强化学习优化器库
    add_library(yica_rl STATIC ${YICA_RL_SOURCES} ${YICA_RL_HEADERS})
    
    # 设置包含目录
    target_include_directories(yica_rl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    # 链接依赖库
    target_link_libraries(yica_rl PRIVATE
        yirage_kernel
        yirage_transpiler
        yirage_utils
    )
    
    # 设置编译选项
    target_compile_features(yica_rl PRIVATE cxx_std_17)
    target_compile_options(yica_rl PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    )
    
    # 定义宏
    target_compile_definitions(yica_rl PRIVATE
        YICA_RL_ENABLED=1
    )
    
    # 安装目标
    install(TARGETS yica_rl
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    # 安装头文件
    install(FILES ${YICA_RL_HEADERS}
        DESTINATION include/yirage/yica
    )
    
else()
    message(STATUS "YICA Reinforcement Learning Optimizer: DISABLED")
    
    # 创建空的库目标以保持兼容性
    add_library(yica_rl INTERFACE)
    
    # 定义禁用宏
    target_compile_definitions(yica_rl INTERFACE
        YICA_RL_ENABLED=0
    )
endif()

# 添加演示程序目标
if(YICA_RL_ENABLED AND BUILD_EXAMPLES)
    add_executable(yica_rl_demo
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/yica_rl_demo.cc
    )
    
    target_include_directories(yica_rl_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(yica_rl_demo PRIVATE
        yica_rl
        yirage_kernel
        yirage_transpiler
    )
    
    target_compile_features(yica_rl_demo PRIVATE cxx_std_17)
    
    # 安装演示程序
    install(TARGETS yica_rl_demo
        RUNTIME DESTINATION bin
    )
endif()

# 添加测试目标
if(YICA_RL_ENABLED AND BUILD_TESTING)
    add_executable(yica_rl_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_rl_optimizer.cc
    )
    
    target_include_directories(yica_rl_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    
    target_link_libraries(yica_rl_test PRIVATE
        yica_rl
        yirage_kernel
        yirage_transpiler
        gtest
        gtest_main
    )
    
    target_compile_features(yica_rl_test PRIVATE cxx_std_17)
    
    # 添加到测试套件
    add_test(NAME YICAReinforcementLearningOptimizerTest 
             COMMAND yica_rl_test)
    
    # 设置测试属性
    set_tests_properties(YICAReinforcementLearningOptimizerTest PROPERTIES
        TIMEOUT 300
        LABELS "yica;rl;optimization"
    )
endif()

# 导出配置
if(YICA_RL_ENABLED)
    export(TARGETS yica_rl
        FILE "${CMAKE_CURRENT_BINARY_DIR}/YICAReinforcementLearningTargets.cmake"
    )
    
    # 创建配置文件
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/YICAReinforcementLearningConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/YICAReinforcementLearningConfig.cmake"
        @ONLY
    )
    
    # 安装配置文件
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/YICAReinforcementLearningConfig.cmake"
        DESTINATION lib/cmake/YICAReinforcementLearning
    )
endif()

# 打印配置摘要
message(STATUS "YICA RL Configuration Summary:")
message(STATUS "  RL Optimizer: ${YICA_RL_ENABLED}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Testing: ${BUILD_TESTING}")
if(YICA_RL_ENABLED)
    message(STATUS "  Source Files: ${YICA_RL_SOURCES}")
    message(STATUS "  Header Files: ${YICA_RL_HEADERS}")
endif()
