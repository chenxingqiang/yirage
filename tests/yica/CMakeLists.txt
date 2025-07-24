# YICA测试配置

# 设置测试源文件
set(YICA_TEST_SOURCES
    test_yica_analyzer.cc
    test_strategy_library.cc
    test_code_generator.cc
)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/json/include)

# 创建测试可执行文件
add_executable(yica_tests ${YICA_TEST_SOURCES})

# 链接必要的库
target_link_libraries(yica_tests
    gtest
    gtest_main
    pthread
)

# 如果需要链接其他Yirage库
# target_link_libraries(yica_tests yirage_core)

# 添加编译选项
target_compile_options(yica_tests PRIVATE -std=c++17 -Wall -Wextra)

# 添加测试到CTest
add_test(NAME YICATests COMMAND yica_tests)

# YICA优化器测试

add_executable(test_yica_analyzer
    test_yica_analyzer.cc
    ../src/search/yica/yica_analyzer.cc
    ../src/search/yica/yica_types.cc
)

target_link_libraries(test_yica_analyzer
    gtest
    gtest_main
    pthread
)

target_include_directories(test_yica_analyzer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/deps/json/include
)

add_test(NAME YICAAnalyzerTest COMMAND test_yica_analyzer)

# YICA策略库测试
add_executable(test_strategy_library
    test_strategy_library.cc
    ../src/search/yica/optimization_strategy.cc
    ../src/search/yica/strategy_library.cc
    ../src/search/yica/yica_types.cc
)

target_link_libraries(test_strategy_library
    gtest
    gtest_main
    pthread
)

target_include_directories(test_strategy_library PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/deps/json/include
)

add_test(NAME StrategyLibraryTest COMMAND test_strategy_library)

# YICA代码生成器测试
add_executable(test_code_generator
    test_code_generator.cc
    ../src/search/yica/code_generator.cc
    ../src/search/yica/operator_generators.cc
    ../src/search/yica/yica_types.cc
)

target_link_libraries(test_code_generator
    gtest
    gtest_main
    pthread
)

target_include_directories(test_code_generator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/deps/json/include
)

add_test(NAME CodeGeneratorTest COMMAND test_code_generator)

# YICA运行时优化器测试
add_executable(test_runtime_optimizer
    test_runtime_optimizer.cc
    ../src/search/yica/runtime_types.cc
    ../src/search/yica/performance_monitor.cc
    ../src/search/yica/runtime_optimizer.cc
    ../src/search/yica/ml_optimizer.cc
    ../src/search/yica/yica_runtime.cc
    ../src/search/yica/yica_types.cc
)

target_link_libraries(test_runtime_optimizer
    gtest
    gtest_main
    pthread
)

target_include_directories(test_runtime_optimizer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/deps/json/include
)

add_test(NAME RuntimeOptimizerTest COMMAND test_runtime_optimizer) 