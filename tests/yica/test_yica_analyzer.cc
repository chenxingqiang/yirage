#include "yirage/search/yica/yica_analyzer.h"
#include "yirage/kernel/graph.h"
#include <cassert>
#include <iostream>

using namespace yirage::search::yica;
using namespace yirage::kernel;

void test_basic_analysis() {
    std::cout << "Testing basic YICA analysis..." << std::endl;
    
    // 创建YICA配置
    YICAConfig config;
    config.cim_array_rows = 256;
    config.cim_array_cols = 256;
    config.spm_size_per_die = 2 * 1024 * 1024;
    config.num_cim_dies = 16;
    
    // 创建分析器
    YICAArchitectureAnalyzer analyzer(config);
    
    // 创建简单的kernel graph
    Graph graph;
    auto input1 = graph.new_input({128, 128}, {128, 1}, 
                                 yirage::type::DT_FLOAT32, 
                                 yirage::layout::DmemRowMajor);
    auto input2 = graph.new_input({128, 128}, {128, 1}, 
                                 yirage::type::DT_FLOAT32, 
                                 yirage::layout::DmemRowMajor);
    
    // 添加矩阵乘法操作
    auto output = graph.matmul(input1, input2);
    graph.mark_output(output);
    
    // 执行分析
    AnalysisResult result = analyzer.analyze_computation_pattern(graph);
    
    // 验证结果
    assert(result.cim_friendliness_score > 0.5f);  // 矩阵乘法应该有高CIM友好度
    assert(result.memory_locality_score >= 0.0f && result.memory_locality_score <= 1.0f);
    assert(result.estimated_speedup >= 1.0f);
    assert(!result.cim_friendly_ops.empty());  // 应该识别出CIM友好的操作
    
    std::cout << "✓ Basic analysis test passed" << std::endl;
    std::cout << "  CIM friendliness: " << result.cim_friendliness_score << std::endl;
    std::cout << "  Memory locality: " << result.memory_locality_score << std::endl;
    std::cout << "  Estimated speedup: " << result.estimated_speedup << "x" << std::endl;
}

void test_cim_operation_identification() {
    std::cout << "Testing CIM operation identification..." << std::endl;
    
    YICAConfig config;
    YICAArchitectureAnalyzer analyzer(config);
    
    Graph graph;
    auto input1 = graph.new_input({64, 64}, {64, 1}, 
                                 yirage::type::DT_FLOAT16, 
                                 yirage::layout::DmemRowMajor);
    auto input2 = graph.new_input({64, 64}, {64, 1}, 
                                 yirage::type::DT_FLOAT16, 
                                 yirage::layout::DmemRowMajor);
    
    // 添加各种操作
    auto matmul_out = graph.matmul(input1, input2);
    auto add_out = graph.add(input1, input2);
    auto exp_out = graph.exp(input1);
    
    graph.mark_output(matmul_out);
    graph.mark_output(add_out);
    graph.mark_output(exp_out);
    
    // 识别CIM友好操作
    auto cim_ops = analyzer.identify_cim_operations(graph);
    
    // 验证结果
    assert(!cim_ops.empty());  // 应该至少识别出一些CIM友好操作
    
    std::cout << "✓ CIM operation identification test passed" << std::endl;
    std::cout << "  Found " << cim_ops.size() << " CIM-friendly operations" << std::endl;
}

void test_parallelization_opportunities() {
    std::cout << "Testing parallelization opportunity detection..." << std::endl;
    
    YICAConfig config;
    config.num_cim_dies = 8;
    YICAArchitectureAnalyzer analyzer(config);
    
    Graph graph;
    auto input1 = graph.new_input({1024, 1024}, {1024, 1}, 
                                 yirage::type::DT_FLOAT32, 
                                 yirage::layout::DmemRowMajor);
    auto input2 = graph.new_input({1024, 1024}, {1024, 1}, 
                                 yirage::type::DT_FLOAT32, 
                                 yirage::layout::DmemRowMajor);
    
    // 大型矩阵乘法
    auto output = graph.matmul(input1, input2);
    graph.mark_output(output);
    
    // 查找并行化机会
    auto opportunities = analyzer.find_parallel_patterns(graph);
    
    // 验证结果
    assert(!opportunities.empty());  // 大型矩阵乘法应该有并行化机会
    
    for (const auto& opp : opportunities) {
        assert(opp.efficiency_score >= 0.0f && opp.efficiency_score <= 1.0f);
        assert(opp.recommended_parallelism > 0);
        assert(!opp.description.empty());
    }
    
    std::cout << "✓ Parallelization opportunity test passed" << std::endl;
    std::cout << "  Found " << opportunities.size() << " parallelization opportunities" << std::endl;
}

void test_memory_access_analysis() {
    std::cout << "Testing memory access pattern analysis..." << std::endl;
    
    YICAConfig config;
    config.spm_size_per_die = 1024 * 1024;  // 1MB SPM
    YICAArchitectureAnalyzer analyzer(config);
    
    Graph graph;
    
    // 小张量（适合SPM）
    auto small_input = graph.new_input({32, 32}, {32, 1}, 
                                      yirage::type::DT_FLOAT32, 
                                      yirage::layout::DmemRowMajor);
    
    // 大张量（不适合SPM）
    auto large_input = graph.new_input({2048, 2048}, {2048, 1}, 
                                      yirage::type::DT_FLOAT32, 
                                      yirage::layout::DmemRowMajor);
    
    auto small_out = graph.exp(small_input);
    auto large_out = graph.exp(large_input);
    
    graph.mark_output(small_out);
    graph.mark_output(large_out);
    
    // 分析内存访问模式
    float memory_score = analyzer.analyze_memory_access_pattern(graph);
    
    // 验证结果
    assert(memory_score >= 0.0f && memory_score <= 1.0f);
    
    std::cout << "✓ Memory access analysis test passed" << std::endl;
    std::cout << "  Memory locality score: " << memory_score << std::endl;
}

void test_config_update() {
    std::cout << "Testing configuration update..." << std::endl;
    
    YICAConfig initial_config;
    initial_config.num_cim_dies = 8;
    
    YICAArchitectureAnalyzer analyzer(initial_config);
    
    // 验证初始配置
    assert(analyzer.get_config().num_cim_dies == 8);
    
    // 更新配置
    YICAConfig new_config;
    new_config.num_cim_dies = 16;
    new_config.spm_size_per_die = 4 * 1024 * 1024;
    
    analyzer.update_config(new_config);
    
    // 验证更新后的配置
    assert(analyzer.get_config().num_cim_dies == 16);
    assert(analyzer.get_config().spm_size_per_die == 4 * 1024 * 1024);
    
    std::cout << "✓ Configuration update test passed" << std::endl;
}

int main() {
    std::cout << "Running YICA Architecture Analyzer tests..." << std::endl;
    
    try {
        test_basic_analysis();
        test_cim_operation_identification();
        test_parallelization_opportunities();
        test_memory_access_analysis();
        test_config_update();
        
        std::cout << "\n✅ All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
} 