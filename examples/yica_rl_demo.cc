/**
 * @file yica_rl_demo.cc
 * @brief YICA 强化学习优化器 C++ 演示程序
 * 
 * 这个演示程序展示了如何使用 YICA 强化学习优化器来优化计算图：
 * 1. 创建测试计算图
 * 2. 初始化 YICA 后端和 RL 优化器
 * 3. 比较传统优化和 RL 优化的效果
 * 4. 训练 RL 优化器
 * 5. 保存和加载模型
 */

#include "yirage/yica/yica_backend.h"
#include "yirage/yica/config.h"
#include "yirage/kernel/graph.h"
#include "yirage/kernel/operator.h"

#include <iostream>
#include <chrono>
#include <memory>
#include <vector>

using namespace yirage;
using namespace yirage::yica;

// 创建测试计算图
std::unique_ptr<kernel::Graph> create_test_graph(const std::string& name) {
    auto graph = std::make_unique<kernel::Graph>();
    
    if (name == "matmul_chain") {
        // 创建矩阵乘法链
        for (int i = 0; i < 3; ++i) {
            auto op = std::make_unique<kernel::KNOperator>();
            op->op_type = kernel::KNOperatorType::kMatmul;
            
            // 设置输入张量
            kernel::DTensor input_a, input_b, output;
            input_a.num_dims = 2;
            input_a.dim[0] = 1024;
            input_a.dim[1] = 512;
            
            input_b.num_dims = 2;
            input_b.dim[0] = 512;
            input_b.dim[1] = 256;
            
            output.num_dims = 2;
            output.dim[0] = 1024;
            output.dim[1] = 256;
            
            op->input_tensors.push_back(input_a);
            op->input_tensors.push_back(input_b);
            op->output_tensors.push_back(output);
            
            graph->operators.push_back(std::move(op));
        }
    } else if (name == "mixed_ops") {
        // 创建混合操作图
        std::vector<kernel::KNOperatorType> op_types = {
            kernel::KNOperatorType::kMatmul,
            kernel::KNOperatorType::kElementBinary,
            kernel::KNOperatorType::kElementUnary,
            kernel::KNOperatorType::kReduction
        };
        
        for (auto op_type : op_types) {
            auto op = std::make_unique<kernel::KNOperator>();
            op->op_type = op_type;
            
            // 设置通用张量
            kernel::DTensor tensor;
            tensor.num_dims = 2;
            tensor.dim[0] = 512;
            tensor.dim[1] = 512;
            
            op->input_tensors.push_back(tensor);
            op->output_tensors.push_back(tensor);
            
            graph->operators.push_back(std::move(op));
        }
    }
    
    return graph;
}

// 基准测试函数
void benchmark_optimization(YICABackend& backend, const kernel::Graph& graph, const std::string& graph_name) {
    std::cout << "\n=== Benchmarking " << graph_name << " ===\n";
    
    // 测试传统优化
    auto start_time = std::chrono::high_resolution_clock::now();
    auto traditional_result = backend.optimize_for_yica(&graph);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto traditional_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Traditional Optimization:\n";
    std::cout << "  Time: " << traditional_time.count() << " μs\n";
    std::cout << "  Speedup: " << traditional_result.estimated_speedup << "x\n";
    std::cout << "  Memory: " << traditional_result.memory_footprint << " bytes\n";
    std::cout << "  Used RL: " << (traditional_result.used_rl_optimization ? "Yes" : "No") << "\n";
    
    // 测试 RL 优化
    start_time = std::chrono::high_resolution_clock::now();
    auto rl_result = backend.optimize_with_reinforcement_learning(&graph);
    end_time = std::chrono::high_resolution_clock::now();
    
    auto rl_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "RL Optimization:\n";
    std::cout << "  Time: " << rl_time.count() << " μs\n";
    std::cout << "  Speedup: " << rl_result.estimated_speedup << "x\n";
    std::cout << "  Memory: " << rl_result.memory_footprint << " bytes\n";
    std::cout << "  Used RL: " << (rl_result.used_rl_optimization ? "Yes" : "No") << "\n";
    
    // 计算改进
    float speedup_improvement = rl_result.estimated_speedup / traditional_result.estimated_speedup;
    float time_overhead = static_cast<float>(rl_time.count()) / traditional_time.count();
    
    std::cout << "RL vs Traditional:\n";
    std::cout << "  Speedup improvement: " << speedup_improvement << "x\n";
    std::cout << "  Time overhead: " << time_overhead << "x\n";
    
    // 输出优化日志
    if (!rl_result.optimization_log.empty()) {
        std::cout << "RL Optimization Log:\n";
        for (const auto& log : rl_result.optimization_log) {
            std::cout << "  - " << log << "\n";
        }
    }
}

// 训练演示函数
void demonstrate_training(YICABackend& backend) {
    std::cout << "\n=== RL Optimizer Training Demo ===\n";
    
    // 创建训练数据集
    std::vector<kernel::Graph> training_graphs;
    
    // 添加不同类型的图
    auto graph1 = create_test_graph("matmul_chain");
    auto graph2 = create_test_graph("mixed_ops");
    
    training_graphs.push_back(*graph1);
    training_graphs.push_back(*graph2);
    
    // 复制更多变体
    for (int i = 0; i < 5; ++i) {
        training_graphs.push_back(*graph1);
        training_graphs.push_back(*graph2);
    }
    
    std::cout << "Created training dataset with " << training_graphs.size() << " graphs\n";
    
    // 执行训练
    const size_t episodes = 200;
    std::cout << "Training RL optimizer for " << episodes << " episodes...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    backend.train_rl_optimizer(training_graphs, episodes);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto training_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Training completed in " << training_time.count() << " ms\n";
    
    // 保存模型
    const std::string model_path = "yica_rl_model.bin";
    backend.save_rl_model(model_path);
    std::cout << "Model saved to " << model_path << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "YICA Reinforcement Learning Optimizer Demo\n";
    std::cout << "==========================================\n";
    
    try {
        // 初始化 YICA 配置
        YICAConfig config;
        config.num_cim_arrays = 16;
        config.spm_size_kb = 1024;
        config.dram_bandwidth_gbps = 1600.0f;
        config.peak_tops = 100.0f;
        
        std::cout << "YICA Configuration:\n";
        std::cout << "  CIM Arrays: " << config.num_cim_arrays << "\n";
        std::cout << "  SPM Size: " << config.spm_size_kb << " KB\n";
        std::cout << "  DRAM Bandwidth: " << config.dram_bandwidth_gbps << " GB/s\n";
        std::cout << "  Peak Performance: " << config.peak_tops << " TOPS\n";
        
        // 创建 YICA 后端
        YICABackend backend(config);
        std::cout << "YICA Backend initialized successfully\n";
        
        // 创建测试图
        auto matmul_graph = create_test_graph("matmul_chain");
        auto mixed_graph = create_test_graph("mixed_ops");
        
        std::cout << "Created test graphs:\n";
        std::cout << "  MatMul Chain: " << matmul_graph->operators.size() << " operators\n";
        std::cout << "  Mixed Ops: " << mixed_graph->operators.size() << " operators\n";
        
        // 基准测试（训练前）
        std::cout << "\n--- Before Training ---\n";
        benchmark_optimization(backend, *matmul_graph, "MatMul Chain");
        benchmark_optimization(backend, *mixed_graph, "Mixed Operations");
        
        // 训练演示
        if (argc > 1 && std::string(argv[1]) == "--train") {
            demonstrate_training(backend);
            
            // 基准测试（训练后）
            std::cout << "\n--- After Training ---\n";
            benchmark_optimization(backend, *matmul_graph, "MatMul Chain (Trained)");
            benchmark_optimization(backend, *mixed_graph, "Mixed Operations (Trained)");
        } else {
            std::cout << "\nSkipping training. Use --train to enable training.\n";
        }
        
        // 模型保存/加载演示
        std::cout << "\n=== Model Persistence Demo ===\n";
        const std::string model_path = "demo_model.bin";
        
        try {
            backend.save_rl_model(model_path);
            std::cout << "Model saved successfully to " << model_path << "\n";
            
            backend.load_rl_model(model_path);
            std::cout << "Model loaded successfully from " << model_path << "\n";
            
            // 验证加载后的模型仍然工作
            auto result = backend.optimize_with_reinforcement_learning(matmul_graph.get());
            std::cout << "Post-load optimization test: " 
                      << (result.used_rl_optimization ? "PASSED" : "FAILED") << "\n";
                      
        } catch (const std::exception& e) {
            std::cout << "Model persistence error: " << e.what() << "\n";
        }
        
        std::cout << "\n=== Demo Summary ===\n";
        std::cout << "✓ YICA Backend initialization\n";
        std::cout << "✓ Test graph creation\n";
        std::cout << "✓ Traditional vs RL optimization comparison\n";
        if (argc > 1 && std::string(argv[1]) == "--train") {
            std::cout << "✓ RL optimizer training\n";
        }
        std::cout << "✓ Model persistence (save/load)\n";
        std::cout << "\nDemo completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
