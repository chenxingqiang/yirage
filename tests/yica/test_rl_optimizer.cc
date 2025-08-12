/**
 * @file test_rl_optimizer.cc
 * @brief YICA 强化学习优化器测试
 * 
 * 测试强化学习优化器的核心功能：
 * - 状态特征提取
 * - 动作生成和选择
 * - 奖励计算
 * - 模型训练和推理
 * - 与 YICA 后端的集成
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "yirage/yica/yica_backend.h"
#include "yirage/yica/config.h"
#include "yirage/kernel/graph.h"
#include "yirage/kernel/operator.h"

namespace yirage {
namespace yica {

class YICAReinforcementLearningOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置测试配置
        config_.num_cim_arrays = 8;
        config_.spm_size_kb = 512;
        config_.dram_bandwidth_gbps = 1000.0f;
        config_.peak_tops = 50.0f;
        
        // 创建 YICA 后端
        backend_ = std::make_unique<YICABackend>(config_);
        
        // 创建测试图
        test_graph_ = create_test_graph();
    }
    
    void TearDown() override {
        backend_.reset();
    }
    
    kernel::Graph create_test_graph() {
        kernel::Graph graph;
        
        // 创建简单的矩阵乘法 + 激活函数图
        auto matmul_op = std::make_unique<kernel::KNOperator>();
        matmul_op->op_type = kernel::KNOperatorType::kMatmul;
        
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
        
        matmul_op->input_tensors.push_back(input_a);
        matmul_op->input_tensors.push_back(input_b);
        matmul_op->output_tensors.push_back(output);
        
        graph.operators.push_back(std::move(matmul_op));
        
        // 添加激活函数
        auto activation_op = std::make_unique<kernel::KNOperator>();
        activation_op->op_type = kernel::KNOperatorType::kElementUnary;
        activation_op->input_tensors.push_back(output);
        activation_op->output_tensors.push_back(output);
        
        graph.operators.push_back(std::move(activation_op));
        
        return graph;
    }
    
    YICAConfig config_;
    std::unique_ptr<YICABackend> backend_;
    kernel::Graph test_graph_;
};

// 测试强化学习优化器的基本功能
TEST_F(YICAReinforcementLearningOptimizerTest, BasicOptimization) {
    ASSERT_TRUE(backend_ != nullptr);
    
    // 执行强化学习优化
    auto result = backend_->optimize_with_reinforcement_learning(&test_graph_);
    
    // 验证结果
    EXPECT_TRUE(result.used_rl_optimization);
    EXPECT_FALSE(result.yis_kernel_code.empty());
    EXPECT_FALSE(result.triton_kernel_code.empty());
    EXPECT_GT(result.estimated_speedup, 0.0f);
    EXPECT_FALSE(result.optimization_log.empty());
    
    // 验证日志包含 RL 相关信息
    bool found_rl_log = false;
    for (const auto& log : result.optimization_log) {
        if (log.find("RL") != std::string::npos) {
            found_rl_log = true;
            break;
        }
    }
    EXPECT_TRUE(found_rl_log);
}

// 测试传统优化与强化学习优化的比较
TEST_F(YICAReinforcementLearningOptimizerTest, CompareWithTraditionalOptimization) {
    // 执行传统优化
    auto traditional_result = backend_->optimize_for_yica(&test_graph_);
    
    // 执行强化学习优化
    auto rl_result = backend_->optimize_with_reinforcement_learning(&test_graph_);
    
    // 比较结果
    EXPECT_FALSE(traditional_result.used_rl_optimization);
    EXPECT_TRUE(rl_result.used_rl_optimization);
    
    // RL 优化应该至少不比传统优化差
    EXPECT_GE(rl_result.estimated_speedup, traditional_result.estimated_speedup * 0.9f);
}

// 测试模型保存和加载
TEST_F(YICAReinforcementLearningOptimizerTest, ModelSaveAndLoad) {
    const std::string model_path = "/tmp/test_rl_model.bin";
    
    // 保存模型
    EXPECT_NO_THROW(backend_->save_rl_model(model_path));
    
    // 加载模型
    EXPECT_NO_THROW(backend_->load_rl_model(model_path));
    
    // 验证加载后仍能正常工作
    auto result = backend_->optimize_with_reinforcement_learning(&test_graph_);
    EXPECT_TRUE(result.used_rl_optimization);
}

// 测试训练功能
TEST_F(YICAReinforcementLearningOptimizerTest, Training) {
    // 创建训练数据集
    std::vector<kernel::Graph> training_graphs;
    
    for (int i = 0; i < 10; ++i) {
        training_graphs.push_back(create_test_graph());
    }
    
    // 执行训练
    EXPECT_NO_THROW(backend_->train_rl_optimizer(training_graphs, 50));
    
    // 验证训练后的优化效果
    auto result = backend_->optimize_with_reinforcement_learning(&test_graph_);
    EXPECT_TRUE(result.used_rl_optimization);
}

// 测试空图的处理
TEST_F(YICAReinforcementLearningOptimizerTest, EmptyGraphHandling) {
    kernel::Graph empty_graph;
    
    // 空图应该能正常处理
    auto result = backend_->optimize_with_reinforcement_learning(&empty_graph);
    
    // 应该回退到传统优化
    EXPECT_FALSE(result.used_rl_optimization);
}

// 测试异常情况处理
TEST_F(YICAReinforcementLearningOptimizerTest, ExceptionHandling) {
    // 测试空指针
    auto result = backend_->optimize_with_reinforcement_learning(nullptr);
    EXPECT_FALSE(result.used_rl_optimization);
    
    // 测试无效模型路径
    EXPECT_THROW(backend_->load_rl_model("/invalid/path/model.bin"), std::runtime_error);
}

// 性能基准测试
TEST_F(YICAReinforcementLearningOptimizerTest, PerformanceBenchmark) {
    const int num_iterations = 10;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto result = backend_->optimize_with_reinforcement_learning(&test_graph_);
        EXPECT_TRUE(result.used_rl_optimization);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "RL Optimization Performance: " 
              << duration.count() / num_iterations 
              << " ms per optimization" << std::endl;
    
    // 确保优化时间在合理范围内（每次少于1秒）
    EXPECT_LT(duration.count() / num_iterations, 1000);
}

// 测试不同图结构的适应性
TEST_F(YICAReinforcementLearningOptimizerTest, GraphStructureAdaptability) {
    std::vector<kernel::Graph> different_graphs;
    
    // 创建不同类型的图结构
    
    // 1. 纯矩阵乘法图
    kernel::Graph matmul_only_graph;
    for (int i = 0; i < 5; ++i) {
        auto op = std::make_unique<kernel::KNOperator>();
        op->op_type = kernel::KNOperatorType::kMatmul;
        
        kernel::DTensor input, output;
        input.num_dims = 2;
        input.dim[0] = 256;
        input.dim[1] = 256;
        output.num_dims = 2;
        output.dim[0] = 256;
        output.dim[1] = 256;
        
        op->input_tensors.push_back(input);
        op->output_tensors.push_back(output);
        
        matmul_only_graph.operators.push_back(std::move(op));
    }
    different_graphs.push_back(matmul_only_graph);
    
    // 2. 纯逐元素操作图
    kernel::Graph elementwise_only_graph;
    for (int i = 0; i < 8; ++i) {
        auto op = std::make_unique<kernel::KNOperator>();
        op->op_type = (i % 2 == 0) ? kernel::KNOperatorType::kElementBinary : 
                                    kernel::KNOperatorType::kElementUnary;
        
        kernel::DTensor tensor;
        tensor.num_dims = 3;
        tensor.dim[0] = 64;
        tensor.dim[1] = 64;
        tensor.dim[2] = 128;
        
        op->input_tensors.push_back(tensor);
        op->output_tensors.push_back(tensor);
        
        elementwise_only_graph.operators.push_back(std::move(op));
    }
    different_graphs.push_back(elementwise_only_graph);
    
    // 测试每种图结构
    for (size_t i = 0; i < different_graphs.size(); ++i) {
        auto result = backend_->optimize_with_reinforcement_learning(&different_graphs[i]);
        
        EXPECT_TRUE(result.used_rl_optimization) 
            << "Failed for graph structure " << i;
        EXPECT_GT(result.estimated_speedup, 0.0f) 
            << "Invalid speedup for graph structure " << i;
    }
}

// 集成测试：端到端优化流程
TEST_F(YICAReinforcementLearningOptimizerTest, EndToEndOptimization) {
    // 1. 创建复杂的测试图
    kernel::Graph complex_graph;
    
    // 添加多种操作类型
    std::vector<kernel::KNOperatorType> op_types = {
        kernel::KNOperatorType::kMatmul,
        kernel::KNOperatorType::kElementBinary,
        kernel::KNOperatorType::kElementUnary,
        kernel::KNOperatorType::kReduction,
        kernel::KNOperatorType::kRMSNorm
    };
    
    for (auto op_type : op_types) {
        auto op = std::make_unique<kernel::KNOperator>();
        op->op_type = op_type;
        
        // 设置合适的张量维度
        kernel::DTensor tensor;
        tensor.num_dims = 2;
        tensor.dim[0] = 512;
        tensor.dim[1] = 512;
        
        op->input_tensors.push_back(tensor);
        op->output_tensors.push_back(tensor);
        
        complex_graph.operators.push_back(std::move(op));
    }
    
    // 2. 执行完整的优化流程
    auto result = backend_->optimize_with_reinforcement_learning(&complex_graph);
    
    // 3. 验证结果完整性
    EXPECT_TRUE(result.used_rl_optimization);
    EXPECT_FALSE(result.yis_kernel_code.empty());
    EXPECT_FALSE(result.triton_kernel_code.empty());
    EXPECT_GT(result.estimated_speedup, 0.5f);  // 至少有一些加速
    EXPECT_GT(result.memory_footprint, 0);
    EXPECT_FALSE(result.optimization_log.empty());
    
    // 4. 验证生成的代码包含关键信息
    EXPECT_TRUE(result.yis_kernel_code.find("yica_optimized_kernel") != std::string::npos);
    EXPECT_TRUE(result.triton_kernel_code.find("@triton.jit") != std::string::npos);
    EXPECT_TRUE(result.triton_kernel_code.find("CIM_ARRAYS") != std::string::npos);
}

} // namespace yica
} // namespace yirage

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
