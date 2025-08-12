#pragma once

#include "yirage/transpiler/transpiler.h"
#include "yirage/kernel/graph.h"
#include "yirage/yica/config.h"
#include "yirage/yica/yis_instruction_set.h"
#include "yirage/yica/cim_resource_manager.h"
#include "yirage/yica/spm_memory_manager.h"

namespace yirage {
namespace yica {

// 前置类型声明
struct MemoryHotspot {
    std::string location;
    size_t access_count;
    float bandwidth_utilization;
};

struct PerformanceBottleneck {
    std::string component;
    std::string description;
    float severity_score;
};

struct OptimizationRecommendation {
    std::string category;
    std::string suggestion;
    float estimated_improvement;
};

// 前置类声明
class ComputeIntensityAnalyzer;
class MemoryAccessAnalyzer; 
class CIMUtilizationAnalyzer;

// YICA 专用后端实现
class YICABackend {
public:
    explicit YICABackend(const YICAConfig& config);
    ~YICABackend();

    // 主要接口：将 Yirage 计算图转译为 YICA 优化代码
    transpiler::TranspileResult transpile(kernel::Graph const* graph);
    
    // YICA 特定优化接口
    struct YICAOptimizationResult {
        std::string yis_kernel_code;           // 生成的 YIS 内核代码
        std::string triton_kernel_code;        // 生成的 Triton 内核代码
        CIMAllocation cim_allocation;  // CIM 资源分配方案
        SPMMemoryPlan spm_memory_plan;        // SPM 内存分配计划
        float estimated_speedup;              // 预估加速比
        size_t memory_footprint;              // 内存占用
        std::vector<std::string> optimization_log; // 优化日志
        bool used_rl_optimization;            // 是否使用了强化学习优化
    };
    
    // 执行 YICA 专用优化
    YICAOptimizationResult optimize_for_yica(kernel::Graph const* graph);
    
    // 使用强化学习执行优化
    YICAOptimizationResult optimize_with_reinforcement_learning(kernel::Graph const* graph);
    
    // 训练强化学习优化器
    void train_rl_optimizer(const std::vector<kernel::Graph>& training_graphs, size_t episodes);
    
    // 保存和加载强化学习模型
    void save_rl_model(const std::string& path);
    void load_rl_model(const std::string& path);
    
    // 性能分析接口
    struct PerformanceAnalysis {
        float compute_intensity;              // 计算密度
        float memory_bandwidth_requirement;   // 内存带宽需求
        float cim_friendliness_score;         // CIM 友好度评分
        std::vector<std::string> bottlenecks; // 瓶颈分析
    };
    
    PerformanceAnalysis analyze_performance(kernel::Graph const* graph);

private:
    // 核心组件
    YICAConfig config_;
    std::unique_ptr<YISInstructionSet> yis_generator_;
    std::unique_ptr<CIMResourceManager> cim_manager_;
    std::unique_ptr<SPMMemoryManager> spm_manager_;
    std::unique_ptr<YICAOptimizer> graph_optimizer_;
    std::unique_ptr<YICAReinforcementLearningOptimizer> rl_optimizer_;
    
    // 内部优化方法
    kernel::Graph apply_yica_graph_optimizations(const kernel::Graph& graph);
    std::string generate_yis_code(const kernel::Graph& optimized_graph);
    std::string generate_triton_wrapper(const std::string& yis_code);
    CIMAllocation allocate_cim_resources(const kernel::Graph& graph);
    SPMMemoryPlan plan_spm_memory(const kernel::Graph& graph);
    
    // 具体的图优化方法
    kernel::Graph apply_cim_data_reuse_optimization(const kernel::Graph& graph);
    kernel::Graph apply_operator_fusion_optimization(const kernel::Graph& graph);
    kernel::Graph apply_memory_layout_optimization(const kernel::Graph& graph);
    kernel::Graph apply_yica_specific_optimizations(const kernel::Graph& graph);
    bool validate_optimized_graph(const kernel::Graph& graph);
    
    // 辅助优化方法
    std::unique_ptr<kernel::KNOperator> create_data_reuse_optimized_operation(
        kernel::KNOperator* original_op, 
        const std::map<kernel::DTensor*, std::vector<kernel::KNOperator*>>& tensor_consumers);
    bool can_fuse_with_chain(const std::vector<kernel::KNOperator*>& fusion_chain, 
                           kernel::KNOperator* candidate);
    std::unique_ptr<kernel::KNOperator> create_fused_operation(
        const std::vector<kernel::KNOperator*>& ops_to_fuse);
    std::unique_ptr<kernel::KNOperator> clone_operation(kernel::KNOperator* op);
    
    // 内存访问模式分析
    struct MemoryAccessPattern;
    void analyze_tensor_access_patterns(const kernel::Graph& graph,
        std::map<kernel::DTensor*, MemoryAccessPattern>& patterns);
    void analyze_matmul_access_pattern(kernel::KNOperator* op, 
        std::map<kernel::DTensor*, MemoryAccessPattern>& patterns);
    void analyze_elementwise_access_pattern(kernel::KNOperator* op, 
        std::map<kernel::DTensor*, MemoryAccessPattern>& patterns);
    void analyze_reduction_access_pattern(kernel::KNOperator* op, 
        std::map<kernel::DTensor*, MemoryAccessPattern>& patterns);
    void analyze_normalization_access_pattern(kernel::KNOperator* op, 
        std::map<kernel::DTensor*, MemoryAccessPattern>& patterns);
    void analyze_generic_access_pattern(kernel::KNOperator* op, 
        std::map<kernel::DTensor*, MemoryAccessPattern>& patterns);
    void calculate_access_pattern_scores(kernel::DTensor* tensor, MemoryAccessPattern& pattern);
    
    // 布局优化方法
    layout::DmemLayout select_optimal_layout(kernel::DTensor* tensor, const MemoryAccessPattern& pattern);
    bool requires_layout_transformation(kernel::DTensor* tensor, const layout::DmemLayout& target_layout);
    std::unique_ptr<kernel::KNOperator> create_layout_optimized_operation(
        kernel::KNOperator* op, const std::map<kernel::DTensor*, layout::DmemLayout>& optimal_layouts);
    
    // YICA 特定优化方法
    std::unique_ptr<kernel::KNOperator> apply_yica_operation_optimization(kernel::KNOperator* op);
    std::vector<std::unique_ptr<kernel::KNOperator>> apply_global_yica_optimizations(
        std::vector<std::unique_ptr<kernel::KNOperator>>& ops);
    bool validate_graph_data_flow(const kernel::Graph& graph);
    
    // 标记和辅助方法
    void mark_tensor_for_spm_caching(kernel::DTensor* tensor);
    void mark_tensor_for_layout_transformation(kernel::DTensor* tensor, const layout::DmemLayout& target_layout);
    size_t estimate_fusion_chain_spm_usage(const std::vector<kernel::KNOperator*>& fusion_chain);
    size_t estimate_operation_spm_usage(kernel::KNOperator* op);
    kernel::KNOperator::KNOperatorType determine_fused_operation_type(const std::vector<kernel::KNOperator*>& ops_to_fuse);
    void add_fusion_metadata(kernel::KNOperator* fused_op, const std::vector<kernel::KNOperator*>& original_ops);
    
    // YICA 操作优化方法
    void apply_yica_matmul_optimization(kernel::KNOperator* op);
    void apply_yica_elementwise_optimization(kernel::KNOperator* op);
    void apply_yica_reduction_optimization(kernel::KNOperator* op);
    void apply_yica_normalization_optimization(kernel::KNOperator* op);
    void apply_yica_generic_optimization(kernel::KNOperator* op);
    
    // 全局优化方法
    void apply_cim_load_balancing(std::vector<std::unique_ptr<kernel::KNOperator>>& ops);
    void apply_spm_allocation_optimization(std::vector<std::unique_ptr<kernel::KNOperator>>& ops);
    void apply_dataflow_optimization(std::vector<std::unique_ptr<kernel::KNOperator>>& ops);
    void apply_instruction_level_parallelism(std::vector<std::unique_ptr<kernel::KNOperator>>& ops);
    
    // 并行执行分析
    bool can_execute_in_parallel(kernel::KNOperator* op1, kernel::KNOperator* op2);
    bool estimate_resource_conflict(kernel::KNOperator* op1, kernel::KNOperator* op2);
    
    // 策略设置方法（简化实现）
    void set_cim_allocation_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_spm_caching_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_data_reuse_pattern(kernel::KNOperator* op, const std::string& pattern);
    void set_parallelization_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_vectorization_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_memory_coalescing_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_cim_parallelism_level(kernel::KNOperator* op, uint32_t level);
    void set_reduction_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_accumulation_buffer_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_multilevel_reduction(kernel::KNOperator* op, bool enable);
    void set_statistics_caching_strategy(kernel::KNOperator* op, const std::string& strategy);
    void set_multipass_optimization(kernel::KNOperator* op, const std::string& strategy);
    void set_numerical_stability_mode(kernel::KNOperator* op, const std::string& mode);
    void set_memory_access_pattern(kernel::KNOperator* op, const std::string& pattern);
    void set_basic_parallelization(kernel::KNOperator* op, bool enable);
    void set_basic_caching_strategy(kernel::KNOperator* op, const std::string& strategy);
    void assign_operation_to_cim(kernel::KNOperator* op, size_t cim_index);
    float estimate_operation_computational_load(kernel::KNOperator* op);
    void mark_tensor_for_spm_allocation(kernel::DTensor* tensor, size_t size);
    void optimize_tensor_broadcast(kernel::DTensor* tensor, const std::vector<size_t>& consumers);
    void mark_for_parallel_execution(kernel::KNOperator* op, const std::vector<size_t>& parallel_group);
    
    // 优化策略集合
    // std::vector<std::unique_ptr<YICAOptimizationPass>> optimization_passes_;
    
    void initialize_optimization_passes();
};

// YICA 优化 Pass 基类
class YICAOptimizationPass {
public:
    virtual ~YICAOptimizationPass() = default;
    
    struct PassResult {
        kernel::Graph transformed_graph;
        bool applied;
        std::string description;
        float estimated_benefit;
    };
    
    virtual PassResult apply(const kernel::Graph& graph, const YICAConfig& config) = 0;
    virtual std::string get_pass_name() const = 0;
    virtual bool is_applicable(const kernel::Graph& graph) const = 0;
};

// CIM 数据重用优化 Pass
class CIMDataReuseOptimizationPass : public YICAOptimizationPass {
public:
    PassResult apply(const kernel::Graph& graph, const YICAConfig& config) override;
    std::string get_pass_name() const override { return "CIM Data Reuse Optimization"; }
    bool is_applicable(const kernel::Graph& graph) const override;

private:
    struct DataReusePattern {
        kernel::DTensor* tensor;
        std::vector<kernel::KNOperator*> consumers;
        float reuse_factor;
        size_t spm_cache_requirement;
        float estimated_speedup;
    };
    
    std::vector<DataReusePattern> identify_reuse_patterns(const kernel::Graph& graph);
    kernel::Graph implement_data_reuse(const kernel::Graph& graph, 
                                      const std::vector<DataReusePattern>& patterns);
};

// CIM 算子融合优化 Pass
class CIMOperatorFusionPass : public YICAOptimizationPass {
public:
    PassResult apply(const kernel::Graph& graph, const YICAConfig& config) override;
    std::string get_pass_name() const override { return "CIM Operator Fusion"; }
    bool is_applicable(const kernel::Graph& graph) const override;

private:
    struct FusionCandidate {
        std::vector<kernel::KNOperator*> operators;
        std::string fusion_type;
        float cim_efficiency_gain;
        size_t spm_requirement;
        std::string yis_template;
    };
    
    std::vector<FusionCandidate> identify_fusion_opportunities(const kernel::Graph& graph);
    kernel::Graph apply_operator_fusion(const kernel::Graph& graph, 
                                       const std::vector<FusionCandidate>& candidates);
};

// SPM 内存布局优化 Pass
class SPMMemoryLayoutOptimizationPass : public YICAOptimizationPass {
public:
    PassResult apply(const kernel::Graph& graph, const YICAConfig& config) override;
    std::string get_pass_name() const override { return "SPM Memory Layout Optimization"; }
    bool is_applicable(const kernel::Graph& graph) const override;

private:
    struct LayoutOptimization {
        kernel::DTensor* tensor;
        layout::DmemLayout original_layout;
        layout::DmemLayout optimized_layout;
        float access_efficiency_gain;
        size_t spm_footprint;
    };
    
    std::vector<LayoutOptimization> analyze_memory_layouts(const kernel::Graph& graph);
    kernel::Graph apply_layout_optimizations(const kernel::Graph& graph, 
                                            const std::vector<LayoutOptimization>& optimizations);
};

// YICA 性能分析器
class YICAPerformanceAnalyzer {
public:
    explicit YICAPerformanceAnalyzer(const YICAConfig& config);
    
    struct DetailedAnalysis {
        // 计算分析
        struct ComputeAnalysis {
            float total_flops;
            float peak_flops_utilization;
            std::map<std::string, float> op_type_distribution;
            float cim_compute_efficiency;
        } compute;
        
        // 内存分析
        struct MemoryAnalysis {
            size_t total_memory_access;
            float spm_hit_rate;
            float dram_bandwidth_utilization;
            std::vector<MemoryHotspot> hotspots;
        } memory;
        
        // CIM 阵列分析
        struct CIMAnalysis {
            std::vector<float> array_utilization;
            float load_balance_score;
            float parallel_efficiency;
        } cim;
        
        // 瓶颈分析
        std::vector<PerformanceBottleneck> bottlenecks;
        std::vector<OptimizationRecommendation> recommendations;
    };
    
    DetailedAnalysis analyze(const kernel::Graph& graph);
    
private:
    YICAConfig config_;
    std::unique_ptr<ComputeIntensityAnalyzer> compute_analyzer_;
    std::unique_ptr<MemoryAccessAnalyzer> memory_analyzer_;
    std::unique_ptr<CIMUtilizationAnalyzer> cim_analyzer_;
};

// 强化学习优化器
class YICAReinforcementLearningOptimizer {
public:
    explicit YICAReinforcementLearningOptimizer(const YICAConfig& config);
    ~YICAReinforcementLearningOptimizer();

    // 强化学习状态表示
    struct RLState {
        float compute_intensity;
        float memory_bandwidth_usage;
        float cim_utilization;
        size_t graph_size;
        std::vector<float> feature_vector;
    };

    // 强化学习动作表示
    struct RLAction {
        enum class ActionType {
            FUSION,            // 算子融合
            DATA_REUSE,        // 数据重用
            LAYOUT_TRANSFORM,  // 布局转换
            PARALLELIZATION,   // 并行化策略
            MEMORY_ALLOCATION, // 内存分配
            INSTRUCTION_REORDERING // 指令重排序
        };

        ActionType type;
        float value;  // 动作参数值
        std::string target; // 目标操作或张量
    };

    // 强化学习奖励
    struct RLReward {
        float performance_gain;
        float memory_efficiency;
        float total_reward;
    };

    // 强化学习优化接口
    kernel::Graph optimize_graph_with_rl(const kernel::Graph& graph);
    
    // 训练接口
    void train(const std::vector<kernel::Graph>& training_graphs, size_t episodes);
    
    // 保存和加载模型
    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    // 核心 RL 组件
    YICAConfig config_;
    
    // RL 策略网络参数
    std::vector<std::vector<float>> policy_weights_;
    std::vector<std::vector<float>> value_weights_;
    
    // 经验回放缓冲区
    struct Experience {
        RLState state;
        RLAction action;
        float reward;
        RLState next_state;
        bool done;
    };
    std::vector<Experience> replay_buffer_;
    
    // RL 核心方法
    RLState extract_state_features(const kernel::Graph& graph);
    std::vector<RLAction> generate_possible_actions(const kernel::Graph& graph, const RLState& state);
    RLAction select_action(const RLState& state, const std::vector<RLAction>& possible_actions, bool explore);
    RLReward calculate_reward(const kernel::Graph& original_graph, const kernel::Graph& optimized_graph);
    void update_policy(const Experience& experience);
    
    // 辅助方法
    kernel::Graph apply_action(const kernel::Graph& graph, const RLAction& action);
    float predict_action_value(const RLState& state, const RLAction& action);
    void optimize_model();
    
    // 超参数
    float learning_rate_;
    float discount_factor_;
    float exploration_rate_;
    size_t batch_size_;
    size_t replay_buffer_capacity_;
};

} // namespace yica
} // namespace yirage 