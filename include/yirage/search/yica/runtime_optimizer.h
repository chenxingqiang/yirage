#pragma once

#include <memory>
#include <unordered_map>
#include <random>
#include "runtime_types.h"
#include "performance_monitor.h"

namespace yirage {
namespace search {
namespace yica {

// 强化学习优化器
class ReinforcementOptimizer {
private:
    // Q表：状态-动作价值映射
    std::unordered_map<size_t, std::unordered_map<OptimizationAction, float>> q_table_;
    
    // 学习参数
    float learning_rate_ = 0.1f;
    float discount_factor_ = 0.95f;
    float epsilon_ = 0.1f;           // 探索率
    float epsilon_decay_ = 0.999f;   // 探索率衰减
    float min_epsilon_ = 0.01f;      // 最小探索率
    
    // 随机数生成器
    mutable std::mt19937 rng_;
    
    // 状态哈希缓存
    std::unordered_map<size_t, QLearningState> state_cache_;
    
    // 计算奖励
    float compute_reward(const PerformanceMetrics& before, 
                        const PerformanceMetrics& after,
                        const OptimizationObjective& objective) const;
    
    // 获取最佳动作（贪心策略）
    OptimizationAction get_best_action(const QLearningState& state) const;
    
    // 获取随机动作（探索策略）
    OptimizationAction get_random_action() const;
    
    // 状态特征编码
    std::vector<float> encode_state_features(const QLearningState& state) const;
    
public:
    ReinforcementOptimizer();
    
    // 选择动作（ε-贪心策略）
    OptimizationAction select_action(const QLearningState& state);
    
    // 更新Q值
    void update_q_value(const QLearningState& state, 
                       OptimizationAction action, 
                       float reward, 
                       const QLearningState& next_state);
    
    // 获取Q值
    float get_q_value(const QLearningState& state, OptimizationAction action) const;
    
    // 设置学习参数
    void set_learning_rate(float lr);
    void set_discount_factor(float gamma);
    void set_epsilon(float eps);
    
    // 保存和加载Q表
    bool save_q_table(const std::string& filename) const;
    bool load_q_table(const std::string& filename);
    
    // 重置学习
    void reset();
    
    // 获取学习统计
    size_t get_state_count() const;
    float get_current_epsilon() const;
};

// 多目标优化器
class MultiObjectiveOptimizer {
private:
    struct Individual {
        YICAConfig config;
        std::vector<float> objectives;  // [性能, 能效, 延迟]
        float fitness = 0.0f;
        int rank = 0;
        float crowding_distance = 0.0f;
    };
    
    // NSGA-II参数
    size_t population_size_ = 50;
    size_t max_generations_ = 100;
    float mutation_rate_ = 0.1f;
    float crossover_rate_ = 0.8f;
    
    // 当前种群
    std::vector<Individual> population_;
    std::vector<Individual> pareto_front_;
    
    // 随机数生成器
    mutable std::mt19937 rng_;
    
    // 初始化种群
    void initialize_population(const std::vector<YICAConfig>& seed_configs);
    
    // 评估个体
    void evaluate_individual(Individual& individual, 
                           const PerformanceMetrics& baseline_metrics);
    
    // 非支配排序
    void non_dominated_sort(std::vector<Individual>& individuals);
    
    // 计算拥挤距离
    void calculate_crowding_distance(std::vector<Individual>& individuals);
    
    // 选择
    std::vector<Individual> selection(const std::vector<Individual>& population);
    
    // 交叉
    Individual crossover(const Individual& parent1, const Individual& parent2);
    
    // 变异
    void mutate(Individual& individual);
    
    // 检查支配关系
    bool dominates(const Individual& a, const Individual& b) const;
    
public:
    MultiObjectiveOptimizer();
    
    // 进化求解
    std::vector<YICAConfig> evolve_solutions(
        const std::vector<YICAConfig>& initial_population,
        const PerformanceMetrics& current_metrics,
        const OptimizationObjective& objective);
    
    // 选择最佳配置
    YICAConfig select_best_config(
        const OptimizationObjective& objective,
        const std::vector<YICAConfig>& candidates,
        const PerformanceMetrics& current_metrics);
    
    // 获取帕累托前沿
    std::vector<YICAConfig> get_pareto_front() const;
    
    // 设置进化参数
    void set_population_size(size_t size);
    void set_max_generations(size_t generations);
    void set_mutation_rate(float rate);
    void set_crossover_rate(float rate);
    
    // 重置优化器
    void reset();
};

// 自适应调度器
class AdaptiveScheduler {
private:
    // 调度策略
    enum class SchedulingPolicy {
        ROUND_ROBIN,        // 轮询调度
        PRIORITY_BASED,     // 基于优先级
        LOAD_BALANCED,      // 负载均衡
        PERFORMANCE_AWARE   // 性能感知
    };
    
    SchedulingPolicy current_policy_ = SchedulingPolicy::PERFORMANCE_AWARE;
    
    // CIM阵列状态
    struct CIMArrayState {
        int array_id;
        float utilization;
        float temperature;
        float power_consumption;
        int active_tasks;
        std::vector<int> pending_tasks;
    };
    
    std::vector<CIMArrayState> cim_arrays_;
    
    // SPM分配状态
    struct SPMAllocation {
        size_t total_size;
        size_t used_size;
        std::vector<std::pair<size_t, size_t>> allocated_blocks;  // (offset, size)
        float fragmentation_ratio;
    };
    
    SPMAllocation spm_allocation_;
    
    // 任务队列
    struct Task {
        int task_id;
        float compute_requirement;
        size_t memory_requirement;
        int priority;
        std::chrono::time_point<std::chrono::steady_clock> arrival_time;
    };
    
    std::vector<Task> pending_tasks_;
    
    // 调度算法
    std::vector<int> schedule_round_robin(const std::vector<Task>& tasks);
    std::vector<int> schedule_priority_based(const std::vector<Task>& tasks);
    std::vector<int> schedule_load_balanced(const std::vector<Task>& tasks);
    std::vector<int> schedule_performance_aware(const std::vector<Task>& tasks);
    
    // SPM分配算法
    bool allocate_spm_first_fit(size_t size, size_t& offset);
    bool allocate_spm_best_fit(size_t size, size_t& offset);
    void deallocate_spm(size_t offset, size_t size);
    void compact_spm();
    
public:
    AdaptiveScheduler(int num_cim_arrays = 16, size_t spm_size = 1024 * 1024);
    
    // 更新CIM阵列状态
    void update_cim_array_states(const std::vector<float>& utilizations);
    
    // 添加任务
    void add_task(const Task& task);
    
    // 执行调度
    std::vector<int> schedule_tasks();
    
    // SPM分配
    bool allocate_spm_memory(size_t size, size_t& offset);
    void deallocate_spm_memory(size_t offset, size_t size);
    
    // 动态调整策略
    void adjust_scheduling_policy(const PerformanceMetrics& metrics);
    
    // 获取调度统计
    float get_average_utilization() const;
    float get_spm_fragmentation() const;
    size_t get_pending_task_count() const;
    
    // 设置调度策略
    void set_scheduling_policy(SchedulingPolicy policy);
    
    // 重置调度器
    void reset();
};

// 运行时优化器主类
class RuntimeOptimizer {
private:
    std::unique_ptr<ReinforcementOptimizer> rl_optimizer_;
    std::unique_ptr<MultiObjectiveOptimizer> mo_optimizer_;
    std::unique_ptr<AdaptiveScheduler> scheduler_;
    
    // 优化历史
    struct OptimizationRecord {
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        QLearningState state;
        OptimizationAction action;
        float reward;
        YICAConfig config;
        PerformanceMetrics metrics_before;
        PerformanceMetrics metrics_after;
    };
    
    std::vector<OptimizationRecord> optimization_history_;
    mutable std::mutex history_mutex_;
    
    // 优化策略
    enum class OptimizationStrategy {
        REINFORCEMENT_LEARNING,   // 强化学习
        MULTI_OBJECTIVE,         // 多目标优化
        HYBRID                   // 混合策略
    };
    
    OptimizationStrategy strategy_ = OptimizationStrategy::HYBRID;
    
    // 性能基线
    PerformanceMetrics baseline_metrics_;
    bool baseline_set_ = false;
    
    // 配置验证
    bool validate_config(const YICAConfig& config) const;
    
    // 应用配置更改
    bool apply_config_change(const YICAConfig& new_config, 
                           OptimizationAction action);
    
    // 选择优化策略
    OptimizationStrategy select_optimization_strategy(
        const OptimizationContext& context) const;
    
public:
    RuntimeOptimizer();
    ~RuntimeOptimizer();
    
    // 初始化优化器
    bool initialize(const YICAConfig& initial_config);
    
    // 执行优化
    OptimizationResult optimize(const OptimizationContext& context);
    
    // 实时策略调整
    OptimizationResult adjust_strategy_realtime(
        const PerformanceMetrics& current_metrics,
        const WorkloadCharacteristics& workload);
    
    // 负载均衡优化
    OptimizationResult optimize_load_balancing(
        const std::vector<float>& cim_utilizations);
    
    // 资源重分配
    OptimizationResult redistribute_resources(
        const PerformanceMetrics& metrics);
    
    // 执行路径优化
    OptimizationResult optimize_execution_path(
        const WorkloadCharacteristics& workload);
    
    // 设置性能基线
    void set_performance_baseline(const PerformanceMetrics& baseline);
    
    // 获取优化历史
    std::vector<OptimizationRecord> get_optimization_history(size_t count = 100) const;
    
    // 获取优化统计
    struct OptimizationStats {
        size_t total_optimizations = 0;
        size_t successful_optimizations = 0;
        float average_improvement = 0.0f;
        float average_optimization_time = 0.0f;
        std::unordered_map<OptimizationAction, size_t> action_counts;
    };
    
    OptimizationStats get_optimization_stats() const;
    
    // 设置优化策略
    void set_optimization_strategy(OptimizationStrategy strategy);
    
    // 保存和加载优化器状态
    bool save_optimizer_state(const std::string& filename) const;
    bool load_optimizer_state(const std::string& filename);
    
    // 重置优化器
    void reset();
};

}  // namespace yica
}  // namespace search
}  // namespace yirage 