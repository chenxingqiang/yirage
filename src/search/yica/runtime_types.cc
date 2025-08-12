#include "yirage/search/yica/runtime_types.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace yirage {
namespace search {
namespace yica {

// PerformanceMetrics 实现
float PerformanceMetrics::compute_performance_score() const {
    // 综合性能得分计算：加权平均
    float score = 0.0f;
    score += cim_utilization * 0.3f;           // CIM利用率权重30%
    score += spm_hit_rate * 0.2f;              // SPM命中率权重20%
    score += (1.0f - memory_bandwidth_usage) * 0.15f;  // 内存带宽使用率（越低越好）权重15%
    score += std::min(compute_throughput / 100.0f, 1.0f) * 0.25f;  // 计算吞吐量权重25%
    score += std::max(0.0f, 1.0f - latency_ms / 100.0f) * 0.1f;   // 延迟（越低越好）权重10%
    
    return std::max(0.0f, std::min(1.0f, score));
}

bool PerformanceMetrics::is_valid() const {
    return cim_utilization >= 0.0f && cim_utilization <= 1.0f &&
           spm_hit_rate >= 0.0f && spm_hit_rate <= 1.0f &&
           memory_bandwidth_usage >= 0.0f && memory_bandwidth_usage <= 1.0f &&
           compute_throughput >= 0.0f &&
           power_consumption >= 0.0f &&
           latency_ms >= 0.0f;
}

// WorkloadCharacteristics 实现
float WorkloadCharacteristics::compute_complexity() const {
    // 工作负载复杂度：计算密集度和内存密集度的组合
    float complexity = 0.0f;
    complexity += compute_intensity * 0.4f;
    complexity += memory_intensity * 0.3f;
    complexity += (1.0f - data_reuse_factor) * 0.2f;  // 数据重用越少，复杂度越高
    complexity += batch_size_variability * 0.1f;
    
    return std::max(0.0f, std::min(1.0f, complexity));
}

// OptimizationObjective 实现
bool OptimizationObjective::validate_weights() const {
    float sum = performance_weight + energy_weight + latency_weight;
    return std::abs(sum - 1.0f) < 1e-6f;
}

float OptimizationObjective::compute_objective_value(const PerformanceMetrics& metrics) const {
    float performance_score = metrics.compute_performance_score();
    float energy_score = 1.0f - std::min(metrics.power_consumption / 300.0f, 1.0f);  // 假设300W为最大功耗
    float latency_score = std::max(0.0f, 1.0f - metrics.latency_ms / 100.0f);  // 假设100ms为最大可接受延迟
    
    return performance_score * performance_weight +
           energy_score * energy_weight +
           latency_score * latency_weight;
}

// RuntimeState 实现
void RuntimeState::add_metrics_to_history(const PerformanceMetrics& metrics) {
    metrics_history.push_back(metrics);
    if (metrics_history.size() > MAX_HISTORY_SIZE) {
        metrics_history.pop_front();
    }
}

std::vector<PerformanceMetrics> RuntimeState::get_recent_metrics(size_t count) const {
    std::vector<PerformanceMetrics> recent;
    size_t start_idx = metrics_history.size() > count ? metrics_history.size() - count : 0;
    
    for (size_t i = start_idx; i < metrics_history.size(); ++i) {
        recent.push_back(metrics_history[i]);
    }
    
    return recent;
}

void RuntimeState::cleanup_old_history() {
    if (metrics_history.size() > MAX_HISTORY_SIZE) {
        size_t excess = metrics_history.size() - MAX_HISTORY_SIZE;
        for (size_t i = 0; i < excess; ++i) {
            metrics_history.pop_front();
        }
    }
}

// OptimizationContext 实现
bool OptimizationContext::validate() const {
    return objective.validate_weights() &&
           optimization_budget_ms > 0.0f &&
           !candidate_configs.empty();
}

// OptimizationResult 实现
OptimizationResult OptimizationResult::create_success(
    const YICAConfig& config,
    OptimizationAction action,
    float improvement,
    const std::string& reason) {
    
    OptimizationResult result;
    result.success = true;
    result.recommended_config = config;
    result.recommended_action = action;
    result.expected_improvement = improvement;
    result.optimization_reason = reason;
    return result;
}

OptimizationResult OptimizationResult::create_failure(const std::string& reason) {
    OptimizationResult result;
    result.success = false;
    result.optimization_reason = reason;
    result.recommended_action = OptimizationAction::NO_ACTION;
    result.expected_improvement = 0.0f;
    return result;
}

// TimeSeriesFeatures 实现
void TimeSeriesFeatures::build_from_metrics_history(
    const std::vector<PerformanceMetrics>& history) {
    
    cim_utilization_history.clear();
    memory_access_pattern.clear();
    workload_characteristics.clear();
    
    for (const auto& metrics : history) {
        cim_utilization_history.push_back(metrics.cim_utilization);
        memory_access_pattern.push_back(metrics.memory_bandwidth_usage);
        
        // 构建工作负载特征
        float compute_ratio = metrics.compute_throughput / 
                             std::max(metrics.compute_throughput + metrics.memory_bandwidth_usage * 100.0f, 1.0f);
        workload_characteristics.push_back(compute_ratio);
    }
    
    // 限制序列长度
    if (cim_utilization_history.size() > static_cast<size_t>(sequence_length)) {
        cim_utilization_history.resize(sequence_length);
        memory_access_pattern.resize(sequence_length);
        workload_characteristics.resize(sequence_length);
    }
}

bool TimeSeriesFeatures::is_valid() const {
    return !cim_utilization_history.empty() &&
           !memory_access_pattern.empty() &&
           !workload_characteristics.empty() &&
           cim_utilization_history.size() == memory_access_pattern.size() &&
           memory_access_pattern.size() == workload_characteristics.size();
}

std::vector<float> TimeSeriesFeatures::to_feature_vector() const {
    std::vector<float> features;
    features.reserve(cim_utilization_history.size() + 
                    memory_access_pattern.size() + 
                    workload_characteristics.size());
    
    features.insert(features.end(), cim_utilization_history.begin(), cim_utilization_history.end());
    features.insert(features.end(), memory_access_pattern.begin(), memory_access_pattern.end());
    features.insert(features.end(), workload_characteristics.begin(), workload_characteristics.end());
    
    return features;
}

// QLearningState 实现
size_t QLearningState::compute_hash() const {
    size_t hash = 0;
    
    // 简单的哈希组合函数
    auto combine_hash = [&hash](size_t new_hash) {
        hash ^= new_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    };
    
    // 对性能指标进行量化后哈希
    combine_hash(std::hash<int>{}(static_cast<int>(current_metrics.cim_utilization * 100)));
    combine_hash(std::hash<int>{}(static_cast<int>(current_metrics.spm_hit_rate * 100)));
    combine_hash(std::hash<int>{}(static_cast<int>(current_metrics.memory_bandwidth_usage * 100)));
    
    // 对配置进行哈希
    combine_hash(std::hash<size_t>{}(current_config.num_cim_dies));
    combine_hash(std::hash<float>{}(current_config.cim_frequency));
    combine_hash(std::hash<size_t>{}(current_config.spm_size_per_die));
    
    // 对工作负载特征进行哈希
    combine_hash(std::hash<int>{}(static_cast<int>(workload.compute_intensity * 100)));
    combine_hash(std::hash<int>{}(static_cast<int>(workload.memory_intensity * 100)));
    
    return hash;
}

bool QLearningState::operator==(const QLearningState& other) const {
    const float tolerance = 1e-3f;
    
    auto float_equal = [tolerance](float a, float b) {
        return std::abs(a - b) < tolerance;
    };
    
    return float_equal(current_metrics.cim_utilization, other.current_metrics.cim_utilization) &&
           float_equal(current_metrics.spm_hit_rate, other.current_metrics.spm_hit_rate) &&
           float_equal(current_metrics.memory_bandwidth_usage, other.current_metrics.memory_bandwidth_usage) &&
           current_config.num_cim_dies == other.current_config.num_cim_dies &&
           current_config.cim_frequency == other.current_config.cim_frequency &&
           current_config.spm_size_per_die == other.current_config.spm_size_per_die &&
           float_equal(workload.compute_intensity, other.workload.compute_intensity) &&
           float_equal(workload.memory_intensity, other.workload.memory_intensity);
}

// ConfigChangeRequest 实现
bool ConfigChangeRequest::is_expired(int timeout_ms) const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - request_time);
    return elapsed.count() > timeout_ms;
}

// PerformanceAnomaly 实现
PerformanceAnomaly::Severity PerformanceAnomaly::get_severity_level() const {
    if (severity >= 0.8f) return Severity::CRITICAL;
    if (severity >= 0.6f) return Severity::HIGH;
    if (severity >= 0.4f) return Severity::MEDIUM;
    return Severity::LOW;
}

}  // namespace yica
}  // namespace search
}  // namespace yirage 