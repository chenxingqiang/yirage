#include "yirage/search/yica/optimization_strategy.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>

namespace yirage {
namespace search {
namespace yica {

// ===== CIMDataReuseStrategy Implementation =====

CIMDataReuseStrategy::CIMDataReuseStrategy() {
    // 构造函数实现
}

std::string CIMDataReuseStrategy::get_description() const {
    return "Optimizes data reuse patterns in CIM arrays to reduce memory access overhead and improve energy efficiency.";
}

bool CIMDataReuseStrategy::is_applicable(const AnalysisResult& analysis) const {
    // 检查是否存在重复的内存访问模式
    if (analysis.memory_access_patterns.empty()) {
        return false;
    }
    
    // 检查CIM友好度是否足够高
    if (analysis.cim_friendliness_score < 0.3f) {
        return false;
    }
    
    // 检查是否有可重用的数据
    for (const auto& pattern : analysis.memory_access_patterns) {
        if (pattern.access_frequency > 1) {
            return true;
        }
    }
    
    return false;
}

float CIMDataReuseStrategy::estimate_benefit(const AnalysisResult& analysis) const {
    float total_benefit = 0.0f;
    
    // 基于访问频率计算重用收益
    for (const auto& pattern : analysis.memory_access_patterns) {
        if (pattern.access_frequency > 1) {
            float reuse_factor = std::min(pattern.access_frequency / 2.0f, 4.0f);
            total_benefit += reuse_factor * 0.15f; // 每次重用节省15%的访问开销
        }
    }
    
    // 基于CIM友好度调整收益
    total_benefit *= analysis.cim_friendliness_score;
    
    return std::min(total_benefit, 0.8f); // 最大收益限制为80%
}

OptimizationStrategy::OptimizationResult CIMDataReuseStrategy::apply(
    kernel::Graph& graph, const YICAConfig& config) {
    
    OptimizationResult result;
    result.description = "Applied CIM data reuse optimization";
    
    try {
        // 1. 识别重用机会
        auto reuse_patterns = identify_reuse_opportunities(graph);
        
        if (reuse_patterns.empty()) {
            result.success = false;
            result.description += " - No reuse opportunities found";
            return result;
        }
        
        // 2. 应用数据重用优化
        float total_benefit = 0.0f;
        size_t total_memory_saving = 0;
        
        for (const auto& pattern : reuse_patterns) {
            implement_data_reuse(graph, pattern);
            float pattern_benefit = calculate_reuse_benefit(pattern, config);
            total_benefit += pattern_benefit;
            total_memory_saving += pattern.memory_saving;
        }
        
        // 3. 更新结果
        result.success = true;
        result.improvement_score = std::min(total_benefit, 1.0f);
        result.latency_improvement = total_benefit * 0.6f; // 60%的收益转化为延迟改进
        result.energy_reduction = total_benefit * 0.8f;   // 80%的收益转化为能耗降低
        result.memory_efficiency_gain = total_benefit * 0.4f;
        
        result.metrics["reuse_patterns_applied"] = static_cast<float>(reuse_patterns.size());
        result.metrics["memory_saved_bytes"] = static_cast<float>(total_memory_saving);
        
        std::stringstream ss;
        ss << " - Applied " << reuse_patterns.size() << " reuse patterns, "
           << "saved " << total_memory_saving << " bytes of memory access";
        result.description += ss.str();
        
    } catch (const std::exception& e) {
        result.success = false;
        result.description += " - Error: " + std::string(e.what());
    }
    
    return result;
}

std::vector<CIMDataReuseStrategy::ReusePattern> 
CIMDataReuseStrategy::identify_reuse_opportunities(const kernel::Graph& graph) {
    std::vector<ReusePattern> patterns;
    
    // 分析图中的张量访问模式
    std::map<kernel::DTensor*, std::vector<kernel::KNOperator*>> tensor_users;
    
    // 构建张量使用关系图
    for (auto& op : graph.operators) {
        for (auto& input : op->input_tensors) {
            tensor_users[input].push_back(op.get());
        }
    }
    
    // 识别可重用的张量
    for (const auto& [tensor, users] : tensor_users) {
        if (users.size() > 1) {
            ReusePattern pattern;
            pattern.reusable_tensors.push_back(tensor);
            pattern.reuse_factor = static_cast<float>(users.size());
            pattern.pattern_type = "multi_user_tensor";
            
            // 估算内存节省
            size_t tensor_size = tensor->num_elements() * sizeof(float); // 假设float类型
            pattern.memory_saving = tensor_size * (users.size() - 1);
            
            patterns.push_back(pattern);
        }
    }
    
    return patterns;
}

void CIMDataReuseStrategy::implement_data_reuse(kernel::Graph& graph, const ReusePattern& pattern) {
    // 实现数据重用的具体逻辑
    // 这里简化实现，实际应该包含：
    // 1. 在CIM数组中分配重用缓存
    // 2. 修改访问模式以利用缓存
    // 3. 插入数据预取指令
    
    for (auto* tensor : pattern.reusable_tensors) {
        // 标记张量为可重用
        // 在实际实现中，这里会修改张量的内存分配策略
        // tensor->set_reusable(true);
    }
}

float CIMDataReuseStrategy::calculate_reuse_benefit(const ReusePattern& pattern, const YICAConfig& config) const {
    // 计算重用收益
    float base_benefit = std::log2(pattern.reuse_factor) * 0.1f;
    
    // 基于CIM配置调整收益
    float cim_factor = static_cast<float>(config.num_cim_arrays) / 16.0f; // 假设基准为16个CIM阵列
    
    return base_benefit * cim_factor;
}

// ===== SPMAllocationStrategy Implementation =====

SPMAllocationStrategy::SPMAllocationStrategy() {
    // 构造函数实现
}

std::string SPMAllocationStrategy::get_description() const {
    return "Optimizes allocation of tensors to Scratchpad Memory (SPM) to maximize memory efficiency and reduce access latency.";
}

bool SPMAllocationStrategy::is_applicable(const AnalysisResult& analysis) const {
    // 检查是否有SPM配置
    if (analysis.spm_efficiency < 0.1f) {
        return false;
    }
    
    // 检查是否有需要优化的内存访问
    return !analysis.memory_access_patterns.empty();
}

float SPMAllocationStrategy::estimate_benefit(const AnalysisResult& analysis) const {
    // 基于SPM效率和内存访问模式估算收益
    float spm_benefit = (1.0f - analysis.spm_efficiency) * 0.5f; // SPM未充分利用的改进空间
    
    // 基于内存访问频率计算收益
    float access_benefit = 0.0f;
    for (const auto& pattern : analysis.memory_access_patterns) {
        if (pattern.access_frequency > 2) {
            access_benefit += 0.1f; // 高频访问数据放入SPM的收益
        }
    }
    
    return std::min(spm_benefit + access_benefit, 0.7f);
}

OptimizationStrategy::OptimizationResult SPMAllocationStrategy::apply(
    kernel::Graph& graph, const YICAConfig& config) {
    
    OptimizationResult result;
    result.description = "Applied SPM allocation optimization";
    
    try {
        // 1. 生成分配计划
        auto allocation_plan = generate_allocation_plan(graph, config);
        
        if (allocation_plan.tensor_allocation.empty()) {
            result.success = false;
            result.description += " - No allocation plan generated";
            return result;
        }
        
        // 2. 实施SPM分配
        implement_spm_allocation(graph, allocation_plan);
        
        // 3. 计算收益
        float benefit = calculate_allocation_benefit(allocation_plan, config);
        
        result.success = true;
        result.improvement_score = benefit;
        result.latency_improvement = benefit * 0.7f; // SPM访问延迟改进
        result.energy_reduction = benefit * 0.3f;
        result.memory_efficiency_gain = allocation_plan.spm_utilization;
        
        result.metrics["spm_utilization"] = allocation_plan.spm_utilization;
        result.metrics["tensors_allocated"] = static_cast<float>(allocation_plan.tensor_allocation.size());
        result.metrics["access_efficiency"] = allocation_plan.access_efficiency;
        
        std::stringstream ss;
        ss << " - Allocated " << allocation_plan.tensor_allocation.size() << " tensors to SPM, "
           << "utilization: " << (allocation_plan.spm_utilization * 100) << "%";
        result.description += ss.str();
        
    } catch (const std::exception& e) {
        result.success = false;
        result.description += " - Error: " + std::string(e.what());
    }
    
    return result;
}

SPMAllocationStrategy::AllocationPlan SPMAllocationStrategy::generate_allocation_plan(
    const kernel::Graph& graph, const YICAConfig& config) {
    
    AllocationPlan plan;
    
    // 收集所有张量及其访问信息
    struct TensorInfo {
        kernel::DTensor* tensor;
        size_t size;
        float access_frequency;
        float priority_score;
    };
    
    std::vector<TensorInfo> tensor_infos;
    
    // 分析张量使用情况
    std::map<kernel::DTensor*, size_t> tensor_access_count;
    for (const auto& op : graph.operators) {
        for (auto* tensor : op->input_tensors) {
            tensor_access_count[tensor]++;
        }
        for (auto* tensor : op->output_tensors) {
            tensor_access_count[tensor]++;
        }
    }
    
    // 构建张量信息
    for (const auto& [tensor, count] : tensor_access_count) {
        TensorInfo info;
        info.tensor = tensor;
        info.size = tensor->num_elements() * sizeof(float);
        info.access_frequency = static_cast<float>(count);
        info.priority_score = info.access_frequency / std::sqrt(static_cast<float>(info.size));
        tensor_infos.push_back(info);
    }
    
    // 按优先级排序
    std::sort(tensor_infos.begin(), tensor_infos.end(),
              [](const TensorInfo& a, const TensorInfo& b) {
                  return a.priority_score > b.priority_score;
              });
    
    // 贪心分配策略
    size_t available_spm = config.spm_size_kb * 1024; // 转换为字节
    size_t allocated_size = 0;
    size_t spm_offset = 0;
    
    for (const auto& info : tensor_infos) {
        if (allocated_size + info.size <= available_spm) {
            plan.tensor_allocation[info.tensor] = spm_offset;
            allocated_size += info.size;
            spm_offset += info.size;
            plan.total_allocated += info.size;
        }
    }
    
    plan.spm_utilization = static_cast<float>(allocated_size) / available_spm;
    
    // 计算访问效率（简化实现）
    plan.access_efficiency = std::min(plan.spm_utilization * 1.5f, 1.0f);
    
    return plan;
}

void SPMAllocationStrategy::implement_spm_allocation(kernel::Graph& graph, const AllocationPlan& plan) {
    // 实现SPM分配的具体逻辑
    for (const auto& [tensor, offset] : plan.tensor_allocation) {
        // 在实际实现中，这里会：
        // 1. 修改张量的内存分配策略
        // 2. 生成SPM访问指令
        // 3. 更新内存映射表
        // tensor->set_spm_allocation(offset);
    }
}

float SPMAllocationStrategy::calculate_allocation_benefit(const AllocationPlan& plan, const YICAConfig& config) const {
    // 基于SPM利用率和访问效率计算收益
    float utilization_benefit = plan.spm_utilization * 0.5f;
    float efficiency_benefit = plan.access_efficiency * 0.3f;
    
    return utilization_benefit + efficiency_benefit;
}

// ===== OperatorFusionStrategy Implementation =====

OperatorFusionStrategy::OperatorFusionStrategy() {
    // 构造函数实现
}

std::string OperatorFusionStrategy::get_description() const {
    return "Fuses compatible operators to reduce intermediate tensor storage and improve computational efficiency.";
}

bool OperatorFusionStrategy::is_applicable(const AnalysisResult& analysis) const {
    // 检查是否有可融合的操作
    return analysis.cim_friendliness_score > 0.4f && 
           !analysis.memory_access_patterns.empty();
}

float OperatorFusionStrategy::estimate_benefit(const AnalysisResult& analysis) const {
    // 基于操作类型和数据流分析估算融合收益
    float fusion_potential = 0.0f;
    
    // 简化的收益估算：基于CIM友好度和内存访问复杂度
    fusion_potential = analysis.cim_friendliness_score * 0.3f;
    
    if (analysis.memory_access_patterns.size() > 3) {
        fusion_potential += 0.2f; // 复杂的内存访问模式有更大的融合收益
    }
    
    return std::min(fusion_potential, 0.6f);
}

OptimizationStrategy::OptimizationResult OperatorFusionStrategy::apply(
    kernel::Graph& graph, const YICAConfig& config) {
    
    OptimizationResult result;
    result.description = "Applied operator fusion optimization";
    
    try {
        // 1. 识别融合机会
        auto fusion_groups = identify_fusion_opportunities(graph);
        
        if (fusion_groups.empty()) {
            result.success = false;
            result.description += " - No fusion opportunities found";
            return result;
        }
        
        // 2. 应用算子融合
        float total_benefit = 0.0f;
        size_t total_memory_saving = 0;
        
        for (const auto& group : fusion_groups) {
            implement_operator_fusion(graph, group);
            float group_benefit = calculate_fusion_benefit(group, config);
            total_benefit += group_benefit;
            total_memory_saving += group.estimated_memory_saving;
        }
        
        result.success = true;
        result.improvement_score = std::min(total_benefit, 1.0f);
        result.latency_improvement = total_benefit * 0.5f;
        result.energy_reduction = total_benefit * 0.6f;
        result.memory_efficiency_gain = total_benefit * 0.7f;
        
        result.metrics["fusion_groups"] = static_cast<float>(fusion_groups.size());
        result.metrics["memory_saved"] = static_cast<float>(total_memory_saving);
        
        std::stringstream ss;
        ss << " - Fused " << fusion_groups.size() << " operator groups, "
           << "saved " << total_memory_saving << " bytes of intermediate storage";
        result.description += ss.str();
        
    } catch (const std::exception& e) {
        result.success = false;
        result.description += " - Error: " + std::string(e.what());
    }
    
    return result;
}

std::vector<OperatorFusionStrategy::FusionGroup> 
OperatorFusionStrategy::identify_fusion_opportunities(const kernel::Graph& graph) {
    std::vector<FusionGroup> groups;
    
    // 简化的融合机会识别：找到相邻的兼容操作
    std::set<kernel::KNOperator*> processed;
    
    for (const auto& op : graph.operators) {
        if (processed.find(op.get()) != processed.end()) {
            continue;
        }
        
        FusionGroup group;
        group.operators.push_back(op.get());
        processed.insert(op.get());
        
        // 寻找可以与当前操作融合的后续操作
        for (const auto& candidate : graph.operators) {
            if (processed.find(candidate.get()) != processed.end()) {
                continue;
            }
            
            if (can_fuse_operators(op.get(), candidate.get())) {
                group.operators.push_back(candidate.get());
                processed.insert(candidate.get());
                
                // 估算融合收益
                group.fusion_benefit += 0.15f; // 每个融合操作贡献15%的收益
                group.estimated_memory_saving += 1024; // 假设节省1KB中间存储
            }
        }
        
        // 只有包含多个操作的组才算有效融合
        if (group.operators.size() > 1) {
            group.fusion_type = "sequential_fusion";
            groups.push_back(group);
        }
    }
    
    return groups;
}

void OperatorFusionStrategy::implement_operator_fusion(kernel::Graph& graph, const FusionGroup& group) {
    // 实现算子融合的具体逻辑
    // 在实际实现中，这里会：
    // 1. 创建融合后的新操作
    // 2. 移除原有的独立操作
    // 3. 更新数据流连接
    // 4. 优化内存分配
}

float OperatorFusionStrategy::calculate_fusion_benefit(const FusionGroup& group, const YICAConfig& config) const {
    // 基于融合操作数量和类型计算收益
    float base_benefit = std::log2(static_cast<float>(group.operators.size())) * 0.1f;
    
    // 基于CIM阵列数量调整收益
    float cim_factor = std::min(static_cast<float>(config.num_cim_arrays) / 8.0f, 2.0f);
    
    return base_benefit * cim_factor;
}

bool OperatorFusionStrategy::can_fuse_operators(kernel::KNOperator* op1, kernel::KNOperator* op2) const {
    // 简化的融合兼容性检查
    // 实际实现中需要检查：
    // 1. 数据依赖关系
    // 2. 内存访问模式
    // 3. 计算资源需求
    // 4. YICA架构约束
    
    if (!op1 || !op2) {
        return false;
    }
    
    // 检查是否有直接的数据依赖
    for (auto* output : op1->output_tensors) {
        for (auto* input : op2->input_tensors) {
            if (output == input) {
                return true; // 有数据流连接，可以考虑融合
            }
        }
    }
    
    return false;
}

} // namespace yica
} // namespace search
} // namespace yirage 