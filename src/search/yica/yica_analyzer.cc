/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yirage/search/yica/yica_analyzer.h"
#include <algorithm>
#include <cmath>

namespace yirage {
namespace search {
namespace yica {

YICAArchitectureAnalyzer::YICAArchitectureAnalyzer(const YICAConfig& config)
    : config_(config) {
}

YICAArchitectureAnalyzer::~YICAArchitectureAnalyzer() {
}

AnalysisResult YICAArchitectureAnalyzer::analyze_computation_pattern(const kernel::Graph& graph) {
    AnalysisResult result;
    
    // 分析CIM友好度
    float total_cim_score = 0.0f;
    size_t total_ops = 0;
    
    for (const auto& op : graph.operators) {
        float cim_score = calculate_cim_friendliness(op);
        total_cim_score += cim_score;
        total_ops++;
        
        if (cim_score > 0.6f) {
            for (const auto& output_tensor : op->output_tensors) {
                result.cim_friendly_ops.push_back(const_cast<kernel::DTensor*>(&output_tensor));
            }
        }
    }
    
    result.cim_friendliness_score = (total_ops > 0) ? total_cim_score / total_ops : 0.0f;
    result.memory_locality_score = analyze_memory_access_pattern(graph);
    result.parallel_opportunities = find_parallel_patterns(graph);
    
    // 计算并行化潜力
    float parallel_score = 0.0f;
    for (const auto& opportunity : result.parallel_opportunities) {
        parallel_score += opportunity.efficiency_score;
    }
    result.parallelization_potential = std::min(1.0f, parallel_score);
    
    // 生成优化建议和性能估算
    result.estimated_speedup = 1.0f + result.cim_friendliness_score * 2.0f;
    result.estimated_energy_reduction = result.cim_friendliness_score * 0.6f;
    
    return result;
}

std::vector<kernel::DTensor*> YICAArchitectureAnalyzer::identify_cim_operations(const kernel::Graph& graph) {
    std::vector<kernel::DTensor*> cim_ops;
    
    for (const auto& op : graph.operators) {
        float cim_score = calculate_cim_friendliness(op);
        if (cim_score > 0.6f) {
            for (const auto& output_tensor : op->output_tensors) {
                cim_ops.push_back(const_cast<kernel::DTensor*>(&output_tensor));
            }
        }
    }
    
    return cim_ops;
}

float YICAArchitectureAnalyzer::analyze_memory_access_pattern(const kernel::Graph& graph) {
    size_t total_memory_accesses = 0;
    size_t spm_friendly_accesses = 0;
    
    for (const auto& op : graph.operators) {
        for (const auto& tensor : op->input_tensors) {
            size_t tensor_size = calculate_tensor_size(&tensor);
            total_memory_accesses += tensor_size;
            
            if (tensor_size <= config_.spm_size_per_die) {
                spm_friendly_accesses += tensor_size;
            }
        }
    }
    
    return (total_memory_accesses > 0) ? 
           static_cast<float>(spm_friendly_accesses) / total_memory_accesses : 0.0f;
}

float YICAArchitectureAnalyzer::calculate_cim_friendliness(const kernel::KNOperator* op) {
    float base_score = 0.3f;
    
    // 根据操作类型确定基础分数
    OpType op_type = identify_operation_type(op);
    switch (op_type) {
        case OpType::MATMUL:
            base_score = 0.9f;
            break;
        case OpType::ELEMENTWISE:
            base_score = 0.7f;
            break;
        case OpType::REDUCTION:
            base_score = 0.6f;
            break;
        default:
            base_score = 0.3f;
            break;
    }
    
    return std::min(1.0f, base_score);
}

OpType YICAArchitectureAnalyzer::identify_operation_type(const kernel::KNOperator* op) {
    switch (op->op_type) {
        case yirage::type::KN_MATMUL_OP:
            return OpType::MATMUL;
        case yirage::type::KN_ADD_OP:
        case yirage::type::KN_MUL_OP:
        case yirage::type::KN_EXP_OP:
            return OpType::ELEMENTWISE;
        case yirage::type::KN_REDUCTION_0_OP:
        case yirage::type::KN_REDUCTION_1_OP:
            return OpType::REDUCTION;
        default:
            return OpType::OTHER;
    }
}

size_t YICAArchitectureAnalyzer::calculate_tensor_size(const kernel::DTensor* tensor) {
    return tensor->data_size();
}

std::vector<ParallelizationOpportunity> YICAArchitectureAnalyzer::find_parallel_patterns(const kernel::Graph& graph) {
    return find_data_parallel_opportunities(graph);
}

std::vector<ParallelizationOpportunity> YICAArchitectureAnalyzer::find_data_parallel_opportunities(const kernel::Graph& graph) {
    std::vector<ParallelizationOpportunity> opportunities;
    
    for (const auto& op : graph.operators) {
        if (op->op_type == yirage::type::KN_MATMUL_OP) {
            ParallelizationOpportunity opportunity;
            opportunity.type = ParallelizationOpportunity::DATA_PARALLEL;
            opportunity.efficiency_score = 0.8f;
            opportunity.recommended_parallelism = config_.num_cim_dies;
            opportunity.description = "Data parallel execution across CIM dies";
            
            for (const auto& tensor : op->output_tensors) {
                opportunity.involved_tensors.push_back(const_cast<kernel::DTensor*>(&tensor));
            }
            
            opportunities.push_back(opportunity);
        }
    }
    
    return opportunities;
}

DataType YICAArchitectureAnalyzer::get_tensor_data_type(const kernel::DTensor* tensor) {
    switch (tensor->data_type) {
        case yirage::type::DT_FLOAT32:
            return DataType::FP32;
        case yirage::type::DT_FLOAT16:
            return DataType::FP16;
        case yirage::type::DT_BFLOAT16:
            return DataType::BF16;
        default:
            return DataType::UNKNOWN;
    }
}

void YICAArchitectureAnalyzer::update_config(const YICAConfig& config) {
    config_ = config;
}

const YICAConfig& YICAArchitectureAnalyzer::get_config() const {
    return config_;
}

// 其他方法的简化实现
float YICAArchitectureAnalyzer::estimate_memory_cost(const kernel::DTensor* tensor) {
    return static_cast<float>(calculate_tensor_size(tensor));
}

MemoryAccessPattern YICAArchitectureAnalyzer::analyze_tensor_access_pattern(const kernel::DTensor* tensor) {
    MemoryAccessPattern pattern;
    pattern.working_set_size = calculate_tensor_size(tensor);
    pattern.spm_friendly = (pattern.working_set_size <= config_.spm_size_per_die);
    pattern.spatial_locality = 0.7f;
    pattern.temporal_locality = 0.6f;
    return pattern;
}

size_t YICAArchitectureAnalyzer::estimate_computation_complexity(const kernel::KNOperator* op) {
    if (op->input_tensors.empty()) return 0;
    return calculate_tensor_size(&op->input_tensors[0]);
}

std::vector<ParallelizationOpportunity> YICAArchitectureAnalyzer::find_model_parallel_opportunities(const kernel::Graph& graph) {
    return std::vector<ParallelizationOpportunity>();  // 简化实现
}

} // namespace yica
} // namespace search
} // namespace yirage
 