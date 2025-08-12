// Copyright 2024 CMU
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include "yirage/yica/config.h"
#include "yirage/kernel/operator.h"

namespace yirage {
namespace yica {

// Kernel Fusion Engine for optimizing kernel combinations
class KernelFusionEngine {
public:
    explicit KernelFusionEngine(const YICAConfig& config);
    ~KernelFusionEngine();
    
    // Analyze fusion opportunities
    std::vector<std::pair<int, int>> find_fusion_opportunities(
        const std::vector<kernel::KNOperator*>& operators);
    
    // Generate fused kernel code
    std::string generate_fused_kernel(
        const std::vector<kernel::KNOperator*>& operators);
    
    // Check if operators can be fused
    bool can_fuse(const kernel::KNOperator* op1, const kernel::KNOperator* op2);
    
    // Get fusion benefit score
    float get_fusion_benefit(const std::vector<kernel::KNOperator*>& operators);
    
private:
    YICAConfig config_;
};

// Extended CIM Array Code Generator
class CIMArrayCodeGeneratorExt {
public:
    explicit CIMArrayCodeGeneratorExt(const YICAConfig& config);
    
    // Extended generation methods
    std::string generate_cim_matmul(
        const std::vector<int>& a_shape,
        const std::vector<int>& b_shape,
        const YICAConfig& hardware_config
    );
    
    std::string generate_cim_conv2d(
        const std::vector<int>& input_shape,
        const std::vector<int>& weight_shape,
        const YICAConfig& hardware_config
    );
    
    std::string generate_cim_attention_qkv(
        const std::vector<int>& input_shape,
        int num_heads, int head_dim,
        const YICAConfig& hardware_config
    );
    
    // CIM array configuration optimization
    YICAKernelConfig::CIMConfig optimize_cim_config(
        const std::vector<int>& computation_shape,
        const YICAConfig& hardware_config
    );
    
private:
    YICAConfig config_;
    
    // Internal methods
    std::string generate_cim_array_allocation(int num_arrays);
    std::string generate_cim_data_loading(const std::vector<int>& shape);
    std::string generate_cim_computation_loop(const std::vector<int>& shape);
    std::string generate_cim_result_collection();
    
    // Optimization methods
    int calculate_optimal_array_count(const std::vector<int>& shape);
    std::vector<int> calculate_optimal_tiling(const std::vector<int>& shape, int num_arrays);
};

// Extended SPM Optimizer
class SPMOptimizerExt {
public:
    explicit SPMOptimizerExt(const YICAConfig& config);
    
    // SPM allocation optimization
    std::unordered_map<std::string, size_t> optimize_spm_allocation(
        const std::vector<std::pair<std::string, size_t>>& tensors
    );
    
    // Generate SPM management code
    std::string generate_spm_management_code(
        const std::unordered_map<std::string, size_t>& allocation
    );
    
    // Data prefetch optimization
    std::string generate_prefetch_code(
        const std::vector<std::string>& tensors,
        const std::string& access_pattern
    );
    
    // Double buffering optimization
    std::string generate_double_buffering_code(
        const std::string& producer,
        const std::string& consumer
    );
    
    // Get SPM usage statistics
    float get_utilization() const;
    size_t get_fragmentation() const;
    
private:
    YICAConfig config_;
    size_t allocated_size_;
    std::unordered_map<std::string, size_t> current_allocation_;
    
    // Internal optimization algorithms
    void apply_first_fit(std::vector<std::pair<std::string, size_t>>& tensors);
    void apply_best_fit(std::vector<std::pair<std::string, size_t>>& tensors);
    void apply_graph_coloring(std::vector<std::pair<std::string, size_t>>& tensors);
};

}  // namespace yica
}  // namespace yirage
