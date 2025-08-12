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
#include <unordered_map>
#include <string>
#include "yirage/yica/config.h"

// 前向声明
namespace yirage {
namespace kernel {
class Graph;
}
}

namespace yirage {
namespace yica {

// SPM Memory allocation plan
struct SPMMemoryPlan {
    size_t total_spm_usage;
    float access_efficiency;
    std::unordered_map<std::string, size_t> tensor_offsets;
    
    SPMMemoryPlan() : total_spm_usage(0), access_efficiency(1.0f) {}
};

// SPM Memory Manager
class SPMMemoryManager {
public:
    explicit SPMMemoryManager(const YICAConfig& config);
    ~SPMMemoryManager();
    
    // Allocate SPM memory
    void* allocate(size_t size, size_t alignment = 64);
    
    // Deallocate SPM memory
    void deallocate(void* ptr);
    
    // Get available SPM size
    size_t get_available_size() const;
    
    // Get total SPM size
    size_t get_total_size() const { return config_.spm_size_kb * 1024; }
    
    // Reset all allocations
    void reset();
    
    // Create memory plan
    SPMMemoryPlan create_memory_plan(const std::vector<std::pair<std::string, size_t>>& tensors);
    
    // From graph memory planning
    SPMMemoryPlan plan_memory_allocation(const kernel::Graph& graph);
    
    // YIS指令引擎所需的方法
    void* allocate_spm_buffer(uint32_t cim_array_id, size_t size);
    void* get_spm_buffer(uint32_t cim_array_id, uint64_t addr);
    void* get_matrix_buffer(uint32_t cim_array_id, const std::string& name);
    
private:
    YICAConfig config_;
    size_t allocated_size_;
    std::unordered_map<void*, size_t> allocations_;
};

// SPM Optimizer
class SPMOptimizer {
public:
    explicit SPMOptimizer(const YICAConfig& config);
    ~SPMOptimizer();
    
    // Optimize SPM allocation
    SPMMemoryPlan optimize_allocation(const std::vector<std::pair<std::string, size_t>>& tensors);
    
    // Analyze memory access pattern
    float analyze_access_pattern(const SPMMemoryPlan& plan);
    
    // Get optimization suggestions
    std::vector<std::string> get_optimization_suggestions(const SPMMemoryPlan& plan);
    
    // Code generation methods
    std::string generate_spm_allocation_code(const SPMMemoryPlan& plan);
    std::string generate_prefetch_code(const std::vector<std::string>& tensors);
    
private:
    YICAConfig config_;
};

}  // namespace yica
}  // namespace yirage
