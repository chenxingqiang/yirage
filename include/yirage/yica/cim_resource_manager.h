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

namespace yirage {
namespace yica {

// CIM Resource allocation result
struct CIMAllocation {
    uint32_t num_allocated_arrays;
    std::vector<uint32_t> array_ids;
    float efficiency_gain;
    size_t memory_footprint;
    
    CIMAllocation() : num_allocated_arrays(0), efficiency_gain(1.0f), memory_footprint(0) {}
};

// CIM Resource Manager for managing CIM arrays
class CIMResourceManager {
public:
    explicit CIMResourceManager(const YICAConfig& config);
    ~CIMResourceManager();
    
    // Allocate CIM arrays for operations
    CIMAllocation allocate_arrays(size_t num_operations, size_t memory_requirement);
    
    // Release allocated CIM arrays
    void release_arrays(const std::vector<uint32_t>& array_ids);
    
    // Get available CIM arrays
    uint32_t get_available_arrays() const;
    
    // Get total CIM arrays
    uint32_t get_total_arrays() const { return config_.num_cim_arrays; }
    
    // Reset all allocations
    void reset();
    
private:
    YICAConfig config_;
    std::vector<bool> array_status_;  // true if allocated
    std::unordered_map<uint32_t, size_t> array_memory_usage_;
};

// CIM Array Code Generator
class CIMArrayCodeGenerator {
public:
    explicit CIMArrayCodeGenerator(const YICAConfig& config);
    ~CIMArrayCodeGenerator();
    
    // Generate CIM-specific code
    std::string generate_cim_code(const std::string& operation, 
                                  uint32_t array_id,
                                  const std::vector<std::string>& operands);
    
    // Generate CIM initialization code
    std::string generate_init_code(uint32_t array_id);
    
    // Generate CIM cleanup code
    std::string generate_cleanup_code(uint32_t array_id);
    
    // Extended methods for kernel generator
    std::string generate_cim_attention_qkv(
        const std::vector<int>& input_shape,
        int num_heads, int head_dim,
        const YICAConfig& hardware_config
    );
    
    int calculate_optimal_array_count(const std::vector<int>& shape);
    std::vector<int> calculate_optimal_tiling(const std::vector<int>& shape, int num_arrays);
    std::string generate_cim_array_allocation(int num_arrays);
    std::string generate_cim_result_collection();
    
private:
    YICAConfig config_;
};

}  // namespace yica
}  // namespace yirage
