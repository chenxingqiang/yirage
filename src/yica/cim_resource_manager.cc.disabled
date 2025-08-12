#include "yirage/yica/cim_resource_manager.h"
#include <algorithm>

namespace yirage {
namespace yica {

CIMResourceManager::CIMResourceManager(const YICAConfig& config) 
    : config_(config) {
    array_status_.resize(config_.num_cim_arrays, false);
}

CIMResourceManager::~CIMResourceManager() = default;

CIMAllocation CIMResourceManager::allocate_arrays(size_t num_operations, size_t memory_requirement) {
    CIMAllocation allocation;
    
    // 简化的分配策略：分配连续的数组
    uint32_t needed_arrays = std::min(static_cast<uint32_t>(num_operations), config_.num_cim_arrays);
    
    uint32_t allocated = 0;
    for (uint32_t i = 0; i < config_.num_cim_arrays && allocated < needed_arrays; ++i) {
        if (!array_status_[i]) {
            array_status_[i] = true;
            allocation.array_ids.push_back(i);
            array_memory_usage_[i] = memory_requirement / needed_arrays;
            allocated++;
        }
    }
    
    allocation.num_allocated_arrays = allocated;
    allocation.efficiency_gain = static_cast<float>(allocated) / static_cast<float>(needed_arrays);
    allocation.memory_footprint = memory_requirement;
    
    return allocation;
}

void CIMResourceManager::release_arrays(const std::vector<uint32_t>& array_ids) {
    for (uint32_t id : array_ids) {
        if (id < array_status_.size()) {
            array_status_[id] = false;
            array_memory_usage_.erase(id);
        }
    }
}

uint32_t CIMResourceManager::get_available_arrays() const {
    return std::count(array_status_.begin(), array_status_.end(), false);
}

void CIMResourceManager::reset() {
    std::fill(array_status_.begin(), array_status_.end(), false);
    array_memory_usage_.clear();
}

// CIMArrayCodeGenerator implementation
CIMArrayCodeGenerator::CIMArrayCodeGenerator(const YICAConfig& config) : config_(config) {}

CIMArrayCodeGenerator::~CIMArrayCodeGenerator() = default;

std::string CIMArrayCodeGenerator::generate_cim_code(const std::string& operation, 
                                                    uint32_t array_id,
                                                    const std::vector<std::string>& operands) {
    std::stringstream code;
    
    code << "// CIM Array Operation: " << operation << "\n";
    code << "cim_array_select " << array_id << "\n";
    
    if (operation == "matmul") {
        code << "cim_matmul_init\n";
        if (operands.size() >= 2) {
            code << "cim_load_operand A from " << operands[0] << "\n";
            code << "cim_load_operand B from " << operands[1] << "\n";
        }
        code << "cim_matmul_execute\n";
        code << "cim_store_result to output\n";
    } else if (operation == "attention") {
        code << "cim_attention_init\n";
        if (operands.size() >= 3) {
            code << "cim_load_qkv " << operands[0] << " " << operands[1] << " " << operands[2] << "\n";
        }
        code << "cim_attention_execute\n";
        code << "cim_store_result to output\n";
    } else {
        code << "cim_generic_operation " << operation << "\n";
        for (const auto& operand : operands) {
            code << "cim_load_operand from " << operand << "\n";
        }
        code << "cim_execute\n";
        code << "cim_store_result to output\n";
    }
    
    return code.str();
}

std::string CIMArrayCodeGenerator::generate_init_code(uint32_t array_id) {
    std::stringstream code;
    code << "cim_array_init " << array_id << "\n";
    code << "cim_array_reset " << array_id << "\n";
    return code.str();
}

std::string CIMArrayCodeGenerator::generate_cleanup_code(uint32_t array_id) {
    std::stringstream code;
    code << "cim_array_sync " << array_id << "\n";
    code << "cim_array_cleanup " << array_id << "\n";
    return code.str();
}

}  // namespace yica
}  // namespace yirage
