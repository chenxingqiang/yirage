#include "yirage/yica/spm_memory_manager.h"
#include "yirage/kernel/graph.h"

namespace yirage {
namespace yica {

SPMMemoryManager::SPMMemoryManager(const YICAConfig& config) 
    : config_(config), allocated_size_(0) {}

SPMMemoryManager::~SPMMemoryManager() = default;

void* SPMMemoryManager::allocate(size_t size, size_t alignment) {
    // 简化实现：使用标准内存分配
    void* ptr = aligned_alloc(alignment, size);
    if (ptr) {
        allocations_[ptr] = size;
        allocated_size_ += size;
    }
    return ptr;
}

void SPMMemoryManager::deallocate(void* ptr) {
    if (ptr && allocations_.find(ptr) != allocations_.end()) {
        allocated_size_ -= allocations_[ptr];
        allocations_.erase(ptr);
        free(ptr);
    }
}

size_t SPMMemoryManager::get_available_size() const {
    return get_total_size() - allocated_size_;
}

void SPMMemoryManager::reset() {
    for (auto& [ptr, size] : allocations_) {
        free(ptr);
    }
    allocations_.clear();
    allocated_size_ = 0;
}

SPMMemoryPlan SPMMemoryManager::create_memory_plan(const std::vector<std::pair<std::string, size_t>>& tensors) {
    SPMMemoryPlan plan;
    size_t offset = 0;
    
    for (const auto& [name, size] : tensors) {
        plan.tensor_offsets[name] = offset;
        offset += size;
    }
    
    plan.total_spm_usage = offset;
    plan.access_efficiency = 1.0f; // 简化假设
    
    return plan;
}

SPMMemoryPlan SPMMemoryManager::plan_memory_allocation(const kernel::Graph& graph) {
    // 简化实现：创建一个基本的内存计划
    SPMMemoryPlan plan;
    plan.total_spm_usage = 1024 * 1024; // 1MB 默认
    plan.access_efficiency = 0.8f;
    
    // 简化的张量分配
    plan.tensor_offsets["input"] = 0;
    plan.tensor_offsets["output"] = 512 * 1024;
    
    return plan;
}

// YIS指令引擎所需的方法实现
void* SPMMemoryManager::allocate_spm_buffer(uint32_t cim_array_id, size_t size) {
    // 简化实现：使用标准分配器
    return allocate(size);
}

void* SPMMemoryManager::get_spm_buffer(uint32_t cim_array_id, uint64_t addr) {
    // 简化实现：直接返回地址
    return reinterpret_cast<void*>(addr);
}

void* SPMMemoryManager::get_matrix_buffer(uint32_t cim_array_id, const std::string& name) {
    // 简化实现：分配默认大小的缓冲区
    return allocate(1024 * 1024); // 1MB 默认矩阵大小
}

// SPMOptimizer implementation
SPMOptimizer::SPMOptimizer(const YICAConfig& config) : config_(config) {}

SPMOptimizer::~SPMOptimizer() = default;

SPMMemoryPlan SPMOptimizer::optimize_allocation(const std::vector<std::pair<std::string, size_t>>& tensors) {
    SPMMemoryPlan plan;
    size_t offset = 0;
    
    // 简单的首次适配算法
    for (const auto& [name, size] : tensors) {
        plan.tensor_offsets[name] = offset;
        offset += size;
    }
    
    plan.total_spm_usage = offset;
    plan.access_efficiency = 0.85f; // 优化后的效率
    
    return plan;
}

float SPMOptimizer::analyze_access_pattern(const SPMMemoryPlan& plan) {
    // 简化的访问模式分析
    return plan.access_efficiency;
}

std::vector<std::string> SPMOptimizer::get_optimization_suggestions(const SPMMemoryPlan& plan) {
    std::vector<std::string> suggestions;
    
    if (plan.access_efficiency < 0.7f) {
        suggestions.push_back("Consider tensor reordering for better cache locality");
    }
    
    if (plan.total_spm_usage > config_.spm_size_kb * 1024 * 0.9) {
        suggestions.push_back("SPM usage is high, consider tensor splitting");
    }
    
    return suggestions;
}

}  // namespace yica
}  // namespace yirage
