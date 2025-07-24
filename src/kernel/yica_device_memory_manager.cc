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

#include "yirage/kernel/yica_device_memory_manager.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <algorithm>
#include <cstring>

namespace yirage {
namespace kernel {

YICADeviceMemoryManager* YICADeviceMemoryManager::singleton = nullptr;

YICADeviceMemoryManager::YICADeviceMemoryManager(int _num_devices, 
                                                 int _device_id,
                                                 const YICAMemoryConfig &config)
    : num_devices(_num_devices), device_id(_device_id), yica_config(config) {
  
  // 初始化三级内存层次
  initialize_memory_hierarchy();
  
  // 初始化SPM管理器
  initialize_spm_managers();
  
  // 初始化内存池
  initialize_memory_pools();
  
  // 初始化压缩引擎
  if (yica_config.enable_compression) {
    initialize_compression_engines();
  }
  
  // 初始化诊断工具
  if (yica_config.enable_memory_diagnostics) {
    initialize_diagnostic_tools();
  }
}

YICADeviceMemoryManager::~YICADeviceMemoryManager() {
  // 清理内存池
  cleanup_memory_pools();
  
  // 清理SPM管理器
  cleanup_spm_managers();
  
  // 清理压缩引擎
  if (compression_engines_) {
    cleanup_compression_engines();
  }
}

bool YICADeviceMemoryManager::initialize_memory_hierarchy() {
  // 初始化寄存器文件层
  register_file_manager_ = std::make_unique<RegisterFileManager>(yica_config.register_file_config);
  
  // 初始化SPM层
  for (int i = 0; i < yica_config.num_spm_banks; i++) {
    spm_managers_.push_back(std::make_unique<SPMManager>(i, yica_config.spm_config));
  }
  
  // 初始化DRAM层
  dram_manager_ = std::make_unique<DRAMManager>(yica_config.dram_config);
  
  return true;
}

bool YICADeviceMemoryManager::initialize_spm_managers() {
  for (auto& spm_manager : spm_managers_) {
    if (!spm_manager->initialize()) {
      return false;
    }
  }
  return true;
}

bool YICADeviceMemoryManager::initialize_memory_pools() {
  // 为每个内存层次创建内存池
  register_pool_ = std::make_unique<MemoryPool>(
    MemoryLevel::REGISTER_FILE, 
    yica_config.register_file_config.total_size);
  
  for (int i = 0; i < yica_config.num_spm_banks; i++) {
    spm_pools_.push_back(std::make_unique<MemoryPool>(
      MemoryLevel::SPM, 
      yica_config.spm_config.bank_size));
  }
  
  dram_pool_ = std::make_unique<MemoryPool>(
    MemoryLevel::DRAM,
    yica_config.dram_config.total_size);
  
  return true;
}

bool YICADeviceMemoryManager::initialize_compression_engines() {
  compression_engines_ = std::make_unique<CompressionEngineManager>(yica_config.compression_config);
  return compression_engines_->initialize();
}

bool YICADeviceMemoryManager::initialize_diagnostic_tools() {
  memory_profiler_ = std::make_unique<MemoryProfiler>();
  bandwidth_monitor_ = std::make_unique<BandwidthMonitor>();
  fragmentation_analyzer_ = std::make_unique<FragmentationAnalyzer>();
  
  return true;
}

YICAMemoryAllocation YICADeviceMemoryManager::allocate_yica_optimized(
    size_t size, 
    MemoryLevel preferred_level,
    AllocationStrategy strategy) {
  
  YICAMemoryAllocation allocation;
  allocation.size = size;
  allocation.strategy = strategy;
  allocation.timestamp = std::chrono::steady_clock::now();
  
  // 根据策略选择内存层次
  switch (strategy) {
    case AllocationStrategy::PERFORMANCE_FIRST:
      allocation = allocate_performance_first(size, preferred_level);
      break;
      
    case AllocationStrategy::CAPACITY_FIRST:
      allocation = allocate_capacity_first(size, preferred_level);
      break;
      
    case AllocationStrategy::BALANCED:
      allocation = allocate_balanced(size, preferred_level);
      break;
      
    case AllocationStrategy::ADAPTIVE:
      allocation = allocate_adaptive(size, preferred_level);
      break;
  }
  
  // 记录分配信息
  if (allocation.ptr != nullptr) {
    active_allocations_[allocation.ptr] = allocation;
    update_allocation_statistics(allocation);
  }
  
  return allocation;
}

YICAMemoryAllocation YICADeviceMemoryManager::allocate_performance_first(
    size_t size, 
    MemoryLevel preferred_level) {
  
  YICAMemoryAllocation allocation;
  allocation.size = size;
  
  // 优先从最快的内存层次分配
  if (preferred_level == MemoryLevel::REGISTER_FILE || 
      (preferred_level == MemoryLevel::AUTO && size <= yica_config.register_file_config.max_allocation_size)) {
    
    allocation.ptr = register_pool_->allocate(size);
    if (allocation.ptr) {
      allocation.level = MemoryLevel::REGISTER_FILE;
      allocation.access_latency_ns = yica_config.register_file_config.access_latency_ns;
      return allocation;
    }
  }
  
  // 尝试SPM
  if (preferred_level == MemoryLevel::SPM || preferred_level == MemoryLevel::AUTO) {
    int best_spm_bank = select_best_spm_bank(size);
    if (best_smp_bank >= 0) {
      allocation.ptr = spm_pools_[best_spm_bank]->allocate(size);
      if (allocation.ptr) {
        allocation.level = MemoryLevel::SPM;
        allocation.spm_bank_id = best_smp_bank;
        allocation.access_latency_ns = yica_config.spm_config.access_latency_ns;
        return allocation;
      }
    }
  }
  
  // 最后尝试DRAM
  allocation.ptr = dram_pool_->allocate(size);
  if (allocation.ptr) {
    allocation.level = MemoryLevel::DRAM;
    allocation.access_latency_ns = yica_config.dram_config.access_latency_ns;
  }
  
  return allocation;
}

YICAMemoryAllocation YICADeviceMemoryManager::allocate_adaptive(
    size_t size, 
    MemoryLevel preferred_level) {
  
  // 基于历史访问模式和当前内存状态做自适应分配
  AdaptiveAllocationDecision decision = adaptive_allocator_.make_decision(
    size, preferred_level, get_current_memory_state());
  
  return allocate_with_decision(size, decision);
}

int YICADeviceMemoryManager::select_best_spm_bank(size_t size) {
  int best_bank = -1;
  float best_score = -1.0f;
  
  for (int i = 0; i < spm_pools_.size(); i++) {
    if (spm_pools_[i]->can_allocate(size)) {
      float score = calculate_smp_bank_score(i, size);
      if (score > best_score) {
        best_score = score;
        best_bank = i;
      }
    }
  }
  
  return best_bank;
}

float YICADeviceMemoryManager::calculate_spm_bank_score(int bank_id, size_t size) {
  const auto& pool = spm_pools_[bank_id];
  
  // 基于多个因素计算分数
  float utilization_score = 1.0f - pool->get_utilization_ratio();
  float fragmentation_score = 1.0f - pool->get_fragmentation_ratio();
  float locality_score = calculate_locality_score(bank_id);
  
  return utilization_score * 0.4f + fragmentation_score * 0.3f + locality_score * 0.3f;
}

bool YICADeviceMemoryManager::deallocate_yica_optimized(void* ptr) {
  auto it = active_allocations_.find(ptr);
  if (it == active_allocations_.end()) {
    return false;
  }
  
  const YICAMemoryAllocation& allocation = it->second;
  
  // 根据内存层次进行释放
  bool success = false;
  switch (allocation.level) {
    case MemoryLevel::REGISTER_FILE:
      success = register_pool_->deallocate(ptr);
      break;
      
    case MemoryLevel::SPM:
      success = spm_pools_[allocation.spm_bank_id]->deallocate(ptr);
      break;
      
    case MemoryLevel::DRAM:
      success = dram_pool_->deallocate(ptr);
      break;
  }
  
  if (success) {
    // 更新统计信息
    update_deallocation_statistics(allocation);
    active_allocations_.erase(it);
  }
  
  return success;
}

bool YICADeviceMemoryManager::enable_spm_caching(const SPMCacheConfig& config) {
  spm_cache_config_ = config;
  
  // 为每个SPM银行启用缓存
  for (auto& smp_manager : spm_managers_) {
    if (!spm_manager->enable_caching(config)) {
      return false;
    }
  }
  
  return true;
}

YICASPMCacheStats YICADeviceMemoryManager::get_spm_cache_stats() const {
  YICASPMCacheStats total_stats = {};
  
  for (const auto& spm_manager : spm_managers_) {
    auto bank_stats = spm_manager->get_cache_stats();
    total_stats.total_accesses += bank_stats.total_accesses;
    total_stats.cache_hits += bank_stats.cache_hits;
    total_stats.cache_misses += bank_stats.cache_misses;
    total_stats.evictions += bank_stats.evictions;
    total_stats.write_backs += bank_stats.write_backs;
  }
  
  // 计算聚合指标
  if (total_stats.total_accesses > 0) {
    total_stats.hit_rate = static_cast<float>(total_stats.cache_hits) / total_stats.total_accesses;
    total_stats.miss_rate = static_cast<float>(total_stats.cache_misses) / total_stats.total_accesses;
  }
  
  return total_stats;
}

bool YICADeviceMemoryManager::optimize_memory_layout(const std::vector<DTensor*>& tensors) {
  LayoutOptimizer optimizer(yica_config.layout_config);
  
  // 分析张量访问模式
  auto access_patterns = analyzer_tensor_access_patterns(tensors);
  
  // 生成优化建议
  auto layout_suggestions = optimizer.generate_layout_suggestions(tensors, access_patterns);
  
  // 应用布局优化
  for (const auto& suggestion : layout_suggestions) {
    if (!apply_layout_optimization(suggestion)) {
      return false;
    }
  }
  
  return true;
}

YICAMemoryDiagnostics YICADeviceMemoryManager::get_memory_diagnostics() const {
  YICAMemoryDiagnostics diagnostics;
  
  if (memory_profiler_) {
    diagnostics.profiling_data = memory_profiler_->get_current_profile();
  }
  
  if (bandwidth_monitor_) {
    diagnostics.bandwidth_stats = bandwidth_monitor_->get_current_stats();
  }
  
  if (fragmentation_analyzer_) {
    diagnostics.fragmentation_stats = fragmentation_analyzer_->analyze_current_state();
  }
  
  // 计算总体内存使用情况
  diagnostics.total_allocated = calculate_total_allocated_memory();
  diagnostics.total_available = calculate_total_available_memory();
  diagnostics.utilization_ratio = static_cast<float>(diagnostics.total_allocated) / 
                                 (diagnostics.total_allocated + diagnostics.total_available);
  
  return diagnostics;
}

size_t YICADeviceMemoryManager::calculate_total_allocated_memory() const {
  size_t total = 0;
  
  total += register_pool_->get_allocated_size();
  
  for (const auto& spm_pool : smp_pools_) {
    total += spm_pool->get_allocated_size();
  }
  
  total += dram_pool_->get_allocated_size();
  
  return total;
}

bool YICADeviceMemoryManager::enable_memory_compression(CompressionAlgorithm algorithm) {
  if (!compression_engines_) {
    return false;
  }
  
  return compression_engines_->enable_algorithm(algorithm);
}

YICACompressionStats YICADeviceMemoryManager::get_compression_stats() const {
  if (!compression_engines_) {
    return {};
  }
  
  return compression_engines_->get_statistics();
}

void YICADeviceMemoryManager::update_allocation_statistics(const YICAMemoryAllocation& allocation) {
  allocation_stats_.total_allocations++;
  allocation_stats_.total_allocated_bytes += allocation.size;
  
  switch (allocation.level) {
    case MemoryLevel::REGISTER_FILE:
      allocation_stats_.register_allocations++;
      break;
    case MemoryLevel::SPM:
      allocation_stats_.spm_allocations++;
      break;
    case MemoryLevel::DRAM:
      allocation_stats_.dram_allocations++;
      break;
  }
  
  // 更新性能指标
  if (memory_profiler_) {
    memory_profiler_->record_allocation(allocation);
  }
}

MemoryState YICADeviceMemoryManager::get_current_memory_state() const {
  MemoryState state;
  
  state.register_utilization = register_pool_->get_utilization_ratio();
  
  state.spm_utilizations.resize(smp_pools_.size());
  for (size_t i = 0; i < spm_pools_.size(); i++) {
    state.spm_utilizations[i] = spm_pools_[i]->get_utilization_ratio();
  }
  
  state.dram_utilization = dram_pool_->get_utilization_ratio();
  state.total_fragmentation = calculate_total_fragmentation();
  
  return state;
}

/*static*/
YICADeviceMemoryManager* YICADeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    YICAMemoryConfig default_config;
    singleton = new YICADeviceMemoryManager(1, 0, default_config);
  }
  return singleton;
}

/*static*/
void YICADeviceMemoryManager::set_yica_config(int device_id, const YICAMemoryConfig& config) {
  assert(singleton == nullptr); // 必须在创建实例之前调用
  singleton = new YICADeviceMemoryManager(1, device_id, config);
}

// 工厂函数实现
YICADeviceMemoryManager* create_yica_memory_manager(
    int num_devices,
    int device_id, 
    const YICADeviceMemoryManager::YICAMemoryConfig& config) {
  
  return new YICADeviceMemoryManager(num_devices, device_id, config);
}

// 辅助函数实现
namespace yica_memory_utils {
  
  OptimalMemoryLayout analyze_optimal_layout(
      const std::vector<DTensor*>& tensors,
      const YICADeviceMemoryManager::YICAMemoryConfig& config) {
    
    OptimalMemoryLayout layout;
    
    // 分析张量大小分布
    size_t total_size = 0;
    size_t max_tensor_size = 0;
    
    for (const auto* tensor : tensors) {
      size_t tensor_size = tensor->data_size();
      total_size += tensor_size;
      max_tensor_size = std::max(max_tensor_size, tensor_size);
    }
    
    // 基于分析结果生成建议
    if (max_tensor_size <= config.register_file_config.max_allocation_size) {
      layout.primary_level = YICADeviceMemoryManager::MemoryLevel::REGISTER_FILE;
      layout.efficiency_score = 0.95f;
    } else if (total_size <= config.spm_config.total_size) {
      layout.primary_level = YICADeviceMemoryManager::MemoryLevel::SPM;
      layout.efficiency_score = 0.85f;
    } else {
      layout.primary_level = YICADeviceMemoryManager::MemoryLevel::DRAM;
      layout.efficiency_score = 0.70f;
    }
    
    layout.estimated_bandwidth = calculate_estimated_bandwidth(layout.primary_level, config);
    layout.estimated_latency = calculate_estimated_latency(layout.primary_level, config);
    
    return layout;
  }
  
  MemoryBandwidthAnalysis analyze_memory_bandwidth(
      const std::vector<YICADeviceMemoryManager::YICAMemoryAllocation>& allocations) {
    
    MemoryBandwidthAnalysis analysis;
    
    // 分析不同内存层次的带宽使用
    for (const auto& allocation : allocations) {
      switch (allocation.level) {
        case YICADeviceMemoryManager::MemoryLevel::REGISTER_FILE:
          analysis.register_bandwidth_usage += allocation.size;
          break;
        case YICADeviceMemoryManager::MemoryLevel::SPM:
          analysis.spm_bandwidth_usage += allocation.size;
          break;
        case YICADeviceMemoryManager::MemoryLevel::DRAM:
          analysis.dram_bandwidth_usage += allocation.size;
          break;
      }
    }
    
    analysis.total_bandwidth_usage = analysis.register_bandwidth_usage + 
                                    analysis.spm_bandwidth_usage + 
                                    analysis.dram_bandwidth_usage;
    
    // 计算带宽效率
    if (analysis.total_bandwidth_usage > 0) {
      analysis.bandwidth_efficiency = 
        (analysis.register_bandwidth_usage * 1.0f + 
         analysis.spm_bandwidth_usage * 0.8f + 
         analysis.dram_bandwidth_usage * 0.5f) / analysis.total_bandwidth_usage;
    }
    
    return analysis;
  }
  
} // namespace yica_memory_utils

} // namespace kernel
} // namespace yirage 