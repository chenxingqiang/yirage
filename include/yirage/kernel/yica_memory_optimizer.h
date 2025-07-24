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

#pragma once

#include "yirage/kernel/device_tensor.h"
#include "yirage/yica/yis_instruction_set.h"
#include <vector>
#include <unordered_map>

namespace yirage {
namespace kernel {

/**
 * @brief YICA内存访问模式分析器
 * 分析张量访问模式，生成最优的YISECOPY/YISICOPY指令序列
 */
class YICAMemoryOptimizer {
public:
  YICAMemoryOptimizer();
  ~YICAMemoryOptimizer();

  // 内存访问模式枚举
  enum class AccessPattern {
    SEQUENTIAL,     // 顺序访问 - 适合YISECOPY_G2S
    STRIDED,        // 跨步访问 - 需要优化数据布局
    RANDOM,         // 随机访问 - 需要SPM缓存
    BROADCAST,      // 广播模式 - 适合YISICOPY_BC
    GATHER_SCATTER, // 收集散射 - 适合YISICOPY_GAT
    CONVOLUTION     // 卷积模式 - 适合IM2COL变换
  };

  // 内存优化策略
  struct MemoryOptimizationPlan {
    std::vector<yica::YISInstruction> copy_instructions;
    std::vector<std::pair<DTensor*, size_t>> spm_allocation;
    AccessPattern dominant_pattern;
    float predicted_bandwidth_utilization;
    size_t total_memory_footprint;
  };

  /**
   * @brief 分析张量的内存访问模式
   */
  AccessPattern analyze_access_pattern(const DTensor& tensor,
                                     const std::vector<int>& access_indices);

  /**
   * @brief 生成内存优化计划
   */
  MemoryOptimizationPlan optimize_memory_access(
    const std::vector<DTensor>& tensors,
    const std::vector<AccessPattern>& patterns);

  /**
   * @brief 生成YISECOPY指令 (外部拷贝)
   */
  std::vector<yica::YISInstruction> generate_external_copy_instructions(
    const DTensor& src, const DTensor& dst,
    yica::YISECopyType copy_type = yica::YISECopyType::G2S);

  /**
   * @brief 生成YISICOPY指令 (内部拷贝)
   */
  std::vector<yica::YISInstruction> generate_internal_copy_instructions(
    const std::vector<DTensor>& tensors,
    yica::YISICopyMode mode = yica::YISICopyMode::MULTICAST);

  /**
   * @brief 优化数据布局以匹配YICA架构
   */
  yica::YICADataLayout recommend_optimal_layout(
    const DTensor& tensor, AccessPattern pattern);

  /**
   * @brief SPM内存分配优化
   */
  struct SPMAllocationResult {
    std::unordered_map<DTensor*, size_t> tensor_to_spm_offset;
    size_t total_spm_usage;
    float spm_utilization_rate;
    bool allocation_successful;
  };

  SPMAllocationResult allocate_spm_memory(
    const std::vector<DTensor>& tensors,
    size_t available_spm_size);

  /**
   * @brief 生成IM2COL优化的卷积数据变换
   */
  std::vector<yica::YISInstruction> generate_im2col_instructions(
    const DTensor& input_tensor,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

  /**
   * @brief 预测内存访问性能
   */
  struct MemoryPerformancePrediction {
    float dram_to_spm_bandwidth_mbps;
    float spm_hit_rate;
    float average_access_latency_cycles;
    size_t total_memory_transactions;
  };

  MemoryPerformancePrediction predict_memory_performance(
    const MemoryOptimizationPlan& plan);

private:
  // 内存访问模式分析
  AccessPattern classify_access_pattern(const std::vector<int>& indices);
  
  // 数据布局转换
  bool can_convert_layout(yica::YICADataLayout from, yica::YICADataLayout to);
  std::vector<yica::YISInstruction> generate_layout_conversion(
    const DTensor& tensor, yica::YICADataLayout target_layout);
  
  // SPM分配策略
  std::vector<DTensor*> prioritize_tensors_for_spm(
    const std::vector<DTensor>& tensors);
  
  // 性能模型
  float calculate_bandwidth_efficiency(const std::vector<yica::YISInstruction>& instructions);
  size_t estimate_instruction_cycles(const yica::YISInstruction& instruction);

  // 内部状态
  size_t available_spm_size_;
  std::unordered_map<DTensor*, AccessPattern> tensor_patterns_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA内存访问优化的辅助函数
 */
namespace yica_memory_utils {
  
  /**
   * @brief 检查张量是否适合CIM计算
   */
  bool is_cim_friendly(const DTensor& tensor);
  
  /**
   * @brief 计算最优的SPM分块大小
   */
  std::tuple<int, int, int> calculate_optimal_spm_tiling(
    int M, int N, int K, size_t spm_size);
  
  /**
   * @brief 生成数据预取指令
   */
  std::vector<yica::YISInstruction> generate_prefetch_instructions(
    const DTensor& tensor, const std::vector<int>& future_access_pattern);

} // namespace yica_memory_utils

} // namespace kernel
} // namespace yirage 