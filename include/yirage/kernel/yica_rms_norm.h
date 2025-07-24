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

#include "yirage/kernel/operator.h"
#include "yirage/yica/yis_instruction_set.h"
#include <vector>
#include <memory>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用RMS Normalization算子
 * 利用CIM阵列实现存算一体的均方根归一化计算
 */
class YICARMSNormOp : public yirage::kernel::KNOperator {
public:
  YICARMSNormOp(Graph *_graph,
                DTensor const &input,
                std::vector<int> const &normalized_shape,
                const YICARMSNormConfig &config = {});
  
  // 支持带权重的RMS Norm
  YICARMSNormOp(Graph *_graph,
                DTensor const &input,
                DTensor const &weight,
                std::vector<int> const &normalized_shape,
                const YICARMSNormConfig &config = {});
                
  ~YICARMSNormOp();
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;

  // YICA特定功能
  bool optimize_for_cim_computation(int num_cim_arrays);
  bool enable_spm_intermediate_storage(size_t buffer_size);
  bool use_fused_sqrt_reciprocal();
  bool enable_residual_connection_fusion(const DTensor& residual);
  
  // RMS Norm计算模式
  enum class RMSNormMode {
    STANDARD,           // 标准RMS Norm
    FUSED_RESIDUAL,     // 融合残差连接
    FUSED_GATING,       // 融合门控机制
    QUANTIZED,          // 量化版本
    MIXED_PRECISION     // 混合精度
  };
  
  bool set_computation_mode(RMSNormMode mode);
  
  // 精度配置
  enum class ComputePrecision {
    FP32,    // 32位浮点
    FP16,    // 16位浮点
    BF16,    // Brain Float 16
    INT8,    // 8位整数
    MIXED    // 混合精度
  };

  // YICA性能分析
  struct YICARMSNormMetrics {
    float cim_sqrt_efficiency;           // CIM平方根计算效率
    float spm_hit_rate;                  // SPM命中率
    size_t yis_sqrt_instruction_count;   // YIS平方根指令数
    float memory_bandwidth_utilization; // 内存带宽利用率
    float compute_intensity;             // 计算密度
    size_t total_operations;             // 总操作数
    float numerical_stability_score;     // 数值稳定性评分
  };
  
  YICARMSNormMetrics get_yica_metrics() const;

public:
  // YICA RMS Norm配置
  struct YICARMSNormConfig {
    // CIM配置
    int num_cim_arrays = 4;              // CIM阵列数量
    bool enable_cim_sqrt = true;         // 启用CIM内平方根计算
    bool enable_cim_reciprocal = true;   // 启用CIM内倒数计算
    
    // SPM配置
    size_t spm_buffer_size = 32 * 1024 * 1024; // SPM缓冲区大小 (32MB)
    bool enable_spm_caching = true;      // 启用SPM缓存
    
    // 计算配置
    RMSNormMode computation_mode = RMSNormMode::STANDARD;
    ComputePrecision precision = ComputePrecision::FP16;
    float epsilon = 1e-6f;               // 数值稳定性参数
    
    // 融合优化
    bool enable_residual_fusion = false; // 启用残差融合
    bool enable_weight_fusion = true;    // 启用权重融合
    bool enable_bias_fusion = false;     // 启用偏置融合
    
    // 数值优化
    bool enable_numerical_stability = true; // 启用数值稳定性优化
    bool enable_fast_math = false;       // 启用快速数学库
    float stability_threshold = 1e-8f;   // 稳定性阈值
    
    // 内存优化
    bool enable_in_place_computation = true; // 启用原地计算
    bool enable_memory_coalescing = true;    // 启用内存合并访问
  } yica_config;

private:
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_rms_norm_instructions();
  std::vector<yica::YISInstruction> generate_cim_sqrt_instructions();
  std::vector<yica::YISInstruction> generate_fused_residual_instructions();
  
  // CIM计算优化
  struct CIMComputationPlan {
    std::vector<int> cim_array_allocation;
    std::vector<std::pair<int, int>> computation_schedule;
    size_t parallel_degree;
    float expected_efficiency;
  };
  
  CIMComputationPlan plan_cim_computation();
  bool optimize_cim_sqrt_computation();
  
  // SPM内存管理
  struct SPMLayout {
    size_t input_offset;
    size_t square_sum_offset;
    size_t rms_offset;
    size_t output_offset;
    size_t weight_offset;
    size_t total_usage;
  };
  
  SPMLayout plan_spm_layout();
  bool setup_spm_double_buffering();
  
  // 数值稳定性优化
  bool apply_numerical_stability_optimization();
  float calculate_stability_factor(const DTensor& input);
  
  // 融合操作实现
  std::vector<yica::YISInstruction> implement_residual_fusion(const DTensor& residual);
  std::vector<yica::YISInstruction> implement_weight_fusion(const DTensor& weight);
  
  // 性能预测
  float predict_computation_time();
  float predict_memory_access_time();
  size_t estimate_operation_count();

  // 内部状态
  std::vector<int> normalized_shape_;
  bool has_weight_;
  bool has_residual_fusion_;
  YICARMSNormMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA RMS Norm工厂函数
 */
YICARMSNormOp* create_yica_rms_norm(
  Graph *graph,
  DTensor const &input,
  std::vector<int> const &normalized_shape,
  const YICARMSNormOp::YICARMSNormConfig &config = {});

YICARMSNormOp* create_yica_rms_norm_with_weight(
  Graph *graph,
  DTensor const &input,
  DTensor const &weight,
  std::vector<int> const &normalized_shape,
  const YICARMSNormOp::YICARMSNormConfig &config = {});

/**
 * @brief YICA RMS Norm辅助函数
 */
namespace yica_rmsnorm_utils {
  
  /**
   * @brief 计算最优的CIM阵列分配
   */
  std::vector<int> calculate_optimal_cim_allocation(
    const std::vector<int>& normalized_shape,
    int num_available_cim_arrays);
  
  /**
   * @brief 估算RMS Norm的数值稳定性
   */
  float estimate_numerical_stability(
    const DTensor& input,
    float epsilon);
  
  /**
   * @brief 生成融合残差连接的指令
   */
  std::vector<yica::YISInstruction> generate_fused_residual_pattern(
    const DTensor& input,
    const DTensor& residual,
    const std::vector<int>& normalized_shape);
  
  /**
   * @brief 优化SPM访问模式
   */
  struct SPMAccessPattern {
    std::vector<size_t> read_offsets;
    std::vector<size_t> write_offsets;
    size_t access_stride;
    bool coalesced_access;
  };
  
  SPMAccessPattern optimize_spm_access_pattern(
    const std::vector<int>& tensor_shape,
    const std::vector<int>& normalized_shape);

} // namespace yica_rmsnorm_utils

void from_json(json const &j, YICARMSNormOp &op);

} // namespace kernel
} // namespace yirage 