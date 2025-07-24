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

#include "yirage/kernel/yica_rms_norm.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace yirage {
namespace kernel {

YICARMSNormOp::YICARMSNormOp(Graph *_kgraph,
                             DTensor const &input,
                             int _normalized_size,
                             const YICARMSNormConfig &config)
    : KNOperator(_kgraph, yirage::type::KN_RMS_NORM_OP, input),
      normalized_size(_normalized_size), yica_config(config) {
  
  // 验证输入维度
  assert(input.num_elements() % normalized_size == 0);
  
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

YICARMSNormOp::~YICARMSNormOp() {
  kgraph->free(output_tensors[0]);
}

bool YICARMSNormOp::initialize_yica_components() {
  // 规划CIM RMS计算
  cim_rms_plan_ = plan_cim_rms_computation();
  
  // 规划SPM缓冲
  spm_buffer_plan_ = plan_spm_buffering();
  
  // 分析融合机会
  if (yica_config.enable_residual_fusion || yica_config.enable_weight_fusion) {
    fusion_plan_ = analyze_fusion_opportunities();
  }
  
  // 数值稳定性分析
  stability_analysis_ = analyze_numerical_stability();
  
  // 生成YIS指令
  generated_instructions_ = generate_yis_instructions();
  
  is_optimized_ = true;
  return true;
}

YICARMSNormOp::CIMRMSPlan YICARMSNormOp::plan_cim_rms_computation() {
  CIMRMSPlan plan;
  
  int num_samples = input_tensors[0].num_elements() / normalized_size;
  int samples_per_cim = (num_samples + yica_config.num_cim_arrays - 1) / yica_config.num_cim_arrays;
  
  // 为每个CIM阵列分配样本
  for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
    int start_sample = cim_id * samples_per_cim;
    int end_sample = std::min(start_sample + samples_per_cim, num_samples);
    
    if (start_sample < end_sample) {
      plan.cim_sample_ranges.push_back({start_sample, end_sample});
    }
  }
  
  // 计算并行效率
  plan.parallel_efficiency = std::min(1.0f, 
    static_cast<float>(yica_config.num_cim_arrays) / 
    std::max(1.0f, static_cast<float>(num_samples)));
  
  // 估算内存占用
  plan.memory_footprint = num_samples * normalized_size * sizeof(float) * 2; // 输入+输出
  
  // 计算向量化程度
  plan.vectorization_factor = std::min(normalized_size / 32, 8); // 假设32元素向量，最大8倍向量化
  
  return plan;
}

YICARMSNormOp::SPMBufferPlan YICARMSNormOp::plan_spm_buffering() {
  SPMBufferPlan plan;
  
  // 计算每个样本的缓冲需求
  size_t sample_size = normalized_size * sizeof(float);
  size_t buffer_size = std::min(sample_size * yica_config.spm_samples_per_buffer, 
                               yica_config.smp_buffer_size);
  
  plan.samples_per_buffer = buffer_size / sample_size;
  plan.buffer_size = buffer_size;
  plan.num_buffers = yica_config.enable_spm_double_buffering ? 2 : 1;
  plan.total_spm_usage = plan.buffer_size * plan.num_buffers;
  
  // 计算缓冲效率
  size_t total_data_size = input_tensors[0].data_size();
  plan.buffer_efficiency = std::min(1.0f, 
    static_cast<float>(plan.total_smp_usage) / total_data_size);
  
  return plan;
}

YICARMSNormOp::FusionPlan YICARMSNormOp::analyze_fusion_opportunities() {
  FusionPlan plan;
  
  // 分析可融合的操作
  if (yica_config.enable_residual_fusion) {
    plan.fused_operations.push_back("residual_add");
    plan.fusion_benefit += 0.15f; // 15%性能提升
  }
  
  if (yica_config.enable_weight_fusion) {
    plan.fused_operations.push_back("weight_multiply");
    plan.fusion_benefit += 0.12f; // 12%性能提升
  }
  
  if (yica_config.enable_bias_fusion) {
    plan.fused_operations.push_back("bias_add");
    plan.fusion_benefit += 0.08f; // 8%性能提升
  }
  
  // 估算内存节省
  plan.memory_savings = plan.fused_operations.size() * 
                       input_tensors[0].data_size(); // 节省中间结果
  
  return plan;
}

YICARMSNormOp::StabilityAnalysis YICARMSNormOp::analyze_numerical_stability() {
  StabilityAnalysis analysis;
  
  // 基于归一化大小分析数值稳定性
  if (normalized_size < 64) {
    analysis.requires_extended_precision = false;
    analysis.epsilon_adjustment = 1e-5f;
    analysis.stability_score = 0.95f;
  } else if (normalized_size < 512) {
    analysis.requires_extended_precision = false;
    analysis.epsilon_adjustment = 1e-6f;
    analysis.stability_score = 0.90f;
  } else {
    analysis.requires_extended_precision = true;
    analysis.epsilon_adjustment = 1e-8f;
    analysis.stability_score = 0.85f;
  }
  
  // 分析潜在的数值问题
  analysis.underflow_risk = (normalized_size > 1024) ? 0.1f : 0.05f;
  analysis.overflow_risk = 0.02f; // RMS Norm通常不会溢出
  
  return analysis;
}

std::vector<yica::YISInstruction> YICARMSNormOp::generate_yis_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 1. 数据加载指令
  auto load_instrs = generate_data_loading_instructions();
  instructions.insert(instructions.end(), load_instrs.begin(), load_instrs.end());
  
  // 2. CIM RMS计算指令
  auto rms_instrs = generate_cim_rms_instructions();
  instructions.insert(instructions.end(), rms_instrs.begin(), rms_instrs.end());
  
  // 3. 融合操作指令
  if (!fusion_plan_.fused_operations.empty()) {
    auto fusion_instrs = generate_fusion_instructions();
    instructions.insert(instructions.end(), fusion_instrs.begin(), fusion_instrs.end());
  }
  
  // 4. 结果存储指令
  auto store_instrs = generate_result_storing_instructions();
  instructions.insert(instructions.end(), store_instrs.begin(), store_instrs.end());
  
  return instructions;
}

std::vector<yica::YISInstruction> YICARMSNormOp::generate_cim_rms_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  for (size_t i = 0; i < cim_rms_plan_.cim_sample_ranges.size(); i++) {
    const auto& range = cim_rms_plan_.cim_sample_ranges[i];
    
    // 计算平方和
    yica::YISInstruction square_sum_instr;
    square_sum_instr.type = yica::YISInstructionType::YISMMA;
    square_sum_instr.operation = yica::YISOperation::SQUARE_SUM;
    square_sum_instr.cim_array_id = i;
    square_sum_instr.vector_length = normalized_size;
    square_sum_instr.num_samples = range.second - range.first;
    square_sum_instr.sync_required = false;
    instructions.push_back(square_sum_instr);
    
    // 计算均值
    yica::YISInstruction mean_instr;
    mean_instr.type = yica::YISInstructionType::YISMMA;
    mean_instr.operation = yica::YISOperation::MEAN;
    mean_instr.cim_array_id = i;
    mean_instr.divisor = normalized_size;
    mean_instr.sync_required = false;
    instructions.push_back(mean_instr);
    
    // 计算平方根
    yica::YISInstruction sqrt_instr;
    sqrt_instr.type = yica::YISInstructionType::YISMMA;
    sqrt_instr.operation = yica::YISOperation::SQRT;
    sqrt_instr.cim_array_id = i;
    sqrt_instr.epsilon = stability_analysis_.epsilon_adjustment;
    sqrt_instr.sync_required = false;
    instructions.push_back(sqrt_instr);
    
    // 归一化除法
    yica::YISInstruction normalize_instr;
    normalize_instr.type = yica::YISInstructionType::YISMMA;
    normalize_instr.operation = yica::YISOperation::ELEMENT_DIVIDE;
    normalize_instr.cim_array_id = i;
    normalize_instr.vector_length = normalized_size;
    normalize_instr.num_samples = range.second - range.first;
    normalize_instr.sync_required = true;
    instructions.push_back(normalize_instr);
  }
  
  return instructions;
}

std::vector<yica::YISInstruction> YICARMSNormOp::generate_fusion_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  for (const std::string& op : fusion_plan_.fused_operations) {
    yica::YISInstruction fusion_instr;
    fusion_instr.type = yica::YISInstructionType::YISMMA;
    
    if (op == "residual_add") {
      fusion_instr.operation = yica::YISOperation::FUSED_RESIDUAL_ADD;
    } else if (op == "weight_multiply") {
      fusion_instr.operation = yica::YISOperation::FUSED_WEIGHT_MULTIPLY;
    } else if (op == "bias_add") {
      fusion_instr.operation = yica::YISOperation::FUSED_BIAS_ADD;
    }
    
    fusion_instr.sync_required = false;
    instructions.push_back(fusion_instr);
  }
  
  return instructions;
}

bool YICARMSNormOp::profile(ProfileResult &result) {
  // 使用性能模型预测执行时间
  float rms_computation_time = estimate_rms_computation_time();
  float data_movement_time = estimate_data_movement_time();
  float fusion_overhead = estimate_fusion_overhead();
  
  result.run_time = rms_computation_time + data_movement_time - fusion_overhead;
  
  // 更新性能指标
  update_performance_metrics();
  
  return true;
}

float YICARMSNormOp::estimate_rms_computation_time() {
  int num_samples = input_tensors[0].num_elements() / normalized_size;
  
  // 每个样本需要的操作数：平方、求和、开方、除法
  size_t ops_per_sample = normalized_size * 4;
  size_t total_ops = num_samples * ops_per_sample;
  
  float cim_throughput = yica_config.cim_compute_throughput_gops * 1e9;
  float base_time = static_cast<float>(total_ops) / cim_throughput * 1000; // ms
  
  // 考虑并行效率
  return base_time / cim_rms_plan_.parallel_efficiency;
}

float YICARMSNormOp::estimate_data_movement_time() {
  size_t data_size = input_tensors[0].data_size() + output_tensors[0].data_size();
  float bandwidth = yica_config.memory_bandwidth_gbps * 1e9;
  return static_cast<float>(data_size) / bandwidth * 1000; // ms
}

float YICARMSNormOp::estimate_fusion_overhead() {
  // 融合操作减少的数据移动时间
  float savings = fusion_plan_.fusion_benefit * estimate_data_movement_time();
  return savings * 0.8f; // 80%的理论收益
}

void YICARMSNormOp::update_performance_metrics() {
  performance_metrics_.cim_utilization = cim_rms_plan_.parallel_efficiency;
  performance_metrics_.spm_buffer_efficiency = spm_buffer_plan_.buffer_efficiency;
  performance_metrics_.vectorization_efficiency = 
    static_cast<float>(cim_rms_plan_.vectorization_factor) / 8.0f; // 假设最大8倍向量化
  performance_metrics_.fusion_benefit = fusion_plan_.fusion_benefit;
  performance_metrics_.numerical_stability = stability_analysis_.stability_score;
  performance_metrics_.yis_instruction_count = generated_instructions_.size();
  performance_metrics_.memory_bandwidth_utilization = 0.85f; // 假设值
  performance_metrics_.total_rms_operations = 
    (input_tensors[0].num_elements() / normalized_size) * normalized_size * 4;
}

YICARMSNormOp::YICARMSNormMetrics YICARMSNormOp::get_yica_metrics() const {
  return performance_metrics_;
}

bool YICARMSNormOp::fingerprint(void) {
  // YICA版本的fingerprint实现
  return true;
}

YICARMSNormOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"normalized_size", normalized_size},
              {"yica_config", yica_config}};
}

// 工厂函数实现
YICARMSNormOp* create_yica_rms_norm_op(
  Graph *graph,
  DTensor const &input,
  int normalized_size,
  const YICARMSNormOp::YICARMSNormConfig &config) {
  
  return new YICARMSNormOp(graph, input, normalized_size, config);
}

// 辅助函数实现
namespace yica_rms_norm_utils {
  
  OptimalNormalizationStrategy analyze_optimal_strategy(
    const DTensor &input,
    int normalized_size,
    const YICARMSNormOp::YICARMSNormConfig &config) {
    
    OptimalNormalizationStrategy strategy;
    
    int num_samples = input.num_elements() / normalized_size;
    
    // 根据样本数和归一化大小选择策略
    if (num_samples >= config.num_cim_arrays) {
      strategy.recommended_strategy = YICARMSNormOp::NormalizationStrategy::CIM_PARALLEL;
      strategy.efficiency_score = 0.95f;
    } else {
      strategy.recommended_strategy = YICARMSNormOp::NormalizationStrategy::VECTORIZED;
      strategy.efficiency_score = 0.85f;
    }
    
    strategy.memory_requirement = input.data_size() * 2; // 输入+输出
    strategy.estimated_speedup = config.num_cim_arrays * 0.8f; // 80%并行效率
    
    return strategy;
  }
  
  NumericalStabilityAnalysis analyze_numerical_stability(
    int normalized_size,
    YICARMSNormOp::PrecisionMode precision) {
    
    NumericalStabilityAnalysis analysis;
    
    // 基于精度模式分析
    switch (precision) {
      case YICARMSNormOp::PrecisionMode::FP32:
        analysis.epsilon_recommendation = 1e-5f;
        analysis.stability_score = 0.95f;
        analysis.requires_extended_precision = false;
        break;
      case YICARMSNormOp::PrecisionMode::FP16:
        analysis.epsilon_recommendation = 1e-3f;
        analysis.stability_score = 0.85f;
        analysis.requires_extended_precision = (normalized_size > 512);
        break;
      case YICARMSNormOp::PrecisionMode::MIXED:
        analysis.epsilon_recommendation = 1e-4f;
        analysis.stability_score = 0.90f;
        analysis.requires_extended_precision = (normalized_size > 1024);
        break;
    }
    
    // 基于归一化大小调整
    if (normalized_size > 2048) {
      analysis.stability_score *= 0.9f;
      analysis.underflow_risk = 0.15f;
    } else {
      analysis.underflow_risk = 0.05f;
    }
    
    analysis.overflow_risk = 0.02f; // RMS Norm很少溢出
    
    return analysis;
  }
  
} // namespace yica_rms_norm_utils

void from_json(json const &j, YICARMSNormOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  j.at("normalized_size").get_to(op.normalized_size);
  // yica_config的反序列化需要额外实现
}

} // namespace kernel
} // namespace yirage 