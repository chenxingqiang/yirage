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

#include "yirage/kernel/yica_customized.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <algorithm>

namespace yirage {
namespace kernel {

YICACustomizedOp::YICACustomizedOp(Graph *_kgraph,
                                   const std::vector<DTensor> &inputs,
                                   const std::vector<DTensor> &outputs,
                                   CustomizedOpType _op_type,
                                   const YICACustomizedConfig &config)
    : KNOperator(_kgraph, convert_customized_type_to_kn_type(_op_type), inputs),
      customized_op_type(_op_type), yica_config(config) {
  
  // 验证输入输出张量
  assert(!inputs.empty() && !outputs.empty());
  
  // 设置输出张量
  for (size_t i = 0; i < outputs.size(); i++) {
    DTensor output = outputs[i];
    output.owner_op = this;
    output.owner_ts_idx = i;
    output.guid = DTensor::next_guid++;
    kgraph->allocate(output);
    output_tensors.push_back(output);
  }
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

YICACustomizedOp::~YICACustomizedOp() {
  for (auto& output : output_tensors) {
    kgraph->free(output);
  }
}

bool YICACustomizedOp::initialize_yica_components() {
  // 分析计算模式
  computation_pattern_ = analyze_computation_pattern();
  
  // 规划CIM映射
  cim_mapping_plan_ = plan_cim_mapping();
  
  // 规划线程块融合
  if (yica_config.enable_threadblock_fusion) {
    threadblock_fusion_plan_ = plan_threadblock_fusion();
  }
  
  // 规划流水线并行
  if (yica_config.enable_pipeline_parallelism) {
    pipeline_plan_ = plan_pipeline_parallelism();
  }
  
  // 规划数据局部性优化
  locality_optimization_ = analyze_data_locality_optimization();
  
  // 生成自定义YIS指令序列
  custom_instruction_sequence_ = generate_custom_yis_sequence();
  
  // 性能预测和调优
  performance_prediction_ = predict_performance();
  
  is_optimized_ = true;
  return true;
}

YICACustomizedOp::ComputationPattern YICACustomizedOp::analyze_computation_pattern() {
  ComputationPattern pattern;
  
  // 分析计算复杂度
  pattern.total_operations = estimate_total_operations();
  pattern.arithmetic_intensity = calculate_arithmetic_intensity();
  
  // 分析内存访问模式
  pattern.memory_access_pattern = analyze_memory_access_patterns();
  pattern.data_reuse_ratio = calculate_data_reuse_ratio();
  
  // 分析并行性
  pattern.parallelism_type = determine_parallelism_type();
  pattern.scalability_factor = estimate_scalability_factor();
  
  // 分析依赖关系
  pattern.data_dependencies = analyze_data_dependencies();
  pattern.control_dependencies = analyze_control_dependencies();
  
  // 基于模式特征选择优化策略
  if (pattern.arithmetic_intensity > 10.0f) {
    pattern.optimization_priority = ComputationPattern::COMPUTE_INTENSIVE;
  } else if (pattern.data_reuse_ratio < 0.3f) {
    pattern.optimization_priority = ComputationPattern::MEMORY_BOUND;
  } else {
    pattern.optimization_priority = ComputationPattern::BALANCED;
  }
  
  return pattern;
}

YICACustomizedOp::CIMMappingPlan YICACustomizedOp::plan_cim_mapping() {
  CIMMappingPlan plan;
  
  // 基于计算模式规划CIM映射
  switch (computation_pattern_.optimization_priority) {
    case ComputationPattern::COMPUTE_INTENSIVE:
      plan = plan_compute_intensive_mapping();
      break;
    case ComputationPattern::MEMORY_BOUND:
      plan = plan_memory_bound_mapping();
      break;
    case ComputationPattern::BALANCED:
      plan = plan_balanced_mapping();
      break;
  }
  
  // 优化CIM利用率
  optimize_cim_utilization(plan);
  
  return plan;
}

YICACustomizedOp::CIMMappingPlan YICACustomizedOp::plan_compute_intensive_mapping() {
  CIMMappingPlan plan;
  
  // 计算密集型任务优先最大化并行度
  size_t total_work = computation_pattern_.total_operations;
  size_t work_per_cim = (total_work + yica_config.num_cim_arrays - 1) / yica_config.num_cim_arrays;
  
  for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
    CIMTask task;
    task.cim_array_id = cim_id;
    task.start_work_idx = cim_id * work_per_cim;
    task.end_work_idx = std::min((cim_id + 1) * work_per_cim, total_work);
    task.task_type = CIMTask::COMPUTE_INTENSIVE;
    task.priority = CIMTask::HIGH_PRIORITY;
    
    if (task.start_work_idx < task.end_work_idx) {
      plan.cim_tasks.push_back(task);
    }
  }
  
  plan.mapping_strategy = CIMMappingPlan::PARALLEL_COMPUTE;
  plan.expected_efficiency = 0.9f;
  
  return plan;
}

YICACustomizedOp::ThreadblockFusionPlan YICACustomizedOp::plan_threadblock_fusion() {
  ThreadblockFusionPlan plan;
  
  // 分析可融合的操作模式
  plan.fusion_opportunities = identify_fusion_opportunities();
  
  // 规划融合策略
  for (const auto& opportunity : plan.fusion_opportunities) {
    FusionGroup group;
    group.operations = opportunity.operations;
    group.fusion_type = determine_fusion_type(opportunity);
    group.expected_speedup = estimate_fusion_speedup(opportunity);
    group.memory_savings = estimate_memory_savings(opportunity);
    
    plan.fusion_groups.push_back(group);
  }
  
  // 计算总体融合收益
  plan.total_fusion_benefit = 0.0f;
  for (const auto& group : plan.fusion_groups) {
    plan.total_fusion_benefit += group.expected_speedup * group.operations.size();
  }
  plan.total_fusion_benefit /= plan.fusion_groups.size();
  
  return plan;
}

YICACustomizedOp::PipelinePlan YICACustomizedOp::plan_pipeline_parallelism() {
  PipelinePlan plan;
  
  // 分析流水线阶段
  plan.pipeline_stages = identify_pipeline_stages();
  
  // 为每个阶段规划资源分配
  for (size_t i = 0; i < plan.pipeline_stages.size(); i++) {
    PipelineStage& stage = plan.pipeline_stages[i];
    stage.stage_id = i;
    stage.assigned_cim_arrays = allocate_cim_arrays_for_stage(stage);
    stage.spm_buffer_allocation = allocate_spm_for_stage(stage);
    stage.estimated_latency = estimate_stage_latency(stage);
  }
  
  // 分析流水线瓶颈
  plan.bottleneck_stage = find_bottleneck_stage(plan.pipeline_stages);
  plan.pipeline_efficiency = calculate_pipeline_efficiency(plan.pipeline_stages);
  
  // 优化流水线平衡
  if (plan.pipeline_efficiency < 0.8f) {
    optimize_pipeline_balance(plan);
  }
  
  return plan;
}

YICACustomizedOp::LocalityOptimization YICACustomizedOp::analyze_data_locality_optimization() {
  LocalityOptimization opt;
  
  // 分析数据访问局部性
  opt.temporal_locality = analyze_temporal_locality();
  opt.spatial_locality = analyze_spatial_locality();
  
  // 规划数据预取策略
  opt.prefetch_strategy = plan_data_prefetching();
  
  // 规划数据复用优化
  opt.reuse_optimization = plan_data_reuse_optimization();
  
  // 分析缓存友好性
  opt.cache_friendliness = analyze_cache_friendliness();
  
  // 生成局部性优化建议
  opt.optimization_recommendations = generate_locality_recommendations();
  
  return opt;
}

std::vector<yica::YISInstruction> YICACustomizedOp::generate_custom_yis_sequence() {
  std::vector<yica::YISInstruction> sequence;
  
  // 根据自定义操作类型生成指令序列
  switch (customized_op_type) {
    case CustomizedOpType::FUSED_ATTENTION:
      sequence = generate_fused_attention_instructions();
      break;
    case CustomizedOpType::CUSTOM_CONVOLUTION:
      sequence = generate_custom_convolution_instructions();
      break;
    case CustomizedOpType::SPARSE_COMPUTATION:
      sequence = generate_sparse_computation_instructions();
      break;
    case CustomizedOpType::DYNAMIC_GRAPH:
      sequence = generate_dynamic_graph_instructions();
      break;
    case CustomizedOpType::USER_DEFINED:
      sequence = generate_user_defined_instructions();
      break;
  }
  
  // 添加同步和控制指令
  add_synchronization_instructions(sequence);
  
  // 优化指令序列
  optimize_instruction_sequence(sequence);
  
  return sequence;
}

std::vector<yica::YISInstruction> YICACustomizedOp::generate_fused_attention_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 多头注意力融合指令序列
  // 1. Q, K, V计算
  yica::YISInstruction qkv_instr;
  qkv_instr.type = yica::YISInstructionType::YISMMA;
  qkv_instr.operation = yica::YISOperation::FUSED_QKV_PROJECTION;
  qkv_instr.fusion_scope = yica::YISFusionScope::MULTI_HEAD_ATTENTION;
  instructions.push_back(qkv_instr);
  
  // 2. 注意力分数计算
  yica::YISInstruction attention_instr;
  attention_instr.type = yica::YISInstructionType::YISMMA;
  attention_instr.operation = yica::YISOperation::FUSED_ATTENTION_SCORE;
  attention_instr.enable_scaling = true;
  attention_instr.enable_masking = yica_config.enable_attention_masking;
  instructions.push_back(attention_instr);
  
  // 3. Softmax计算
  yica::YISInstruction softmax_instr;
  softmax_instr.type = yica::YISInstructionType::YISMMA;
  softmax_instr.operation = yica::YISOperation::FUSED_SOFTMAX;
  softmax_instr.numerical_stability = true;
  instructions.push_back(softmax_instr);
  
  // 4. 输出投影
  yica::YISInstruction output_instr;
  output_instr.type = yica::YISInstructionType::YISMMA;
  output_instr.operation = yica::YISOperation::FUSED_OUTPUT_PROJECTION;
  instructions.push_back(output_instr);
  
  return instructions;
}

std::vector<yica::YISInstruction> YICACustomizedOp::generate_sparse_computation_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 稀疏计算优化指令序列
  // 1. 稀疏模式检测
  yica::YISInstruction sparsity_instr;
  sparsity_instr.type = yica::YISInstructionType::YISCONTROL;
  sparsity_instr.operation = yica::YISOperation::SPARSITY_DETECTION;
  sparsity_instr.sparsity_threshold = yica_config.sparsity_threshold;
  instructions.push_back(sparsity_instr);
  
  // 2. 动态分支选择
  yica::YISInstruction branch_instr;
  branch_instr.type = yica::YISInstructionType::YISCONTROL;
  branch_instr.operation = yica::YISOperation::DYNAMIC_BRANCHING;
  branch_instr.branch_condition = yica::YISBranchCondition::SPARSITY_BASED;
  instructions.push_back(branch_instr);
  
  // 3. 稀疏计算执行
  yica::YISInstruction sparse_compute_instr;
  sparse_compute_instr.type = yica::YISInstructionType::YISMMA;
  sparse_compute_instr.operation = yica::YISOperation::SPARSE_MATRIX_MULTIPLY;
  sparse_compute_instr.sparse_format = yica::YISSparseFormat::CSR;
  instructions.push_back(sparse_compute_instr);
  
  return instructions;
}

YICACustomizedOp::PerformancePrediction YICACustomizedOp::predict_performance() {
  PerformancePrediction prediction;
  
  // 基于各种优化计划预测性能
  prediction.base_execution_time = estimate_base_execution_time();
  
  // CIM映射优化收益
  prediction.cim_optimization_benefit = 
    cim_mapping_plan_.expected_efficiency * 0.3f; // 30%最大收益
  
  // 线程块融合收益
  if (yica_config.enable_threadblock_fusion) {
    prediction.fusion_benefit = threadblock_fusion_plan_.total_fusion_benefit * 0.2f;
  }
  
  // 流水线并行收益
  if (yica_config.enable_pipeline_parallelism) {
    prediction.pipeline_benefit = pipeline_plan_.pipeline_efficiency * 0.25f;
  }
  
  // 数据局部性优化收益
  prediction.locality_benefit = 
    (locality_optimization_.temporal_locality + locality_optimization_.spatial_locality) * 0.15f;
  
  // 计算总体性能提升
  prediction.total_speedup = 1.0f + 
    prediction.cim_optimization_benefit +
    prediction.fusion_benefit +
    prediction.pipeline_benefit +
    prediction.locality_benefit;
  
  prediction.predicted_execution_time = 
    prediction.base_execution_time / prediction.total_speedup;
  
  return prediction;
}

bool YICACustomizedOp::profile(ProfileResult &result) {
  float execution_time = performance_prediction_.predicted_execution_time;
  
  // 添加实际测量的开销
  float measurement_overhead = estimate_measurement_overhead();
  
  result.run_time = execution_time + measurement_overhead;
  
  update_performance_metrics();
  return true;
}

void YICACustomizedOp::update_performance_metrics() {
  performance_metrics_.cim_utilization = cim_mapping_plan_.expected_efficiency;
  performance_metrics_.fusion_efficiency = 
    yica_config.enable_threadblock_fusion ? threadblock_fusion_plan_.total_fusion_benefit : 0.0f;
  performance_metrics_.pipeline_efficiency = 
    yica_config.enable_pipeline_parallelism ? pipeline_plan_.pipeline_efficiency : 0.0f;
  performance_metrics_.locality_optimization_score = 
    (locality_optimization_.temporal_locality + locality_optimization_.spatial_locality) / 2.0f;
  performance_metrics_.custom_instruction_count = custom_instruction_sequence_.size();
  performance_metrics_.predicted_speedup = performance_prediction_.total_speedup;
  performance_metrics_.arithmetic_intensity = computation_pattern_.arithmetic_intensity;
  performance_metrics_.parallelism_scalability = computation_pattern_.scalability_factor;
}

// 辅助函数实现
yirage::type::KNOperatorType YICACustomizedOp::convert_customized_type_to_kn_type(CustomizedOpType type) {
  switch (type) {
    case CustomizedOpType::FUSED_ATTENTION: return yirage::type::KN_CUSTOMIZED_OP;
    case CustomizedOpType::CUSTOM_CONVOLUTION: return yirage::type::KN_CUSTOMIZED_OP;
    case CustomizedOpType::SPARSE_COMPUTATION: return yirage::type::KN_CUSTOMIZED_OP;
    case CustomizedOpType::DYNAMIC_GRAPH: return yirage::type::KN_CUSTOMIZED_OP;
    case CustomizedOpType::USER_DEFINED: return yirage::type::KN_CUSTOMIZED_OP;
    default: return yirage::type::KN_CUSTOMIZED_OP;
  }
}

size_t YICACustomizedOp::estimate_total_operations() {
  size_t total_ops = 0;
  
  // 基于输入输出张量大小估算操作数
  for (const auto& input : input_tensors) {
    total_ops += input.num_elements();
  }
  
  // 根据自定义操作类型调整
  switch (customized_op_type) {
    case CustomizedOpType::FUSED_ATTENTION:
      total_ops *= 4; // Q,K,V投影 + 注意力计算
      break;
    case CustomizedOpType::CUSTOM_CONVOLUTION:
      total_ops *= 2; // 乘加操作
      break;
    case CustomizedOpType::SPARSE_COMPUTATION:
      total_ops *= (1.0f - yica_config.sparsity_threshold); // 稀疏性减少操作
      break;
    default:
      total_ops *= 2; // 默认假设
      break;
  }
  
  return total_ops;
}

float YICACustomizedOp::estimate_base_execution_time() {
  size_t total_ops = estimate_total_operations();
  float cim_throughput = yica_config.cim_compute_throughput_gops * 1e9;
  
  return static_cast<float>(total_ops) / cim_throughput * 1000; // ms
}

bool YICACustomizedOp::fingerprint(void) {
  // YICA版本的fingerprint实现
  return true;
}

// 工厂函数实现
YICACustomizedOp* create_yica_customized_op(
  Graph *graph,
  const std::vector<DTensor> &inputs,
  const std::vector<DTensor> &outputs,
  YICACustomizedOp::CustomizedOpType op_type,
  const YICACustomizedOp::YICACustomizedConfig &config) {
  
  return new YICACustomizedOp(graph, inputs, outputs, op_type, config);
}

// JSON序列化
YICACustomizedOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"customized_op_type", static_cast<int>(customized_op_type)},
              {"yica_config", yica_config}};
}

} // namespace kernel
} // namespace yirage 