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

#include "yirage/kernel/yica_element_ops.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <algorithm>

namespace yirage {
namespace kernel {

// Unary Operations Implementation
YICAElementUnaryOp::YICAElementUnaryOp(Graph *_kgraph,
                                       DTensor const &input,
                                       ElementUnaryType _op_type,
                                       const YICAElementConfig &config)
    : KNOperator(_kgraph, convert_unary_type_to_kn_type(_op_type), input),
      unary_op_type(_op_type), yica_config(config) {
  
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

// Binary Operations Implementation
YICAElementBinaryOp::YICAElementBinaryOp(Graph *_kgraph,
                                         DTensor const &input1,
                                         DTensor const &input2,
                                         ElementBinaryType _op_type,
                                         const YICAElementConfig &config)
    : KNOperator(_kgraph, convert_binary_type_to_kn_type(_op_type), input1, input2),
      binary_op_type(_op_type), yica_config(config) {
  
  // 验证广播兼容性
  assert(input1.num_dims == input2.num_dims);
  
  DTensor output = compute_broadcast_output_shape(input1, input2);
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

YICAElementUnaryOp::~YICAElementUnaryOp() {
  kgraph->free(output_tensors[0]);
}

YICAElementBinaryOp::~YICAElementBinaryOp() {
  kgraph->free(output_tensors[0]);
}

bool YICAElementUnaryOp::initialize_yica_components() {
  // 规划CIM向量化
  cim_vectorization_plan_ = plan_cim_vectorization();
  
  // 规划SPM缓冲
  spm_buffer_plan_ = plan_spm_buffering();
  
  // 分析操作融合机会
  if (yica_config.enable_operation_fusion) {
    fusion_analysis_ = analyze_fusion_opportunities();
  }
  
  // 生成YIS指令
  generated_instructions_ = generate_yis_instructions();
  
  is_optimized_ = true;
  return true;
}

bool YICAElementBinaryOp::initialize_yica_components() {
  // 规划CIM向量化
  cim_vectorization_plan_ = plan_cim_vectorization();
  
  // 规划SPM缓冲
  smp_buffer_plan_ = plan_spm_buffering();
  
  // 分析广播优化
  broadcast_optimization_ = analyze_broadcast_optimization();
  
  // 分析操作融合机会
  if (yica_config.enable_operation_fusion) {
    fusion_analysis_ = analyze_fusion_opportunities();
  }
  
  // 生成YIS指令
  generated_instructions_ = generate_yis_instructions();
  
  is_optimized_ = true;
  return true;
}

YICAElementUnaryOp::CIMVectorizationPlan YICAElementUnaryOp::plan_cim_vectorization() {
  CIMVectorizationPlan plan;
  
  size_t num_elements = input_tensors[0].num_elements();
  size_t elements_per_cim = (num_elements + yica_config.num_cim_arrays - 1) / yica_config.num_cim_arrays;
  
  // 为每个CIM阵列分配元素范围
  for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
    size_t start_idx = cim_id * elements_per_cim;
    size_t end_idx = std::min(start_idx + elements_per_cim, num_elements);
    
    if (start_idx < end_idx) {
      plan.cim_element_ranges.push_back({start_idx, end_idx});
    }
  }
  
  // 计算向量化因子
  plan.vector_width = std::min(yica_config.vector_width, 
                              static_cast<int>(elements_per_cim));
  plan.vectorization_efficiency = std::min(1.0f, 
    static_cast<float>(plan.vector_width) / yica_config.vector_width);
  
  // 估算并行效率
  plan.parallel_efficiency = std::min(1.0f,
    static_cast<float>(yica_config.num_cim_arrays) / 
    std::max(1.0f, static_cast<float>(num_elements / 1024))); // 假设1024元素需要一个CIM
  
  return plan;
}

YICAElementBinaryOp::CIMVectorizationPlan YICAElementBinaryOp::plan_cim_vectorization() {
  CIMVectorizationPlan plan;
  
  size_t num_elements = output_tensors[0].num_elements();
  size_t elements_per_cim = (num_elements + yica_config.num_cim_arrays - 1) / yica_config.num_cim_arrays;
  
  // 为每个CIM阵列分配元素范围
  for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
    size_t start_idx = cim_id * elements_per_cim;
    size_t end_idx = std::min(start_idx + elements_per_cim, num_elements);
    
    if (start_idx < end_idx) {
      plan.cim_element_ranges.push_back({start_idx, end_idx});
    }
  }
  
  // 计算向量化因子
  plan.vector_width = std::min(yica_config.vector_width, 
                              static_cast<int>(elements_per_cim));
  plan.vectorization_efficiency = std::min(1.0f, 
    static_cast<float>(plan.vector_width) / yica_config.vector_width);
  
  // 估算并行效率
  plan.parallel_efficiency = std::min(1.0f,
    static_cast<float>(yica_config.num_cim_arrays) / 
    std::max(1.0f, static_cast<float>(num_elements / 1024)));
  
  return plan;
}

YICAElementBinaryOp::BroadcastOptimization YICAElementBinaryOp::analyze_broadcast_optimization() {
  BroadcastOptimization opt;
  
  const DTensor& input1 = input_tensors[0];
  const DTensor& input2 = input_tensors[1];
  const DTensor& output = output_tensors[0];
  
  // 分析广播模式
  opt.input1_needs_broadcast = false;
  opt.input2_needs_broadcast = false;
  
  for (int i = 0; i < output.num_dims; i++) {
    if (input1.dim[i] == 1 && output.dim[i] > 1) {
      opt.input1_needs_broadcast = true;
      opt.input1_broadcast_dims.push_back(i);
    }
    if (input2.dim[i] == 1 && output.dim[i] > 1) {
      opt.input2_needs_broadcast = true;
      opt.input2_broadcast_dims.push_back(i);
    }
  }
  
  // 计算广播开销
  if (opt.input1_needs_broadcast) {
    opt.input1_broadcast_factor = output.num_elements() / input1.num_elements();
  } else {
    opt.input1_broadcast_factor = 1;
  }
  
  if (opt.input2_needs_broadcast) {
    opt.input2_broadcast_factor = output.num_elements() / input2.num_elements();
  } else {
    opt.input2_broadcast_factor = 1;
  }
  
  // 优化策略
  if (opt.input1_needs_broadcast || opt.input2_needs_broadcast) {
    opt.optimization_strategy = BroadcastOptimization::CIM_BROADCAST;
    opt.expected_speedup = 0.8f; // CIM广播比CPU广播快20%
  } else {
    opt.optimization_strategy = BroadcastOptimization::NO_BROADCAST;
    opt.expected_speedup = 1.0f;
  }
  
  return opt;
}

YICAElementUnaryOp::FusionAnalysis YICAElementUnaryOp::analyze_fusion_opportunities() {
  FusionAnalysis analysis;
  
  // 基于操作类型分析融合机会
  switch (unary_op_type) {
    case ElementUnaryType::EXP:
    case ElementUnaryType::SILU:
    case ElementUnaryType::GELU:
      // 这些操作经常与其他操作融合
      analysis.fusion_potential = 0.8f;
      analysis.recommended_fusion_ops.push_back("scale");
      analysis.recommended_fusion_ops.push_back("bias_add");
      break;
      
    case ElementUnaryType::RELU:
    case ElementUnaryType::CLAMP:
      // 激活函数经常作为融合的最后一步
      analysis.fusion_potential = 0.6f;
      analysis.recommended_fusion_ops.push_back("linear");
      break;
      
    default:
      analysis.fusion_potential = 0.3f;
      break;
  }
  
  analysis.memory_savings = analysis.fusion_potential * input_tensors[0].data_size();
  analysis.compute_savings = analysis.fusion_potential * 0.2f; // 20%计算节省
  
  return analysis;
}

YICAElementBinaryOp::FusionAnalysis YICAElementBinaryOp::analyze_fusion_opportunities() {
  FusionAnalysis analysis;
  
  // 基于操作类型分析融合机会
  switch (binary_op_type) {
    case ElementBinaryType::ADD:
      // ADD经常与其他操作融合
      analysis.fusion_potential = 0.9f;
      analysis.recommended_fusion_ops.push_back("matmul");
      analysis.recommended_fusion_ops.push_back("conv");
      break;
      
    case ElementBinaryType::MUL:
      // MUL也经常融合
      analysis.fusion_potential = 0.8f;
      analysis.recommended_fusion_ops.push_back("norm");
      break;
      
    case ElementBinaryType::DIV:
      // DIV融合机会较少
      analysis.fusion_potential = 0.4f;
      break;
      
    default:
      analysis.fusion_potential = 0.5f;
      break;
  }
  
  analysis.memory_savings = analysis.fusion_potential * 
    (input_tensors[0].data_size() + input_tensors[1].data_size());
  analysis.compute_savings = analysis.fusion_potential * 0.15f;
  
  return analysis;
}

std::vector<yica::YISInstruction> YICAElementUnaryOp::generate_yis_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 为每个CIM阵列生成指令
  for (size_t i = 0; i < cim_vectorization_plan_.cim_element_ranges.size(); i++) {
    const auto& range = cim_vectorization_plan_.cim_element_ranges[i];
    
    yica::YISInstruction instr;
    instr.type = yica::YISInstructionType::YISMMA;
    instr.operation = convert_unary_to_yis_operation(unary_op_type);
    instr.cim_array_id = i;
    instr.vector_length = range.second - range.first;
    instr.vector_width = cim_vectorization_plan_.vector_width;
    instr.sync_required = false;
    
    instructions.push_back(instr);
  }
  
  // 添加同步指令
  yica::YISInstruction sync_instr;
  sync_instr.type = yica::YISInstructionType::YISSYNC;
  sync_instr.sync_scope = yica::YISSyncScope::CIM_ARRAY_LEVEL;
  instructions.push_back(sync_instr);
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAElementBinaryOp::generate_yis_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 为每个CIM阵列生成指令
  for (size_t i = 0; i < cim_vectorization_plan_.cim_element_ranges.size(); i++) {
    const auto& range = cim_vectorization_plan_.cim_element_ranges[i];
    
    yica::YISInstruction instr;
    instr.type = yica::YISInstructionType::YISMMA;
    instr.operation = convert_binary_to_yis_operation(binary_op_type);
    instr.cim_array_id = i;
    instr.vector_length = range.second - range.first;
    instr.vector_width = cim_vectorization_plan_.vector_width;
    
    // 设置广播信息
    if (broadcast_optimization_.input1_needs_broadcast) {
      instr.broadcast_input1 = true;
      instr.broadcast1_factor = broadcast_optimization_.input1_broadcast_factor;
    }
    if (broadcast_optimization_.input2_needs_broadcast) {
      instr.broadcast_input2 = true;
      instr.broadcast2_factor = broadcast_optimization_.input2_broadcast_factor;
    }
    
    instr.sync_required = false;
    instructions.push_back(instr);
  }
  
  // 添加同步指令
  yica::YISInstruction sync_instr;
  sync_instr.type = yica::YISInstructionType::YISSYNC;
  sync_instr.sync_scope = yica::YISSyncScope::CIM_ARRAY_LEVEL;
  instructions.push_back(sync_instr);
  
  return instructions;
}

bool YICAElementUnaryOp::profile(ProfileResult &result) {
  float computation_time = estimate_computation_time();
  float data_movement_time = estimate_data_movement_time();
  float fusion_savings = estimate_fusion_savings();
  
  result.run_time = computation_time + data_movement_time - fusion_savings;
  
  update_performance_metrics();
  return true;
}

bool YICAElementBinaryOp::profile(ProfileResult &result) {
  float computation_time = estimate_computation_time();
  float data_movement_time = estimate_data_movement_time();
  float broadcast_overhead = estimate_broadcast_overhead();
  float fusion_savings = estimate_fusion_savings();
  
  result.run_time = computation_time + data_movement_time + broadcast_overhead - fusion_savings;
  
  update_performance_metrics();
  return true;
}

float YICAElementUnaryOp::estimate_computation_time() {
  size_t num_elements = input_tensors[0].num_elements();
  float ops_per_element = get_unary_operation_complexity(unary_op_type);
  size_t total_ops = static_cast<size_t>(num_elements * ops_per_element);
  
  float cim_throughput = yica_config.cim_compute_throughput_gops * 1e9;
  float base_time = static_cast<float>(total_ops) / cim_throughput * 1000; // ms
  
  return base_time / cim_vectorization_plan_.parallel_efficiency;
}

float YICAElementBinaryOp::estimate_computation_time() {
  size_t num_elements = output_tensors[0].num_elements();
  float ops_per_element = get_binary_operation_complexity(binary_op_type);
  size_t total_ops = static_cast<size_t>(num_elements * ops_per_element);
  
  float cim_throughput = yica_config.cim_compute_throughput_gops * 1e9;
  float base_time = static_cast<float>(total_ops) / cim_throughput * 1000; // ms
  
  return base_time / cim_vectorization_plan_.parallel_efficiency;
}

float YICAElementBinaryOp::estimate_broadcast_overhead() {
  if (!broadcast_optimization_.input1_needs_broadcast && 
      !broadcast_optimization_.input2_needs_broadcast) {
    return 0.0f;
  }
  
  // 估算广播开销
  size_t broadcast_data = 0;
  if (broadcast_optimization_.input1_needs_broadcast) {
    broadcast_data += input_tensors[0].data_size() * 
                     (broadcast_optimization_.input1_broadcast_factor - 1);
  }
  if (broadcast_optimization_.input2_needs_broadcast) {
    broadcast_data += input_tensors[1].data_size() * 
                     (broadcast_optimization_.input2_broadcast_factor - 1);
  }
  
  float bandwidth = yica_config.memory_bandwidth_gbps * 1e9;
  float overhead = static_cast<float>(broadcast_data) / bandwidth * 1000; // ms
  
  // CIM广播比CPU广播快
  return overhead * (1.0f - broadcast_optimization_.expected_speedup);
}

// 辅助函数实现
yirage::type::KNOperatorType YICAElementUnaryOp::convert_unary_type_to_kn_type(ElementUnaryType type) {
  switch (type) {
    case ElementUnaryType::EXP: return yirage::type::KN_EXP_OP;
    case ElementUnaryType::SQUARE: return yirage::type::KN_SQUARE_OP;
    case ElementUnaryType::SQRT: return yirage::type::KN_SQRT_OP;
    case ElementUnaryType::SILU: return yirage::type::KN_SILU_OP;
    case ElementUnaryType::GELU: return yirage::type::KN_GELU_OP;
    case ElementUnaryType::RELU: return yirage::type::KN_RELU_OP;
    case ElementUnaryType::CLAMP: return yirage::type::KN_CLAMP_OP;
    default: return yirage::type::KN_EXP_OP;
  }
}

yirage::type::KNOperatorType YICAElementBinaryOp::convert_binary_type_to_kn_type(ElementBinaryType type) {
  switch (type) {
    case ElementBinaryType::ADD: return yirage::type::KN_ADD_OP;
    case ElementBinaryType::SUB: return yirage::type::KN_SUB_OP;
    case ElementBinaryType::MUL: return yirage::type::KN_MUL_OP;
    case ElementBinaryType::DIV: return yirage::type::KN_DIV_OP;
    case ElementBinaryType::POW: return yirage::type::KN_POW_OP;
    default: return yirage::type::KN_ADD_OP;
  }
}

DTensor YICAElementBinaryOp::compute_broadcast_output_shape(const DTensor& input1, const DTensor& input2) {
  DTensor output = input1;
  
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }
  
  return output;
}

// 工厂函数实现
YICAElementUnaryOp* create_yica_element_unary_op(
  Graph *graph,
  DTensor const &input,
  YICAElementUnaryOp::ElementUnaryType op_type,
  const YICAElementUnaryOp::YICAElementConfig &config) {
  
  return new YICAElementUnaryOp(graph, input, op_type, config);
}

YICAElementBinaryOp* create_yica_element_binary_op(
  Graph *graph,
  DTensor const &input1,
  DTensor const &input2,
  YICAElementBinaryOp::ElementBinaryType op_type,
  const YICAElementBinaryOp::YICAElementConfig &config) {
  
  return new YICAElementBinaryOp(graph, input1, input2, op_type, config);
}

// JSON序列化
YICAElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"unary_op_type", static_cast<int>(unary_op_type)},
              {"yica_config", yica_config}};
}

YICAElementBinaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"binary_op_type", static_cast<int>(binary_op_type)},
              {"yica_config", yica_config}};
}

} // namespace kernel
} // namespace yirage 