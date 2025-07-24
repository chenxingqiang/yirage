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

#include "yirage/kernel/yica_reduction.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace yirage {
namespace kernel {

YICAReductionOp::YICAReductionOp(Graph *_kgraph,
                                 DTensor const &input,
                                 int _reduction_dim,
                                 ReductionType _reduction_type,
                                 const YICAReductionConfig &config)
    : KNOperator(_kgraph, convert_reduction_type_to_kn_type(_reduction_type), input),
      reduction_dim(_reduction_dim), reduction_type(_reduction_type), yica_config(config) {
  
  // 验证归约维度
  assert(reduction_dim >= 0 && reduction_dim < input.num_dims);
  
  // 计算输出张量形状
  DTensor output = input;
  output.dim[reduction_dim] = 1; // 归约维度变为1
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  
  kgraph->allocate(output);
  output_tensors.push_back(output);
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

YICAReductionOp::~YICAReductionOp() {
  kgraph->free(output_tensors[0]);
}

bool YICAReductionOp::initialize_yica_components() {
  // 规划CIM并行归约
  cim_reduction_plan_ = plan_cim_parallel_reduction();
  
  // 规划层次化归约
  if (yica_config.enable_hierarchical_reduction) {
    hierarchical_plan_ = plan_hierarchical_reduction();
  }
  
  // 规划树状归约
  if (yica_config.enable_tree_reduction) {
    tree_reduction_plan_ = plan_tree_reduction();
  }
  
  // 规划SPM缓冲
  spm_buffer_plan_ = plan_spm_buffering();
  
  // 分析向量化机会
  vectorization_analysis_ = analyze_vectorization_opportunities();
  
  // 数值稳定性分析
  stability_analysis_ = analyze_numerical_stability();
  
  // 生成YIS指令
  generated_instructions_ = generate_yis_instructions();
  
  is_optimized_ = true;
  return true;
}

YICAReductionOp::CIMReductionPlan YICAReductionOp::plan_cim_parallel_reduction() {
  CIMReductionPlan plan;
  
  const DTensor& input = input_tensors[0];
  size_t reduction_size = input.dim[reduction_dim];
  size_t elements_per_reduction = input.num_elements() / reduction_size;
  
  // 计算每个CIM阵列处理的归约数量
  size_t reductions_per_cim = (elements_per_reduction + yica_config.num_cim_arrays - 1) / 
                             yica_config.num_cim_arrays;
  
  // 为每个CIM阵列分配归约任务
  for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
    size_t start_reduction = cim_id * reductions_per_cim;
    size_t end_reduction = std::min(start_reduction + reductions_per_cim, elements_per_reduction);
    
    if (start_reduction < end_reduction) {
      CIMReductionTask task;
      task.cim_array_id = cim_id;
      task.start_reduction_idx = start_reduction;
      task.end_reduction_idx = end_reduction;
      task.reduction_size = reduction_size;
      task.elements_per_reduction = end_reduction - start_reduction;
      
      plan.cim_tasks.push_back(task);
    }
  }
  
  // 计算并行效率
  plan.parallel_efficiency = std::min(1.0f, 
    static_cast<float>(yica_config.num_cim_arrays) / 
    std::max(1.0f, static_cast<float>(elements_per_reduction / 1024))); // 假设1024个归约需要一个CIM
  
  // 估算内存访问模式
  plan.memory_access_pattern = analyze_memory_access_pattern();
  plan.estimated_memory_footprint = input.data_size() + output_tensors[0].data_size();
  
  return plan;
}

YICAReductionOp::HierarchicalPlan YICAReductionOp::plan_hierarchical_reduction() {
  HierarchicalPlan plan;
  
  size_t reduction_size = input_tensors[0].dim[reduction_dim];
  
  // 计算层次化归约的层数
  plan.num_levels = static_cast<int>(std::ceil(std::log2(reduction_size)));
  plan.level_configs.resize(plan.num_levels);
  
  size_t current_size = reduction_size;
  for (int level = 0; level < plan.num_levels; level++) {
    LevelConfig& config = plan.level_configs[level];
    config.level_id = level;
    config.input_size = current_size;
    config.output_size = (current_size + 1) / 2; // 每层减半
    config.reduction_factor = 2;
    config.num_parallel_reductions = std::min(
      static_cast<size_t>(yica_config.num_cim_arrays), 
      config.output_size);
    
    current_size = config.output_size;
  }
  
  // 估算层次化归约的收益
  plan.efficiency_gain = std::min(0.3f, 
    static_cast<float>(plan.num_levels) * 0.05f); // 每层5%收益，最多30%
  
  return plan;
}

YICAReductionOp::TreeReductionPlan YICAReductionOp::plan_tree_reduction() {
  TreeReductionPlan plan;
  
  size_t reduction_size = input_tensors[0].dim[reduction_dim];
  
  // 计算树的高度和每层节点数
  plan.tree_height = static_cast<int>(std::ceil(std::log2(reduction_size)));
  plan.tree_levels.resize(plan.tree_height);
  
  size_t nodes_at_level = reduction_size;
  for (int level = 0; level < plan.tree_height; level++) {
    TreeLevel& tree_level = plan.tree_levels[level];
    tree_level.level_id = level;
    tree_level.num_nodes = nodes_at_level;
    tree_level.nodes_per_cim = (nodes_at_level + yica_config.num_cim_arrays - 1) / 
                              yica_config.num_cim_arrays;
    
    // 分配节点到CIM阵列
    for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
      size_t start_node = cim_id * tree_level.nodes_per_cim;
      size_t end_node = std::min(start_node + tree_level.nodes_per_cim, nodes_at_level);
      
      if (start_node < end_node) {
        tree_level.cim_node_ranges.push_back({start_node, end_node});
      }
    }
    
    nodes_at_level = (nodes_at_level + 1) / 2;
  }
  
  // 估算树状归约的内存使用
  plan.intermediate_buffer_size = reduction_size * sizeof(float);
  plan.communication_overhead = estimate_tree_communication_overhead();
  
  return plan;
}

YICAReductionOp::SPMBufferPlan YICAReductionOp::plan_spm_buffering() {
  SPMBufferPlan plan;
  
  size_t reduction_size = input_tensors[0].dim[reduction_dim];
  size_t element_size = sizeof(float); // 假设float类型
  
  // 计算缓冲区大小需求
  size_t required_buffer_size = reduction_size * element_size * yica_config.spm_reduction_buffers;
  size_t available_buffer_size = std::min(required_buffer_size, yica_config.spm_buffer_size);
  
  plan.buffer_size = available_buffer_size;
  plan.elements_per_buffer = available_buffer_size / element_size / yica_config.spm_reduction_buffers;
  plan.num_buffers = yica_config.spm_reduction_buffers;
  plan.enable_double_buffering = yica_config.enable_spm_double_buffering;
  
  // 计算缓冲效率
  plan.buffer_efficiency = std::min(1.0f, 
    static_cast<float>(available_buffer_size) / required_buffer_size);
  
  // 分析缓冲策略
  if (plan.elements_per_buffer >= reduction_size) {
    plan.buffering_strategy = SPMBufferPlan::FULL_BUFFER;
  } else if (plan.elements_per_buffer >= reduction_size / 2) {
    plan.buffering_strategy = SPMBufferPlan::PARTIAL_BUFFER;
  } else {
    plan.buffering_strategy = SPMBufferPlan::STREAMING_BUFFER;
  }
  
  return plan;
}

YICAReductionOp::VectorizationAnalysis YICAReductionOp::analyze_vectorization_opportunities() {
  VectorizationAnalysis analysis;
  
  size_t reduction_size = input_tensors[0].dim[reduction_dim];
  
  // 分析向量化宽度
  analysis.optimal_vector_width = std::min(
    static_cast<size_t>(yica_config.vector_width),
    reduction_size);
  
  // 计算向量化效率
  analysis.vectorization_efficiency = 
    static_cast<float>(analysis.optimal_vector_width) / yica_config.vector_width;
  
  // 分析不同归约类型的向量化收益
  switch (reduction_type) {
    case ReductionType::SUM:
    case ReductionType::MEAN:
      analysis.vectorization_benefit = 0.8f; // 80%收益
      break;
    case ReductionType::MAX:
    case ReductionType::MIN:
      analysis.vectorization_benefit = 0.6f; // 60%收益
      break;
    case ReductionType::PROD:
      analysis.vectorization_benefit = 0.7f; // 70%收益
      break;
    default:
      analysis.vectorization_benefit = 0.5f;
      break;
  }
  
  // 估算向量化后的性能提升
  analysis.estimated_speedup = 1.0f + 
    analysis.vectorization_efficiency * analysis.vectorization_benefit;
  
  return analysis;
}

YICAReductionOp::StabilityAnalysis YICAReductionOp::analyze_numerical_stability() {
  StabilityAnalysis analysis;
  
  size_t reduction_size = input_tensors[0].dim[reduction_dim];
  
  // 基于归约类型和大小分析数值稳定性
  switch (reduction_type) {
    case ReductionType::SUM:
    case ReductionType::MEAN:
      if (reduction_size > 10000) {
        analysis.requires_extended_precision = true;
        analysis.stability_score = 0.7f;
        analysis.overflow_risk = 0.2f;
      } else {
        analysis.requires_extended_precision = false;
        analysis.stability_score = 0.9f;
        analysis.overflow_risk = 0.05f;
      }
      analysis.underflow_risk = 0.02f;
      break;
      
    case ReductionType::PROD:
      // 乘积归约容易溢出或下溢
      analysis.requires_extended_precision = (reduction_size > 100);
      analysis.stability_score = 0.6f;
      analysis.overflow_risk = 0.3f;
      analysis.underflow_risk = 0.3f;
      break;
      
    case ReductionType::MAX:
    case ReductionType::MIN:
      // 最值归约数值稳定
      analysis.requires_extended_precision = false;
      analysis.stability_score = 0.95f;
      analysis.overflow_risk = 0.01f;
      analysis.underflow_risk = 0.01f;
      break;
  }
  
  // 基于数值稳定性调整归约策略
  if (analysis.stability_score < 0.8f) {
    analysis.recommended_strategy = StabilityAnalysis::KAHAN_SUMMATION;
  } else {
    analysis.recommended_strategy = StabilityAnalysis::STANDARD_REDUCTION;
  }
  
  return analysis;
}

std::vector<yica::YISInstruction> YICAReductionOp::generate_yis_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 生成数据加载指令
  auto load_instrs = generate_data_loading_instructions();
  instructions.insert(instructions.end(), load_instrs.begin(), load_instrs.end());
  
  // 根据策略生成归约指令
  if (yica_config.enable_hierarchical_reduction) {
    auto hierarchical_instrs = generate_hierarchical_reduction_instructions();
    instructions.insert(instructions.end(), hierarchical_instrs.begin(), hierarchical_instrs.end());
  } else if (yica_config.enable_tree_reduction) {
    auto tree_instrs = generate_tree_reduction_instructions();
    instructions.insert(instructions.end(), tree_instrs.begin(), tree_instrs.end());
  } else {
    auto parallel_instrs = generate_parallel_reduction_instructions();
    instructions.insert(instructions.end(), parallel_instrs.begin(), parallel_instrs.end());
  }
  
  // 生成结果存储指令
  auto store_instrs = generate_result_storing_instructions();
  instructions.insert(instructions.end(), store_instrs.begin(), store_instrs.end());
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAReductionOp::generate_parallel_reduction_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 为每个CIM任务生成指令
  for (const auto& task : cim_reduction_plan_.cim_tasks) {
    yica::YISInstruction instr;
    instr.type = yica::YISInstructionType::YISMMA;
    instr.operation = convert_reduction_type_to_yis_operation(reduction_type);
    instr.cim_array_id = task.cim_array_id;
    instr.reduction_dim = reduction_dim;
    instr.reduction_size = task.reduction_size;
    instr.num_reductions = task.elements_per_reduction;
    instr.vector_width = vectorization_analysis_.optimal_vector_width;
    
    // 设置数值稳定性选项
    if (stability_analysis_.requires_extended_precision) {
      instr.use_extended_precision = true;
    }
    if (stability_analysis_.recommended_strategy == StabilityAnalysis::KAHAN_SUMMATION) {
      instr.use_kahan_summation = true;
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

std::vector<yica::YISInstruction> YICAReductionOp::generate_hierarchical_reduction_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 为每层生成指令
  for (const auto& level_config : hierarchical_plan_.level_configs) {
    yica::YISInstruction level_instr;
    level_instr.type = yica::YISInstructionType::YISMMA;
    level_instr.operation = convert_reduction_type_to_yis_operation(reduction_type);
    level_instr.hierarchical_level = level_config.level_id;
    level_instr.input_size = level_config.input_size;
    level_instr.output_size = level_config.output_size;
    level_instr.reduction_factor = level_config.reduction_factor;
    level_instr.sync_required = true; // 层间需要同步
    
    instructions.push_back(level_instr);
  }
  
  return instructions;
}

bool YICAReductionOp::profile(ProfileResult &result) {
  float computation_time = estimate_computation_time();
  float data_movement_time = estimate_data_movement_time();
  float synchronization_overhead = estimate_synchronization_overhead();
  
  result.run_time = computation_time + data_movement_time + synchronization_overhead;
  
  update_performance_metrics();
  return true;
}

float YICAReductionOp::estimate_computation_time() {
  size_t total_operations = calculate_total_operations();
  float cim_throughput = yica_config.cim_compute_throughput_gops * 1e9;
  
  float base_time = static_cast<float>(total_operations) / cim_throughput * 1000; // ms
  
  // 考虑并行效率和向量化收益
  float efficiency_factor = cim_reduction_plan_.parallel_efficiency * 
                           vectorization_analysis_.estimated_speedup;
  
  return base_time / efficiency_factor;
}

float YICAReductionOp::estimate_synchronization_overhead() {
  float overhead = 0.0f;
  
  if (yica_config.enable_hierarchical_reduction) {
    // 层次化归约的同步开销
    overhead += hierarchical_plan_.num_levels * yica_config.sync_latency_ns / 1e6; // ms
  }
  
  if (yica_config.enable_tree_reduction) {
    // 树状归约的通信开销
    overhead += tree_reduction_plan_.communication_overhead;
  }
  
  // CIM阵列间的同步开销
  overhead += yica_config.num_cim_arrays * yica_config.sync_latency_ns / 1e6; // ms
  
  return overhead;
}

size_t YICAReductionOp::calculate_total_operations() const {
  size_t reduction_size = input_tensors[0].dim[reduction_dim];
  size_t num_reductions = input_tensors[0].num_elements() / reduction_size;
  
  // 基于归约类型计算操作数
  size_t ops_per_reduction;
  switch (reduction_type) {
    case ReductionType::SUM:
    case ReductionType::MEAN:
      ops_per_reduction = reduction_size - 1; // n-1次加法
      break;
    case ReductionType::PROD:
      ops_per_reduction = reduction_size - 1; // n-1次乘法
      break;
    case ReductionType::MAX:
    case ReductionType::MIN:
      ops_per_reduction = reduction_size - 1; // n-1次比较
      break;
    default:
      ops_per_reduction = reduction_size;
      break;
  }
  
  return num_reductions * ops_per_reduction;
}

void YICAReductionOp::update_performance_metrics() {
  performance_metrics_.cim_utilization = cim_reduction_plan_.parallel_efficiency;
  performance_metrics_.vectorization_efficiency = vectorization_analysis_.vectorization_efficiency;
  performance_metrics_.spm_buffer_efficiency = spm_buffer_plan_.buffer_efficiency;
  performance_metrics_.numerical_stability = stability_analysis_.stability_score;
  performance_metrics_.memory_access_efficiency = 
    analyze_memory_access_efficiency(cim_reduction_plan_.memory_access_pattern);
  performance_metrics_.yis_instruction_count = generated_instructions_.size();
  performance_metrics_.total_reduction_operations = calculate_total_operations();
  
  if (yica_config.enable_hierarchical_reduction) {
    performance_metrics_.hierarchical_efficiency = hierarchical_plan_.efficiency_gain;
  }
}

// 辅助函数实现
yirage::type::KNOperatorType YICAReductionOp::convert_reduction_type_to_kn_type(ReductionType type) {
  switch (type) {
    case ReductionType::SUM: return yirage::type::KN_REDUCTION_SUM_OP;
    case ReductionType::MEAN: return yirage::type::KN_REDUCTION_MEAN_OP;
    case ReductionType::MAX: return yirage::type::KN_REDUCTION_MAX_OP;
    case ReductionType::MIN: return yirage::type::KN_REDUCTION_MIN_OP;
    case ReductionType::PROD: return yirage::type::KN_REDUCTION_PROD_OP;
    default: return yirage::type::KN_REDUCTION_SUM_OP;
  }
}

yica::YISOperation YICAReductionOp::convert_reduction_type_to_yis_operation(ReductionType type) {
  switch (type) {
    case ReductionType::SUM: return yica::YISOperation::REDUCTION_SUM;
    case ReductionType::MEAN: return yica::YISOperation::REDUCTION_MEAN;
    case ReductionType::MAX: return yica::YISOperation::REDUCTION_MAX;
    case ReductionType::MIN: return yica::YISOperation::REDUCTION_MIN;
    case ReductionType::PROD: return yica::YISOperation::REDUCTION_PROD;
    default: return yica::YISOperation::REDUCTION_SUM;
  }
}

bool YICAReductionOp::fingerprint(void) {
  // YICA版本的fingerprint实现
  return true;
}

// 工厂函数实现
YICAReductionOp* create_yica_reduction_op(
  Graph *graph,
  DTensor const &input,
  int reduction_dim,
  YICAReductionOp::ReductionType reduction_type,
  const YICAReductionOp::YICAReductionConfig &config) {
  
  return new YICAReductionOp(graph, input, reduction_dim, reduction_type, config);
}

// JSON序列化
YICAReductionOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"reduction_dim", reduction_dim},
              {"reduction_type", static_cast<int>(reduction_type)},
              {"yica_config", yica_config}};
}

} // namespace kernel
} // namespace yirage 