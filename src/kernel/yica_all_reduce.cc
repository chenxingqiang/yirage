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

#include "yirage/kernel/yica_all_reduce.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <iostream>

namespace yirage {
namespace kernel {

YICAAllReduceOp::YICAAllReduceOp(Graph *_kgraph,
                                 DTensor const &input,
                                 bool _inplace,
                                 const YICAAllReduceConfig &config)
    : KNOperator(_kgraph, yirage::type::KN_ALLREDUCE_OP, input),
      inplace(_inplace), yica_config(config) {
  
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  
  if (inplace) {
    assert(output.data_offset == input.data_offset);
    assert(output.fp_offset == input.fp_offset);
  } else {
    kgraph->allocate(output);
  }
  
  output_tensors.push_back(output);
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

YICAAllReduceOp::~YICAAllReduceOp() {
  if (!inplace) {
    kgraph->free(output_tensors[0]);
  }
}

bool YICAAllReduceOp::initialize_yica_components() {
  // 初始化CIM阵列配置
  cim_reduction_plan_ = plan_cim_reduction();
  
  // 初始化SPM缓冲配置
  spm_buffer_plan_ = plan_spm_buffering();
  
  // 初始化层次化归约配置
  hierarchical_plan_ = plan_hierarchical_reduction();
  
  // 生成YIS指令
  generated_instructions_ = generate_yis_instructions();
  
  is_optimized_ = true;
  return true;
}

YICAAllReduceOp::CIMReductionPlan YICAAllReduceOp::plan_cim_reduction() {
  CIMReductionPlan plan;
  
  size_t tensor_size = input_tensors[0].num_elements();
  size_t elements_per_cim = tensor_size / yica_config.num_cim_arrays;
  
  // 为每个CIM阵列分配数据范围
  for (int i = 0; i < yica_config.num_cim_arrays; i++) {
    size_t start_idx = i * elements_per_cim;
    size_t end_idx = (i == yica_config.num_cim_arrays - 1) ? 
                     tensor_size : (i + 1) * elements_per_cim;
    plan.cim_data_ranges.push_back({start_idx, end_idx});
  }
  
  // 计算预期效率
  plan.expected_efficiency = std::min(1.0f, 
    static_cast<float>(yica_config.num_cim_arrays) / 
    static_cast<float>(tensor_size / 1024)); // 假设每1024个元素需要一个CIM
  
  plan.memory_footprint = tensor_size * sizeof(float) * 2; // 输入+临时缓冲
  
  return plan;
}

YICAAllReduceOp::SPMBufferPlan YICAAllReduceOp::plan_spm_buffering() {
  SPMBufferPlan plan;
  
  size_t tensor_size = input_tensors[0].data_size();
  size_t buffer_size = std::min(tensor_size, yica_config.spm_buffer_size);
  
  // 计算需要的缓冲块数
  plan.num_buffer_chunks = (tensor_size + buffer_size - 1) / buffer_size;
  plan.chunk_size = buffer_size;
  plan.double_buffering_enabled = yica_config.enable_spm_double_buffering;
  
  // 如果启用双缓冲，需要两倍的SPM空间
  if (plan.double_buffering_enabled) {
    plan.total_spm_usage = buffer_size * 2;
  } else {
    plan.total_spm_usage = buffer_size;
  }
  
  plan.buffer_efficiency = static_cast<float>(buffer_size) / tensor_size;
  
  return plan;
}

YICAAllReduceOp::HierarchicalPlan YICAAllReduceOp::plan_hierarchical_reduction() {
  HierarchicalPlan plan;
  
  // 配置层次化归约的层级
  if (yica_config.reduction_strategy == ReductionStrategy::HIERARCHICAL_TREE) {
    int num_nodes = yica_config.num_cim_arrays;
    
    // 构建树状归约层级
    while (num_nodes > 1) {
      ReductionLevel level;
      level.num_participants = num_nodes;
      level.reduction_factor = 2; // 二叉树归约
      
      for (int i = 0; i < num_nodes; i += 2) {
        if (i + 1 < num_nodes) {
          level.reduction_pairs.push_back({i, i + 1});
        }
      }
      
      plan.reduction_levels.push_back(level);
      num_nodes = (num_nodes + 1) / 2;
    }
  }
  
  plan.total_levels = plan.reduction_levels.size();
  plan.expected_latency = plan.total_levels * 0.1f; // 假设每层级0.1ms
  
  return plan;
}

std::vector<yica::YISInstruction> YICAAllReduceOp::generate_yis_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 1. 数据加载指令
  instructions.append(generate_data_loading_instructions());
  
  // 2. CIM内存内归约指令
  instructions.append(generate_cim_reduction_instructions());
  
  // 3. 层次化归约指令
  instructions.append(generate_hierarchical_reduction_instructions());
  
  // 4. 结果存储指令
  instructions.append(generate_result_storing_instructions());
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAAllReduceOp::generate_data_loading_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  for (const auto& range : cim_reduction_plan_.cim_data_ranges) {
    yica::YISInstruction load_instr;
    load_instr.type = yica::YISInstructionType::YISECOPY;
    load_instr.src_addr = input_tensors[0].data_offset + range.first * sizeof(float);
    load_instr.dst_addr = 0; // SPM地址
    load_instr.size = (range.second - range.first) * sizeof(float);
    load_instr.sync_required = false;
    instructions.push_back(load_instr);
  }
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAAllReduceOp::generate_cim_reduction_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  for (int i = 0; i < yica_config.num_cim_arrays; i++) {
    yica::YISInstruction reduce_instr;
    reduce_instr.type = yica::YISInstructionType::YISMMA;
    reduce_instr.operation = yica::YISOperation::REDUCE_SUM;
    reduce_instr.cim_array_id = i;
    reduce_instr.sync_required = true;
    instructions.push_back(reduce_instr);
  }
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAAllReduceOp::generate_hierarchical_reduction_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  for (const auto& level : hierarchical_plan_.reduction_levels) {
    for (const auto& pair : level.reduction_pairs) {
      yica::YISInstruction hier_reduce_instr;
      hier_reduce_instr.type = yica::YISInstructionType::YISMMA;
      hier_reduce_instr.operation = yica::YISOperation::HIERARCHICAL_REDUCE;
      hier_reduce_instr.src1_cim_id = pair.first;
      hier_reduce_instr.src2_cim_id = pair.second;
      hier_reduce_instr.sync_required = true;
      instructions.push_back(hier_reduce_instr);
    }
    
    // 层级间同步
    yica::YISInstruction sync_instr;
    sync_instr.type = yica::YISInstructionType::YISSYNC;
    sync_instr.sync_scope = yica::YISSyncScope::CIM_ARRAY_LEVEL;
    instructions.push_back(sync_instr);
  }
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAAllReduceOp::generate_result_storing_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  if (!inplace) {
    yica::YISInstruction store_instr;
    store_instr.type = yica::YISInstructionType::YISECOPY;
    store_instr.src_addr = 0; // SPM结果地址
    store_instr.dst_addr = output_tensors[0].data_offset;
    store_instr.size = output_tensors[0].data_size();
    store_instr.sync_required = false;
    instructions.push_back(store_instr);
  }
  
  return instructions;
}

bool YICAAllReduceOp::profile(ProfileResult &profile) {
  // 使用性能模型预测执行时间
  float data_loading_time = estimate_data_loading_time();
  float cim_reduction_time = estimate_cim_reduction_time();
  float hierarchical_time = estimate_hierarchical_reduction_time();
  float storing_time = estimate_result_storing_time();
  
  profile.run_time = data_loading_time + cim_reduction_time + 
                     hierarchical_time + storing_time;
  
  // 更新性能指标
  update_performance_metrics();
  
  return true;
}

float YICAAllReduceOp::estimate_data_loading_time() {
  size_t data_size = input_tensors[0].data_size();
  float bandwidth = yica_config.memory_bandwidth_gbps * 1e9; // 转换为bytes/s
  return static_cast<float>(data_size) / bandwidth * 1000; // 转换为ms
}

float YICAAllReduceOp::estimate_cim_reduction_time() {
  size_t num_elements = input_tensors[0].num_elements();
  float cim_throughput = yica_config.cim_compute_throughput_gops * 1e9; // ops/s
  return static_cast<float>(num_elements) / cim_throughput * 1000; // 转换为ms
}

float YICAAllReduceOp::estimate_hierarchical_reduction_time() {
  return hierarchical_plan_.expected_latency;
}

float YICAAllReduceOp::estimate_result_storing_time() {
  if (inplace) return 0.0f;
  
  size_t data_size = output_tensors[0].data_size();
  float bandwidth = yica_config.memory_bandwidth_gbps * 1e9;
  return static_cast<float>(data_size) / bandwidth * 1000;
}

void YICAAllReduceOp::update_performance_metrics() {
  performance_metrics_.cim_utilization = cim_reduction_plan_.expected_efficiency;
  performance_metrics_.spm_buffer_efficiency = spm_buffer_plan_.buffer_efficiency;
  performance_metrics_.yis_instruction_count = generated_instructions_.size();
  performance_metrics_.hierarchical_efficiency = 
    1.0f / std::max(1.0f, static_cast<float>(hierarchical_plan_.total_levels));
  performance_metrics_.memory_bandwidth_utilization = 0.85f; // 假设值
  performance_metrics_.total_reduction_operations = input_tensors[0].num_elements();
}

YICAAllReduceOp::YICAAllReduceMetrics YICAAllReduceOp::get_yica_metrics() const {
  return performance_metrics_;
}

bool YICAAllReduceOp::fingerprint(void) {
  // YICA版本的fingerprint实现
  // 这里可以复用原始的fingerprint逻辑，或者针对YICA优化
  return true;
}

YICAAllReduceOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"inplace", inplace},
              {"yica_config", yica_config}};
}

// 工厂函数实现
YICAAllReduceOp* create_yica_all_reduce_op(
  Graph *graph,
  DTensor const &input,
  bool inplace,
  const YICAAllReduceOp::YICAAllReduceConfig &config) {
  
  return new YICAAllReduceOp(graph, input, inplace, config);
}

// 辅助函数实现
namespace yica_all_reduce_utils {
  
  OptimalReductionStrategy analyze_optimal_strategy(
    const DTensor &input,
    const YICAAllReduceOp::YICAAllReduceConfig &config) {
    
    OptimalReductionStrategy strategy;
    
    size_t tensor_size = input.num_elements();
    
    // 根据张量大小选择策略
    if (tensor_size < 1024) {
      strategy.recommended_strategy = YICAAllReduceOp::ReductionStrategy::CIM_PARALLEL;
      strategy.efficiency_score = 0.9f;
    } else if (tensor_size < 1024 * 1024) {
      strategy.recommended_strategy = YICAAllReduceOp::ReductionStrategy::HIERARCHICAL_TREE;
      strategy.efficiency_score = 0.95f;
    } else {
      strategy.recommended_strategy = YICAAllReduceOp::ReductionStrategy::RING_ALLREDUCE;
      strategy.efficiency_score = 0.85f;
    }
    
    strategy.memory_requirement = tensor_size * sizeof(float) * 2;
    strategy.estimated_latency_ms = tensor_size / 1e6; // 粗略估算
    
    return strategy;
  }
  
  CommunicationPattern optimize_communication_pattern(
    int num_participants,
    YICAAllReduceOp::ReductionStrategy strategy) {
    
    CommunicationPattern pattern;
    
    switch (strategy) {
      case YICAAllReduceOp::ReductionStrategy::RING_ALLREDUCE:
        // 环形通信模式
        for (int i = 0; i < num_participants; i++) {
          pattern.communication_steps.push_back({i, (i + 1) % num_participants});
        }
        pattern.total_steps = num_participants - 1;
        break;
        
      case YICAAllReduceOp::ReductionStrategy::HIERARCHICAL_TREE:
        // 树形通信模式
        int level = 1;
        while (level < num_participants) {
          for (int i = 0; i < num_participants; i += level * 2) {
            if (i + level < num_participants) {
              pattern.communication_steps.push_back({i, i + level});
            }
          }
          level *= 2;
        }
        pattern.total_steps = pattern.communication_steps.size();
        break;
        
      default:
        pattern.total_steps = 1;
        break;
    }
    
    pattern.bandwidth_requirement = 1000.0f; // MB/s
    
    return pattern;
  }
  
} // namespace yica_all_reduce_utils

void from_json(json const &j, YICAAllReduceOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  j.at("inplace").get_to(op.inplace);
  // yica_config的反序列化需要额外实现
}

} // namespace kernel
} // namespace yirage 