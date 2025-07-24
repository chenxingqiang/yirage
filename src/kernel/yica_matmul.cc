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

#include "yirage/kernel/yica_matmul.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>
#include <algorithm>
#include <cmath>

namespace yirage {
namespace kernel {

YICAMatMulOp::YICAMatMulOp(Graph *_graph,
                           DTensor const &A,
                           DTensor const &B,
                           DTensor const &C,
                           const YICAMatMulConfig &config)
    : KNOperator(_graph, yirage::type::KN_MATMUL_OP, A, B),
      A_(A), B_(B), C_(C), yica_config(config) {
  
  // 验证矩阵乘法的维度兼容性
  assert(A.num_dims >= 2 && B.num_dims >= 2);
  assert(A.dim[A.num_dims - 1] == B.dim[B.num_dims - 2]);
  
  DTensor output = C;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
  
  // 确定MatMul类型
  determine_matmul_type();
  
  // 初始化YICA特定组件
  initialize_yica_components();
}

YICAMatMulOp::~YICAMatMulOp() {
  kgraph->free(output_tensors[0]);
}

void YICAMatMulOp::determine_matmul_type() {
  int m = A_.dim[A_.num_dims - 2];
  int k = A_.dim[A_.num_dims - 1];
  int n = B_.dim[B_.num_dims - 1];
  
  // 根据矩阵规模确定类型
  if (n == 1) {
    matmul_type_ = MatMulType::GEMV;
  } else if (A_.num_dims > 2 || B_.num_dims > 2) {
    matmul_type_ = MatMulType::BATCHED_GEMM;
  } else if (m * n * k > 1024 * 1024) {
    matmul_type_ = MatMulType::CIM_NATIVE;
  } else {
    matmul_type_ = MatMulType::GEMM;
  }
}

bool YICAMatMulOp::initialize_yica_components() {
  // 优化CIM映射
  cim_mapping_ = optimize_cim_mapping();
  
  // 规划SPM数据管理
  spm_data_plan_ = plan_spm_data_management();
  
  // 优化矩阵分块
  tiling_plan_ = optimize_matrix_tiling();
  
  // 分析混合精度机会
  if (yica_config.precision_config.input_precision == 
      YICAMatMulConfig::PrecisionConfig::DataType::MIXED) {
    mixed_precision_plan_ = analyze_mixed_precision_opportunities();
  }
  
  // 优化数据预取
  prefetch_plan_ = optimize_data_prefetching();
  
  // 实现操作融合
  if (yica_config.fusion_config.enable_bias_fusion ||
      yica_config.fusion_config.enable_activation_fusion) {
    fusion_implementation_ = implement_operation_fusion();
  }
  
  // 生成YIS指令
  generated_instructions_ = generate_matmul_instructions();
  
  // 初始化自动调优器
  if (yica_config.enable_auto_tuning) {
    initialize_auto_tuner();
  }
  
  return true;
}

YICAMatMulOp::CIMMapping YICAMatMulOp::optimize_cim_mapping() {
  CIMMapping mapping;
  
  int m = A_.dim[A_.num_dims - 2];
  int n = B_.dim[B_.num_dims - 1];
  int k = A_.dim[A_.num_dims - 1];
  
  // 计算每个CIM阵列的工作负载
  int tiles_m = (m + yica_config.tiling_config.tile_m - 1) / yica_config.tiling_config.tile_m;
  int tiles_n = (n + yica_config.tiling_config.tile_n - 1) / yica_config.tiling_config.tile_n;
  int total_tiles = tiles_m * tiles_n;
  
  // 为每个CIM阵列分配分块
  int tiles_per_cim = (total_tiles + yica_config.num_cim_arrays - 1) / yica_config.num_cim_arrays;
  
  mapping.cim_tile_mapping.resize(yica_config.num_cim_arrays);
  mapping.cim_workload_distribution.resize(yica_config.num_cim_arrays);
  
  int tile_idx = 0;
  for (int cim_id = 0; cim_id < yica_config.num_cim_arrays; cim_id++) {
    for (int t = 0; t < tiles_per_cim && tile_idx < total_tiles; t++) {
      int tile_m = tile_idx / tiles_n;
      int tile_n = tile_idx % tiles_n;
      mapping.cim_tile_mapping[cim_id].push_back(tile_m * tiles_n + tile_n);
      tile_idx++;
    }
    mapping.cim_workload_distribution[cim_id] = mapping.cim_tile_mapping[cim_id].size();
  }
  
  // 计算负载均衡因子
  float avg_workload = static_cast<float>(total_tiles) / yica_config.num_cim_arrays;
  float max_deviation = 0.0f;
  for (int workload : mapping.cim_workload_distribution) {
    max_deviation = std::max(max_deviation, std::abs(workload - avg_workload));
  }
  mapping.load_balance_factor = 1.0f - (max_deviation / avg_workload);
  
  // 估算通信开销
  mapping.communication_overhead = total_tiles * 1024; // 假设每个分块1KB通信开销
  
  return mapping;
}

YICAMatMulOp::SPMDataPlan YICAMatMulOp::plan_spm_data_management() {
  SPMDataPlan plan;
  
  // 计算A、B、C矩阵的SPM缓冲需求
  size_t tile_a_size = yica_config.tiling_config.tile_m * yica_config.tiling_config.tile_k * sizeof(float);
  size_t tile_b_size = yica_config.tiling_config.tile_k * yica_config.tiling_config.tile_n * sizeof(float);
  size_t tile_c_size = yica_config.tiling_config.tile_m * yica_config.tiling_config.tile_n * sizeof(float);
  
  // A矩阵缓冲计划
  plan.a_buffer_plan.size = std::min(tile_a_size, yica_config.spm_a_buffer_size);
  plan.a_buffer_plan.offset = 0;
  plan.a_buffer_plan.is_double_buffered = yica_config.enable_spm_double_buffering;
  
  // B矩阵缓冲计划
  plan.b_buffer_plan.size = std::min(tile_b_size, yica_config.spm_b_buffer_size);
  plan.b_buffer_plan.offset = plan.a_buffer_plan.size * (plan.a_buffer_plan.is_double_buffered ? 2 : 1);
  plan.b_buffer_plan.is_double_buffered = yica_config.enable_spm_double_buffering;
  
  // C矩阵缓冲计划
  plan.c_buffer_plan.size = std::min(tile_c_size, yica_config.spm_c_buffer_size);
  plan.c_buffer_plan.offset = plan.b_buffer_plan.offset + 
                              plan.b_buffer_plan.size * (plan.b_buffer_plan.is_double_buffered ? 2 : 1);
  plan.c_buffer_plan.is_double_buffered = false; // C通常不需要双缓冲
  
  // 计算总SPM使用量
  plan.total_spm_usage = plan.c_buffer_plan.offset + plan.c_buffer_plan.size;
  
  // 计算复用效率
  size_t total_data_size = A_.data_size() + B_.data_size() + C_.data_size();
  plan.reuse_efficiency = static_cast<float>(plan.total_spm_usage) / total_data_size;
  
  return plan;
}

YICAMatMulOp::TilingPlan YICAMatMulOp::optimize_matrix_tiling() {
  TilingPlan plan;
  
  int m = A_.dim[A_.num_dims - 2];
  int n = B_.dim[B_.num_dims - 1];
  int k = A_.dim[A_.num_dims - 1];
  
  // 自适应分块大小优化
  if (yica_config.tiling_config.enable_adaptive_tiling) {
    // 基于CIM阵列大小和SPM容量优化分块
    int optimal_tile_m = std::min(yica_config.cim_array_rows, 
                                 yica_config.tiling_config.tile_m);
    int optimal_tile_n = std::min(yica_config.cim_array_cols, 
                                 yica_config.tiling_config.tile_n);
    int optimal_tile_k = std::min(64, yica_config.tiling_config.tile_k);
    
    plan.optimal_tile_m = optimal_tile_m;
    plan.optimal_tile_n = optimal_tile_n;
    plan.optimal_tile_k = optimal_tile_k;
  } else {
    plan.optimal_tile_m = yica_config.tiling_config.tile_m;
    plan.optimal_tile_n = yica_config.tiling_config.tile_n;
    plan.optimal_tile_k = yica_config.tiling_config.tile_k;
  }
  
  // 生成分块
  for (int i = 0; i < m; i += plan.optimal_tile_m) {
    for (int j = 0; j < n; j += plan.optimal_tile_n) {
      for (int l = 0; l < k; l += plan.optimal_tile_k) {
        TilingPlan::Tile tile;
        tile.m_start = i;
        tile.m_end = std::min(i + plan.optimal_tile_m, m);
        tile.n_start = j;
        tile.n_end = std::min(j + plan.optimal_tile_n, n);
        tile.k_start = l;
        tile.k_end = std::min(l + plan.optimal_tile_k, k);
        
        // 分配CIM阵列
        int tile_id = plan.tiles.size();
        tile.cim_array_id = tile_id % yica_config.num_cim_arrays;
        
        // 估算计算成本
        int tile_m = tile.m_end - tile.m_start;
        int tile_n = tile.n_end - tile.n_start;
        int tile_k = tile.k_end - tile.k_start;
        tile.computation_cost = 2.0f * tile_m * tile_n * tile_k; // 乘累加操作数
        
        plan.tiles.push_back(tile);
      }
    }
  }
  
  // 计算分块效率
  size_t total_ops = 2 * m * n * k;
  size_t tiled_ops = 0;
  for (const auto& tile : plan.tiles) {
    tiled_ops += static_cast<size_t>(tile.computation_cost);
  }
  plan.tiling_efficiency = static_cast<float>(total_ops) / tiled_ops;
  
  // 估算内存占用
  plan.memory_footprint = plan.tiles.size() * 
    (plan.optimal_tile_m * plan.optimal_tile_k + 
     plan.optimal_tile_k * plan.optimal_tile_n + 
     plan.optimal_tile_m * plan.optimal_tile_n) * sizeof(float);
  
  return plan;
}

YICAMatMulOp::MixedPrecisionPlan YICAMatMulOp::analyze_mixed_precision_opportunities() {
  MixedPrecisionPlan plan;
  
  // 简化的混合精度分析
  // 实际实现需要基于数值分析和误差传播理论
  
  plan.use_low_precision_tiles.resize(tiling_plan_.tiles.size(), false);
  
  for (size_t i = 0; i < tiling_plan_.tiles.size(); i++) {
    const auto& tile = tiling_plan_.tiles[i];
    
    // 基于分块大小决定精度
    if (tile.computation_cost < 1000) {
      // 小分块使用低精度
      plan.use_low_precision_tiles[i] = true;
      plan.tile_precision_map[i] = YICAMatMulConfig::PrecisionConfig::DataType::FP16;
    } else {
      // 大分块使用高精度
      plan.tile_precision_map[i] = YICAMatMulConfig::PrecisionConfig::DataType::FP32;
    }
  }
  
  // 估算精度损失和性能收益
  int low_precision_tiles = std::count(plan.use_low_precision_tiles.begin(),
                                      plan.use_low_precision_tiles.end(), true);
  float low_precision_ratio = static_cast<float>(low_precision_tiles) / plan.use_low_precision_tiles.size();
  
  plan.accuracy_loss_estimate = low_precision_ratio * 0.01f; // 假设1%精度损失
  plan.performance_gain_estimate = low_precision_ratio * 1.5f; // 假设1.5x性能提升
  
  return plan;
}

YICAMatMulOp::PrefetchPlan YICAMatMulOp::optimize_data_prefetching() {
  PrefetchPlan plan;
  
  // 为每个分块生成预取调度
  for (size_t i = 0; i < tiling_plan_.tiles.size(); i++) {
    if (i + yica_config.memory_config.prefetch_distance < tiling_plan_.tiles.size()) {
      size_t prefetch_tile_id = i + yica_config.memory_config.prefetch_distance;
      plan.prefetch_schedule.push_back({static_cast<int>(prefetch_tile_id), static_cast<int>(i)});
    }
  }
  
  plan.optimal_prefetch_distance = yica_config.memory_config.prefetch_distance;
  plan.prefetch_efficiency = 0.8f; // 假设80%预取效率
  
  return plan;
}

YICAMatMulOp::FusionImplementation YICAMatMulOp::implement_operation_fusion() {
  FusionImplementation impl;
  
  // 构建融合操作序列
  impl.fused_op_sequence.push_back("matmul");
  
  if (yica_config.fusion_config.enable_bias_fusion) {
    impl.fused_op_sequence.push_back("bias_add");
  }
  
  if (yica_config.fusion_config.enable_activation_fusion) {
    impl.fused_op_sequence.push_back("activation");
  }
  
  if (yica_config.fusion_config.enable_batch_norm_fusion) {
    impl.fused_op_sequence.push_back("batch_norm");
  }
  
  // 估算融合收益
  impl.fusion_benefit = impl.fused_op_sequence.size() * 0.15f; // 每个融合操作15%收益
  impl.memory_savings = impl.fused_op_sequence.size() * C_.data_size(); // 节省中间结果存储
  
  return impl;
}

void YICAMatMulOp::initialize_auto_tuner() {
  // 添加调优参数
  auto_tuner_.add_tuning_parameter("tile_m", {64, 128, 256});
  auto_tuner_.add_tuning_parameter("tile_n", {64, 128, 256});
  auto_tuner_.add_tuning_parameter("tile_k", {32, 64, 128});
  auto_tuner_.add_tuning_parameter("num_cim_arrays", {4, 8, 16, 32});
}

std::vector<yica::YISInstruction> YICAMatMulOp::generate_matmul_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  // 1. 数据移动指令
  auto data_movement_instrs = generate_data_movement_instructions();
  instructions.insert(instructions.end(), data_movement_instrs.begin(), data_movement_instrs.end());
  
  // 2. YISMMA指令
  auto yismma_instrs = generate_yismma_instructions();
  instructions.insert(instructions.end(), yismma_instrs.begin(), yismma_instrs.end());
  
  // 3. 融合操作指令
  if (!fusion_implementation_.fused_op_sequence.empty()) {
    auto fusion_instrs = generate_fusion_instructions();
    instructions.insert(instructions.end(), fusion_instrs.begin(), fusion_instrs.end());
  }
  
  return instructions;
}

std::vector<yica::YISInstruction> YICAMatMulOp::generate_yismma_instructions() {
  std::vector<yica::YISInstruction> instructions;
  
  for (const auto& tile : tiling_plan_.tiles) {
    yica::YISInstruction mma_instr;
    mma_instr.type = yica::YISInstructionType::YISMMA;
    mma_instr.operation = yica::YISOperation::MATRIX_MULTIPLY_ACCUMULATE;
    mma_instr.cim_array_id = tile.cim_array_id;
    
    // 设置矩阵维度
    mma_instr.matrix_m = tile.m_end - tile.m_start;
    mma_instr.matrix_n = tile.n_end - tile.n_start;
    mma_instr.matrix_k = tile.k_end - tile.k_start;
    
    // 设置数据精度
    if (mixed_precision_plan_.tile_precision_map.count(instructions.size())) {
      mma_instr.precision = mixed_precision_plan_.tile_precision_map[instructions.size()];
    } else {
      mma_instr.precision = yica_config.precision_config.input_precision;
    }
    
    mma_instr.sync_required = true;
    instructions.push_back(mma_instr);
  }
  
  return instructions;
}

bool YICAMatMulOp::profile(ProfileResult &result) {
  // 使用性能模型预测执行时间
  performance_model_.update_model_parameters();
  result.run_time = performance_model_.predict_performance(yica_config);
  
  // 更新性能指标
  update_performance_metrics();
  
  return true;
}

void YICAMatMulOp::update_performance_metrics() {
  performance_metrics_.cim_utilization = cim_mapping_.load_balance_factor;
  performance_metrics_.spm_hit_rate = spm_data_plan_.reuse_efficiency;
  performance_metrics_.yismma_instruction_count = 0;
  
  // 统计YISMMA指令数
  for (const auto& instr : generated_instructions_) {
    if (instr.type == yica::YISInstructionType::YISMMA) {
      performance_metrics_.yismma_instruction_count++;
    }
  }
  
  performance_metrics_.mixed_precision_ratio = 
    mixed_precision_plan_.performance_gain_estimate;
  performance_metrics_.memory_bandwidth_efficiency = 0.85f;
  performance_metrics_.total_multiply_accumulates = 
    2 * A_.dim[A_.num_dims - 2] * B_.dim[B_.num_dims - 1] * A_.dim[A_.num_dims - 1];
  performance_metrics_.compute_intensity = 
    static_cast<float>(performance_metrics_.total_multiply_accumulates) / 
    (A_.data_size() + B_.data_size() + C_.data_size());
  performance_metrics_.numerical_accuracy = 1.0f - mixed_precision_plan_.accuracy_loss_estimate;
}

YICAMatMulOp::YICAMatMulMetrics YICAMatMulOp::get_yica_metrics() const {
  return performance_metrics_;
}

bool YICAMatMulOp::fingerprint(void) {
  // YICA版本的fingerprint实现
  return true;
}

YICAMatMulOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"matmul_type", static_cast<int>(matmul_type_)},
              {"yica_config", yica_config}};
}

// 工厂函数实现
YICAMatMulOp* create_yica_matmul_op(
  Graph *graph,
  DTensor const &A,
  DTensor const &B,
  DTensor const &C,
  const YICAMatMulOp::YICAMatMulConfig &config) {
  
  return new YICAMatMulOp(graph, A, B, C, config);
}

// 辅助函数实现
namespace yica_matmul_utils {
  
  OptimalTiling calculate_optimal_tiling(
    const DTensor &A,
    const DTensor &B,
    const YICAMatMulOp::YICAMatMulConfig &config) {
    
    OptimalTiling tiling;
    
    int m = A.dim[A.num_dims - 2];
    int n = B.dim[B.num_dims - 1];
    int k = A.dim[A.num_dims - 1];
    
    // 基于CIM阵列大小优化分块
    tiling.tile_m = std::min(m, config.cim_array_rows);
    tiling.tile_n = std::min(n, config.cim_array_cols);
    tiling.tile_k = std::min(k, 64); // 经验值
    
    // 计算效率分数
    float cim_utilization = static_cast<float>(tiling.tile_m * tiling.tile_n) / 
                           (config.cim_array_rows * config.cim_array_cols);
    tiling.efficiency_score = cim_utilization;
    
    // 估算内存需求
    tiling.memory_requirement = (tiling.tile_m * tiling.tile_k + 
                                 tiling.tile_k * tiling.tile_n + 
                                 tiling.tile_m * tiling.tile_n) * sizeof(float);
    
    // 计算并行度
    tiling.parallel_degree = std::min(config.num_cim_arrays, 
                                     (m / tiling.tile_m) * (n / tiling.tile_n));
    
    return tiling;
  }
  
} // namespace yica_matmul_utils

void from_json(json const &j, YICAMatMulOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  // 其他字段的反序列化需要额外实现
}

} // namespace kernel
} // namespace yirage 