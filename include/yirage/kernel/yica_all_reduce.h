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
#include "yirage/yica/yccl_communicator.h"
#include <vector>
#include <memory>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用All-Reduce算子
 * 利用YICA存算一体架构实现高效的分布式归约操作
 */
class YICAAllReduceOp : public yirage::kernel::KNOperator {
public:
  YICAAllReduceOp(Graph *_graph, 
                  DTensor const &input, 
                  bool inplace,
                  const YICAAllReduceConfig &config = {});
  ~YICAAllReduceOp();
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;

  // YICA特定功能
  bool optimize_for_cim_reduction(int num_cim_arrays);
  bool enable_spm_buffering(size_t buffer_size);
  bool use_yis_collective_instructions();
  
  // All-Reduce算法选择
  enum class AllReduceAlgorithm {
    RING_ALLREDUCE,          // 环形All-Reduce
    TREE_ALLREDUCE,          // 树形All-Reduce  
    BUTTERFLY_ALLREDUCE,     // 蝶形All-Reduce
    YICA_CIM_ALLREDUCE,      // YICA存算一体All-Reduce
    HIERARCHICAL_ALLREDUCE   // 层次化All-Reduce
  };
  
  bool set_algorithm(AllReduceAlgorithm algorithm);
  AllReduceAlgorithm get_optimal_algorithm(size_t data_size, int world_size);

  // 归约操作类型
  enum class ReduceOperation {
    SUM,     // 求和
    MAX,     // 最大值
    MIN,     // 最小值
    PROD,    // 乘积
    AVG      // 平均值
  };

  // YICA性能分析
  struct YICAAllReduceMetrics {
    float cim_reduction_efficiency;      // CIM归约效率
    float spm_hit_rate;                  // SPM命中率
    size_t yis_collective_instruction_count; // YIS集合通信指令数
    float communication_computation_overlap; // 通信计算重叠度
    float bandwidth_utilization;         // 带宽利用率
    size_t total_data_movement_bytes;    // 总数据移动量
  };
  
  YICAAllReduceMetrics get_yica_metrics() const;

public:
  // YICA All-Reduce配置
  struct YICAAllReduceConfig {
    // CIM配置
    int num_cim_arrays = 8;              // CIM阵列数量
    bool enable_cim_reduction = true;    // 启用CIM内归约
    
    // SPM配置
    size_t spm_buffer_size = 64 * 1024 * 1024; // SPM缓冲区大小 (64MB)
    bool enable_spm_double_buffer = true; // 启用SPM双缓冲
    
    // 通信配置
    AllReduceAlgorithm preferred_algorithm = AllReduceAlgorithm::YICA_CIM_ALLREDUCE;
    ReduceOperation reduce_op = ReduceOperation::SUM;
    bool enable_compression = true;       // 启用数据压缩
    
    // 优化配置
    bool enable_computation_overlap = true; // 启用计算通信重叠
    bool enable_hierarchical_reduction = true; // 启用层次化归约
    float compression_threshold = 1024.0f; // 压缩阈值 (KB)
    
    // 拓扑配置
    yica::DieMeshTopology mesh_topology;  // Die网格拓扑
    int local_group_size = 4;            // 本地组大小
  } yica_config;

private:
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_collective_instructions();
  std::vector<yica::YISInstruction> generate_cim_reduction_instructions();
  std::vector<yica::YISInstruction> generate_hierarchical_reduction_instructions();
  
  // CIM归约优化
  bool optimize_cim_reduction_pattern();
  std::vector<int> allocate_cim_arrays_for_reduction();
  
  // SPM缓冲区管理
  struct SPMBufferPlan {
    size_t input_buffer_offset;
    size_t intermediate_buffer_offset;
    size_t output_buffer_offset;
    size_t total_spm_usage;
    bool double_buffering_enabled;
  };
  
  SPMBufferPlan plan_spm_buffers();
  bool setup_spm_double_buffering();
  
  // 通信算法实现
  std::vector<yica::YISInstruction> implement_ring_allreduce();
  std::vector<yica::YISInstruction> implement_tree_allreduce();
  std::vector<yica::YISInstruction> implement_yica_cim_allreduce();
  std::vector<yica::YISInstruction> implement_hierarchical_allreduce();
  
  // 数据压缩
  struct CompressionResult {
    std::vector<uint8_t> compressed_data;
    float compression_ratio;
    size_t original_size;
    size_t compressed_size;
  };
  
  CompressionResult compress_tensor_data(const DTensor& tensor);
  bool decompress_tensor_data(const CompressionResult& compressed, DTensor& output);
  
  // 性能预测
  float predict_communication_time(AllReduceAlgorithm algorithm, size_t data_size);
  float predict_computation_time(ReduceOperation reduce_op, size_t data_size);
  
  // YCCL通信器集成
  std::shared_ptr<yica::YCCLCommunicator> yccl_communicator_;
  
  // 内部状态
  bool inplace_;
  YICAAllReduceMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA All-Reduce工厂函数
 */
YICAAllReduceOp* create_yica_all_reduce(
  Graph *graph,
  DTensor const &input,
  bool inplace = true,
  const YICAAllReduceOp::YICAAllReduceConfig &config = {});

/**
 * @brief YICA All-Reduce辅助函数
 */
namespace yica_allreduce_utils {
  
  /**
   * @brief 根据数据大小和拓扑选择最优算法
   */
  YICAAllReduceOp::AllReduceAlgorithm select_optimal_algorithm(
    size_t data_size, 
    int world_size,
    const yica::DieMeshTopology& topology);
  
  /**
   * @brief 估算All-Reduce性能
   */
  float estimate_allreduce_time(
    YICAAllReduceOp::AllReduceAlgorithm algorithm,
    size_t data_size,
    int world_size,
    const yica::DieMeshTopology& topology);
  
  /**
   * @brief 生成层次化归约的分组策略
   */
  std::vector<std::vector<int>> generate_hierarchical_groups(
    int world_size,
    const yica::DieMeshTopology& topology);
  
  /**
   * @brief 优化CIM阵列的归约映射
   */
  std::map<int, std::vector<int>> optimize_cim_reduction_mapping(
    const DTensor& tensor,
    int num_cim_arrays);

} // namespace yica_allreduce_utils

void from_json(json const &j, YICAAllReduceOp &op);

} // namespace kernel
} // namespace yirage 