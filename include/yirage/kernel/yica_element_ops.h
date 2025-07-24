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
#include <functional>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用逐元素操作算子
 * 统一处理一元和二元逐元素操作，充分利用CIM阵列的向量化能力
 */
class YICAElementOpsOp : public yirage::kernel::KNOperator {
public:
  // 一元操作构造函数
  YICAElementOpsOp(Graph *_graph,
                   DTensor const &input,
                   UnaryOpType op_type,
                   const YICAElementOpsConfig &config = {});
  
  // 二元操作构造函数
  YICAElementOpsOp(Graph *_graph,
                   DTensor const &input1,
                   DTensor const &input2,
                   BinaryOpType op_type,
                   const YICAElementOpsConfig &config = {});
  
  // 带参数的一元操作（如Clamp）
  YICAElementOpsOp(Graph *_graph,
                   DTensor const &input,
                   UnaryOpType op_type,
                   const std::vector<float> &parameters,
                   const YICAElementOpsConfig &config = {});
                   
  ~YICAElementOpsOp();
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;

  // 一元操作类型
  enum class UnaryOpType {
    // 基础数学函数
    EXP,          // 指数函数
    LOG,          // 自然对数
    SQRT,         // 平方根
    RSQRT,        // 倒数平方根
    SQUARE,       // 平方
    ABS,          // 绝对值
    SIGN,         // 符号函数
    
    // 激活函数
    RELU,         // ReLU
    GELU,         // GELU
    SILU,         // SiLU (Swish)
    TANH,         // 双曲正切
    SIGMOID,      // Sigmoid
    SOFTPLUS,     // Softplus
    
    // 三角函数
    SIN,          // 正弦
    COS,          // 余弦
    TAN,          // 正切
    
    // 其他
    NEG,          // 取负
    RECIPROCAL,   // 倒数
    CLAMP,        // 截断
    ROUND,        // 四舍五入
    FLOOR,        // 向下取整
    CEIL          // 向上取整
  };
  
  // 二元操作类型
  enum class BinaryOpType {
    // 算术运算
    ADD,          // 加法
    SUB,          // 减法
    MUL,          // 乘法
    DIV,          // 除法
    POW,          // 幂运算
    MOD,          // 取模
    
    // 比较运算
    EQ,           // 等于
    NE,           // 不等于
    LT,           // 小于
    LE,           // 小于等于
    GT,           // 大于
    GE,           // 大于等于
    
    // 逻辑运算
    AND,          // 逻辑与
    OR,           // 逻辑或
    XOR,          // 异或
    
    // 特殊运算
    MIN,          // 最小值
    MAX,          // 最大值
    ATAN2,        // 反正切2
    HYPOT         // 欧几里得距离
  };

  // YICA特定功能
  bool optimize_for_cim_vectorization(int vector_width);
  bool enable_spm_vectorized_access();
  bool use_fused_operation_chains();
  bool enable_broadcast_optimization();
  
  // 操作融合
  struct FusedOpChain {
    std::vector<UnaryOpType> unary_ops;
    std::vector<BinaryOpType> binary_ops;
    std::vector<float> parameters;
    bool fusable;
    float expected_speedup;
  };
  
  bool add_fused_operation(UnaryOpType op, const std::vector<float> &params = {});
  bool add_fused_operation(BinaryOpType op, const DTensor &operand);
  FusedOpChain get_fusion_chain() const;

  // YICA性能分析
  struct YICAElementOpsMetrics {
    float cim_vectorization_efficiency; // CIM向量化效率
    float spm_access_efficiency;        // SPM访问效率
    size_t yis_vector_instruction_count; // YIS向量指令数
    float operation_fusion_ratio;       // 操作融合比率
    float memory_bandwidth_utilization; // 内存带宽利用率
    size_t total_element_operations;    // 总元素操作数
    float broadcast_efficiency;         // 广播效率
  };
  
  YICAElementOpsMetrics get_yica_metrics() const;

public:
  // YICA Element Operations配置
  struct YICAElementOpsConfig {
    // CIM配置
    int num_cim_arrays = 16;             // CIM阵列数量
    int vector_width = 32;               // 向量宽度
    bool enable_simd_operations = true;  // 启用SIMD操作
    
    // SPM配置
    size_t spm_buffer_size = 8 * 1024 * 1024; // SPM缓冲区大小 (8MB)
    bool enable_spm_vectorized_load = true;   // 启用SPM向量化加载
    
    // 融合配置
    bool enable_operation_fusion = true; // 启用操作融合
    int max_fusion_depth = 4;            // 最大融合深度
    bool enable_broadcast_fusion = true; // 启用广播融合
    
    // 精度配置
    enum class ComputePrecision {
      FP32, FP16, BF16, INT8, MIXED
    } precision = ComputePrecision::FP16;
    
    // 优化配置
    bool enable_vectorization = true;    // 启用向量化
    bool enable_loop_unrolling = true;   // 启用循环展开
    int unroll_factor = 8;               // 展开因子
    
    // 内存优化
    bool enable_memory_coalescing = true; // 启用内存合并
    bool enable_prefetching = true;      // 启用预取
    size_t prefetch_distance = 128;     // 预取距离
    
    // 广播优化
    bool optimize_broadcast_patterns = true; // 优化广播模式
    bool enable_broadcast_caching = true;    // 启用广播缓存
  } yica_config;

private:
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_unary_instructions();
  std::vector<yica::YISInstruction> generate_binary_instructions();
  std::vector<yica::YISInstruction> generate_fused_instructions();
  std::vector<yica::YISInstruction> generate_vectorized_instructions();
  
  // CIM向量化优化
  struct CIMVectorizationPlan {
    std::vector<int> cim_array_assignment;
    int vector_width;
    int parallel_streams;
    float expected_throughput;
    size_t memory_footprint;
  };
  
  CIMVectorizationPlan plan_cim_vectorization();
  bool optimize_vector_operations();
  
  // SPM访问优化
  struct SPMAccessPlan {
    size_t input1_offset;
    size_t input2_offset;
    size_t output_offset;
    size_t broadcast_cache_offset;
    std::vector<size_t> intermediate_offsets;
    size_t total_usage;
  };
  
  SPMAccessPlan plan_spm_access();
  bool setup_vectorized_spm_access();
  
  // 操作融合实现
  bool analyze_fusion_opportunities();
  std::vector<yica::YISInstruction> implement_fused_chain();
  float calculate_fusion_benefit();
  
  // 广播优化
  struct BroadcastPattern {
    std::vector<int> broadcast_dims;
    size_t broadcast_size;
    bool cacheable;
    float cache_hit_rate;
  };
  
  BroadcastPattern analyze_broadcast_pattern();
  std::vector<yica::YISInstruction> optimize_broadcast_operations();
  
  // 特殊函数实现
  std::vector<yica::YISInstruction> implement_activation_functions();
  std::vector<yica::YISInstruction> implement_math_functions();
  std::vector<yica::YISInstruction> implement_comparison_operations();
  
  // 数值精度优化
  bool apply_mixed_precision_optimization();
  bool validate_numerical_accuracy();
  
  // 性能预测
  float predict_operation_time();
  size_t estimate_memory_requirement();
  float calculate_vectorization_efficiency();

  // 内部状态
  bool is_unary_op_;
  UnaryOpType unary_op_type_;
  BinaryOpType binary_op_type_;
  std::vector<float> operation_parameters_;
  FusedOpChain fusion_chain_;
  YICAElementOpsMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA Element Operations工厂函数
 */
YICAElementOpsOp* create_yica_unary_op(
  Graph *graph,
  DTensor const &input,
  YICAElementOpsOp::UnaryOpType op_type,
  const YICAElementOpsOp::YICAElementOpsConfig &config = {});

YICAElementOpsOp* create_yica_binary_op(
  Graph *graph,
  DTensor const &input1,
  DTensor const &input2,
  YICAElementOpsOp::BinaryOpType op_type,
  const YICAElementOpsOp::YICAElementOpsConfig &config = {});

YICAElementOpsOp* create_yica_clamp_op(
  Graph *graph,
  DTensor const &input,
  float min_val,
  float max_val,
  const YICAElementOpsOp::YICAElementOpsConfig &config = {});

/**
 * @brief YICA Element Operations辅助函数
 */
namespace yica_elementops_utils {
  
  /**
   * @brief 分析操作融合的可行性
   */
  struct FusionAnalysis {
    bool fusable;
    float expected_speedup;
    size_t memory_savings;
    std::vector<std::string> fusion_constraints;
  };
  
  FusionAnalysis analyze_operation_fusion(
    const std::vector<YICAElementOpsOp::UnaryOpType> &unary_ops,
    const std::vector<YICAElementOpsOp::BinaryOpType> &binary_ops);
  
  /**
   * @brief 生成向量化访问模式
   */
  struct VectorAccessPattern {
    size_t vector_width;
    std::vector<size_t> access_offsets;
    bool aligned_access;
    float cache_efficiency;
  };
  
  VectorAccessPattern generate_vector_access_pattern(
    const std::vector<int> &tensor_shape,
    int vector_width);
  
  /**
   * @brief 优化广播操作
   */
  std::vector<yica::YISInstruction> optimize_broadcast_operation(
    const DTensor &input,
    const std::vector<int> &broadcast_shape,
    YICAElementOpsOp::BinaryOpType op_type);
  
  /**
   * @brief 估算元素操作的计算复杂度
   */
  size_t estimate_operation_complexity(
    YICAElementOpsOp::UnaryOpType op_type,
    size_t num_elements);
  
  size_t estimate_operation_complexity(
    YICAElementOpsOp::BinaryOpType op_type,
    size_t num_elements);

} // namespace yica_elementops_utils

void from_json(json const &j, YICAElementOpsOp &op);

} // namespace kernel
} // namespace yirage 