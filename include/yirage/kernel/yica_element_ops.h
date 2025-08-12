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

#include "yirage/kernel/yica_kernel_base.h"

namespace yirage {
namespace kernel {

// Element-wise操作类型
enum class ElementOpType {
    // Unary ops
    EXP, LOG, SQRT, SQUARE,
    RELU, GELU, SILU, TANH, SIGMOID,
    // Binary ops
    ADD, SUB, MUL, DIV, POW,
    MAX, MIN,
    // Ternary ops
    FUSED_MUL_ADD  // a * b + c
};

// YICA Element-wise操作实现
class YICAElementOp : public YICAKernelBase {
public:
    // Unary操作
    YICAElementOp(Graph* graph,
                  const DTensor& input,
                  ElementOpType op_type,
                  const YICAHardwareConfig& config = YICAHardwareConfig{});
    
    // Binary操作
    YICAElementOp(Graph* graph,
                  const DTensor& input1,
                  const DTensor& input2,
                  ElementOpType op_type,
                  const YICAHardwareConfig& config = YICAHardwareConfig{});
    
    // Ternary操作 (FMA)
    YICAElementOp(Graph* graph,
                  const DTensor& a,
                  const DTensor& b,
                  const DTensor& c,
                  ElementOpType op_type,
                  const YICAHardwareConfig& config = YICAHardwareConfig{});
    
    ~YICAElementOp();
    
    // 实现基类接口
    bool initialize() override;
    bool execute() override;
    std::vector<yis::Instruction> generate_yis_instructions() override;
    
private:
    // 内部方法
    std::vector<yis::Instruction> generate_vectorized_instructions();
    bool execute_vectorized_operation();
    size_t calculate_vector_tiles() const;
    
    // 成员变量
    std::vector<DTensor> inputs_;
    DTensor output_;
    ElementOpType op_type_;
    size_t num_operands_;
    
    // 向量化参数
    size_t vector_tiles_;
    size_t elements_per_tile_;
};

} // namespace kernel
} // namespace yirage