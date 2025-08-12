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
#include <vector>

namespace yirage {
namespace kernel {

// AllReduce操作类型
enum class AllReduceOp {
    SUM,
    MEAN, 
    MAX,
    MIN,
    PROD
};

// YICA AllReduce实现
class YICAAllReduceOp : public YICAKernelBase {
public:
    YICAAllReduceOp(Graph* graph,
                    const DTensor& input,
                    AllReduceOp op_type = AllReduceOp::SUM,
                    bool inplace = false,
                    const YICAHardwareConfig& config = YICAHardwareConfig{});
    ~YICAAllReduceOp();
    
    // 实现基类接口
    bool initialize() override;
    bool execute() override;
    std::vector<yis::Instruction> generate_yis_instructions() override;
    
    // AllReduce特定方法
    void set_reduction_op(AllReduceOp op) { reduction_op_ = op; }
    AllReduceOp get_reduction_op() const { return reduction_op_; }
    
private:
    // 内部方法
    std::vector<yis::Instruction> generate_cim_reduction_instructions();
    std::vector<yis::Instruction> generate_hierarchical_reduction_instructions();
    bool perform_reduction();
    bool execute_all_reduce_operation();
    
    // 成员变量
    DTensor input_tensor_;
    DTensor output_tensor_;
    AllReduceOp reduction_op_;
    bool inplace_;
    size_t total_elements_;
    
    // CIM映射
    std::vector<std::pair<size_t, size_t>> cim_data_ranges_;
    
    // 内部验证方法
    bool validate_reduction_parameters() const;
};

} // namespace kernel
} // namespace yirage