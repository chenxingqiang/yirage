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

// YICA RMSNorm实现
class YICARMSNormOp : public YICAKernelBase {
public:
    YICARMSNormOp(Graph* graph,
                  const DTensor& input,
                  const DTensor& weight,
                  float epsilon = 1e-6f,
                  const YICAHardwareConfig& config = YICAHardwareConfig{});
    ~YICARMSNormOp();
    
    // 实现基类接口
    bool initialize() override;
    bool execute() override;
    std::vector<yis::Instruction> generate_yis_instructions() override;
    
    // RMSNorm特定方法
    void set_epsilon(float eps) { epsilon_ = eps; }
    float get_epsilon() const { return epsilon_; }
    
private:
    // 内部方法
    std::vector<yis::Instruction> generate_reduction_instructions();
    std::vector<yis::Instruction> generate_normalization_instructions();
    bool execute_rms_normalization();
    
    // 成员变量
    DTensor input_;
    DTensor weight_;
    DTensor output_;
    float epsilon_;
    
    // 中间结果
    DTensor rms_values_;
};

} // namespace kernel
} // namespace yirage