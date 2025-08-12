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

// YICA矩阵乘法实现
class YICAMatMulOp : public YICAKernelBase {
public:
    YICAMatMulOp(Graph* graph,
                 const DTensor& A,
                 const DTensor& B,
                 bool transpose_a = false,
                 bool transpose_b = false,
                 const YICAHardwareConfig& config = YICAHardwareConfig{});
    ~YICAMatMulOp();
    
    // 实现基类接口
    bool initialize() override;
    bool execute() override;
    std::vector<yis::Instruction> generate_yis_instructions() override;
    
    // MatMul特定方法
    void set_alpha(float alpha) { alpha_ = alpha; }
    void set_beta(float beta) { beta_ = beta; }
    
private:
    // 内部方法
    void calculate_tiling_strategy();
    std::vector<yis::Instruction> generate_tiled_mma_instructions();
    bool execute_cim_matmul();
    
    // 成员变量
    DTensor A_, B_, C_;
    bool transpose_a_, transpose_b_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    
    // 矩阵维度
    int M_, N_, K_;
    
    // Tiling参数
    struct TilingStrategy {
        size_t tile_m, tile_n, tile_k;
        size_t num_tiles;
        bool use_double_buffering;
    } tiling_strategy_;
};

} // namespace kernel
} // namespace yirage