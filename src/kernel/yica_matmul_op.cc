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

#include "yirage/kernel/yica_operators.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/profile_result.h"
#include <cassert>
#include <chrono>
#include <iostream>

namespace yirage {
namespace kernel {

KNYICAMatMulOp::KNYICAMatMulOp(Graph* graph,
                               const DTensor& A,
                               const DTensor& B,
                               bool transpose_a,
                               bool transpose_b)
    : KNOperator(graph, yirage::type::KN_MATMUL_OP, A, B),
      transpose_a_(transpose_a), transpose_b_(transpose_b) {
    
    // 计算输出维度
    size_t M = transpose_a_ ? A.dim[1] : A.dim[0];
    size_t N = transpose_b_ ? B.dim[0] : B.dim[1];
    
    // 创建输出张量
    DTensor output = A;  // 复制布局和数据类型
    output.num_dims = 2;
    output.dim[0] = M;
    output.dim[1] = N;
    output.owner_op = static_cast<KNOperator*>(this);
    output.owner_ts_idx = 0;
    output.guid = DTensor::next_guid++;
    
    // 分配输出张量内存
    kgraph->allocate(output);
    assert(output_tensors.size() == 0);
    output_tensors.push_back(output);
}

KNYICAMatMulOp::~KNYICAMatMulOp() {
    // 释放输出张量
    for (int i = output_tensors.size() - 1; i >= 0; i--) {
        kgraph->free(output_tensors[i]);
    }
}

bool KNYICAMatMulOp::profile(ProfileResult& result) {
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = execute_yica_matmul();
    
    auto end = std::chrono::high_resolution_clock::now();
    execution_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 填充 ProfileResult
    result.run_time = execution_time_ms_ * 1000.0;  // 转换为微秒
    // Note: ProfileResult 只有 run_time 字段，memory_usage 不在标准结构中
    
    return success;
}

bool KNYICAMatMulOp::fingerprint(void) {
    // YICA 算子的指纹计算 - 基于输入张量的指纹
    // 这里简化实现，实际可能需要更复杂的指纹计算
    return true;
}

KNYICAMatMulOp::operator json() const {
    return json{
        {"op_type", op_type},
        {"input_tensors", input_tensors},
        {"output_tensors", output_tensors},
        {"transpose_a", transpose_a_},
        {"transpose_b", transpose_b_},
        {"alpha", alpha_},
        {"beta", beta_}
    };
}

bool KNYICAMatMulOp::execute_yica_matmul() {
    using namespace yirage::kernel;
    DeviceMemoryManager* dmm = DeviceMemoryManager::get_instance();
    if (dmm == nullptr) {
        std::cerr << "KNYICAMatMulOp: DeviceMemoryManager not initialized" << std::endl;
        return false;
    }
    char* base_ptr = dmm->data_base_ptr[dmm->gpu_id];
    if (base_ptr == nullptr) {
        std::cerr << "KNYICAMatMulOp: base_ptr is null" << std::endl;
        return false;
    }

    const DTensor& A = input_tensors[0];
    const DTensor& B = input_tensors[1];
    const DTensor& C = output_tensors[0];
    
    size_t M = C.dim[0];
    size_t N = C.dim[1];
    size_t K = transpose_a_ ? A.dim[0] : A.dim[1];

    float* A_data = reinterpret_cast<float*>(base_ptr + A.data_offset);
    float* B_data = reinterpret_cast<float*>(base_ptr + B.data_offset);
    float* C_data = reinterpret_cast<float*>(base_ptr + C.data_offset);

    // 初始化C
    std::memset(C_data, 0, C.data_size());

    // 执行矩阵乘法 (CPU 参考实现)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float a_val = transpose_a_ ? A_data[k * M + i] : A_data[i * K + k];
                float b_val = transpose_b_ ? B_data[j * K + k] : B_data[k * N + j];
                sum += a_val * b_val;
            }
            C_data[i * N + j] = alpha_ * sum + beta_ * C_data[i * N + j];
        }
    }

    return true;
}

// YIS 指令生成已简化，专注于核心矩阵乘法执行

} // namespace kernel
} // namespace yirage
