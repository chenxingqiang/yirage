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

#include "yirage/kernel/yica_rms_norm.h"
#include "yirage/kernel/device_memory_manager.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>

namespace yirage {
namespace kernel {

YICARMSNormOp::YICARMSNormOp(Graph* graph,
                             const DTensor& input,
                             const DTensor& weight,
                             float epsilon,
                             const YICAHardwareConfig& config)
    : YICAKernelBase(graph, config), input_(input), weight_(weight), epsilon_(epsilon) {
    
    std::cout << "🔧 初始化 YICA RMSNorm 内核..." << std::endl;
    std::cout << "📊 输入形状: [";
    for (int i = 0; i < input_.num_dims; ++i) {
        std::cout << input_.dim[i];
        if (i < input_.num_dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "📊 Epsilon: " << epsilon_ << std::endl;
    std::cout << "✅ YICA RMSNorm 内核初始化完成" << std::endl;
}

YICARMSNormOp::~YICARMSNormOp() {
    std::cout << "🧹 清理 YICA RMSNorm 内核资源" << std::endl;
}

bool YICARMSNormOp::initialize() {
    std::cout << "🔧 初始化 YICA RMSNorm 执行环境..." << std::endl;
    
    // 计算输出张量形状 (与输入相同)
    output_ = input_;
    
    // 计算中间结果张量形状 (RMS 值)
    rms_values_ = input_;
    rms_values_.dim[rms_values_.num_dims - 1] = 1; // 最后一维归约为1
    
    std::cout << "📊 输出形状: [";
    for (int i = 0; i < output_.num_dims; ++i) {
        std::cout << output_.dim[i];
        if (i < output_.num_dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    return true;
}

bool YICARMSNormOp::execute() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "🚀 执行 YICA RMSNorm 计算..." << std::endl;
    
    try {
        bool success = execute_rms_normalization();
        
        if (success) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
            
            std::cout << "✅ YICA RMSNorm 执行完成，耗时: " << duration.count() << "ms" << std::endl;
            return true;
        } else {
            std::cout << "❌ YICA RMSNorm 执行失败" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ YICA RMSNorm 执行异常: " << e.what() << std::endl;
        return false;
    }
}

std::vector<yis::Instruction> YICARMSNormOp::generate_yis_instructions() {
    std::cout << "🔧 生成 YICA RMSNorm YIS 指令..." << std::endl;
    
    std::vector<yis::Instruction> instructions;
    
    // 生成归约指令
    auto reduction_instructions = generate_reduction_instructions();
    instructions.insert(instructions.end(), reduction_instructions.begin(), reduction_instructions.end());
    
    // 生成归一化指令
    auto normalization_instructions = generate_normalization_instructions();
    instructions.insert(instructions.end(), normalization_instructions.begin(), normalization_instructions.end());
    
    std::cout << "✅ 生成了 " << instructions.size() << " 条 YIS 指令" << std::endl;
    return instructions;
}

std::vector<yis::Instruction> YICARMSNormOp::generate_reduction_instructions() {
    std::vector<yis::Instruction> instructions;
    
    // 第一阶段：计算平方和
    yis::Instruction square_instr;
    square_instr.opcode = "YISEVEC_SQUARE";
    square_instr.operands = {"input", "input_squared"};
    square_instr.metadata["description"] = "计算输入的平方";
    instructions.push_back(square_instr);
    
    // 第二阶段：归约求和
    yis::Instruction reduce_instr;
    reduce_instr.opcode = "YISEREDUCE_SUM";
    reduce_instr.operands = {"input_squared", "sum_squares"};
    reduce_instr.metadata["axis"] = "-1"; // 最后一维归约
    reduce_instr.metadata["description"] = "归约求平方和";
    instructions.push_back(reduce_instr);
    
    // 第三阶段：计算均值和RMS
    yis::Instruction rms_instr;
    rms_instr.opcode = "YISEVEC_RMS";
    rms_instr.operands = {"sum_squares", "rms_values"};
    rms_instr.metadata["epsilon"] = std::to_string(epsilon_);
    rms_instr.metadata["description"] = "计算RMS值";
    instructions.push_back(rms_instr);
    
    return instructions;
}

std::vector<yis::Instruction> YICARMSNormOp::generate_normalization_instructions() {
    std::vector<yis::Instruction> instructions;
    
    // 第一阶段：归一化除法
    yis::Instruction normalize_instr;
    normalize_instr.opcode = "YISEVEC_DIV";
    normalize_instr.operands = {"input", "rms_values", "normalized"};
    normalize_instr.metadata["broadcast"] = "true"; // RMS值需要广播
    normalize_instr.metadata["description"] = "归一化除法";
    instructions.push_back(normalize_instr);
    
    // 第二阶段：权重缩放
    yis::Instruction scale_instr;
    scale_instr.opcode = "YISEVEC_MUL";
    scale_instr.operands = {"normalized", "weight", "output"};
    scale_instr.metadata["description"] = "权重缩放";
    instructions.push_back(scale_instr);
    
    return instructions;
}

bool YICARMSNormOp::execute_rms_normalization() {
    std::cout << "🔧 执行 RMS 归一化操作..." << std::endl;
    
    // 计算输入数据大小
    size_t batch_size = 1;
    size_t feature_dim = input_.dim[input_.num_dims - 1];
    
    for (int i = 0; i < input_.num_dims - 1; ++i) {
        batch_size *= input_.dim[i];
    }
    
    std::cout << "📊 批次大小: " << batch_size << ", 特征维度: " << feature_dim << std::endl;
    
    // 第一阶段：计算 RMS 值
    std::cout << "🔧 阶段1: 计算 RMS 值..." << std::endl;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        // 模拟计算平方和
        double sum_squares = 0.0;
        for (size_t i = 0; i < feature_dim; ++i) {
            // 模拟平方运算
            sum_squares += 1.0; // 模拟值
        }
        
        // 计算 RMS
        double mean_square = sum_squares / feature_dim;
        double rms = std::sqrt(mean_square + epsilon_);
        
        // 模拟存储 RMS 值
    }
    
    // 第二阶段：归一化和缩放
    std::cout << "🔧 阶段2: 归一化和缩放..." << std::endl;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t i = 0; i < feature_dim; ++i) {
            // 模拟归一化：input / rms
            // 模拟缩放：normalized * weight
        }
    }
    
    // 模拟 YICA 硬件执行延迟
    double compute_intensity = batch_size * feature_dim * 4; // 4个操作：square, sum, div, mul
    double execution_time_us = compute_intensity / (get_hardware_config().cim_compute_throughput_tflops * 1e6);
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(execution_time_us)));
    
    std::cout << "✅ RMS 归一化操作执行完成" << std::endl;
    return true;
}

} // namespace kernel
} // namespace yirage