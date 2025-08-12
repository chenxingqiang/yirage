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
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>

namespace yirage {
namespace kernel {

YICAAllReduceOp::YICAAllReduceOp(Graph* graph,
                                 const DTensor& input,
                                 AllReduceOp op_type,
                                 bool inplace,
                                 const YICAHardwareConfig& config)
    : YICAKernelBase(graph, config), input_tensor_(input), reduction_op_(op_type), inplace_(inplace) {
    
    std::cout << "🔧 初始化 YICA AllReduce 内核..." << std::endl;
    std::cout << "📊 输入形状: [";
    for (int i = 0; i < input_tensor_.num_dims; ++i) {
        std::cout << input_tensor_.dim[i];
        if (i < input_tensor_.num_dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "📊 归约操作: " << static_cast<int>(reduction_op_) << std::endl;
    std::cout << "📊 就地操作: " << (inplace_ ? "是" : "否") << std::endl;
    std::cout << "✅ YICA AllReduce 内核初始化完成" << std::endl;
}

YICAAllReduceOp::~YICAAllReduceOp() {
    std::cout << "🧹 清理 YICA AllReduce 内核资源" << std::endl;
}

bool YICAAllReduceOp::initialize() {
    std::cout << "🔧 初始化 YICA AllReduce 执行环境..." << std::endl;
    
    // 设置输出张量
    if (inplace_) {
        output_tensor_ = input_tensor_;
    } else {
        output_tensor_ = input_tensor_;
    }
    
    // 计算归约维度
    total_elements_ = input_tensor_.num_elements();
    
    std::cout << "📊 总元素数量: " << total_elements_ << std::endl;
    std::cout << "📊 输出张量设置完成" << std::endl;
    
    return validate_reduction_parameters();
}

bool YICAAllReduceOp::execute() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "🚀 执行 YICA AllReduce 计算..." << std::endl;
    
    try {
        bool success = execute_all_reduce_operation();
        
        if (success) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
            execution_time_ms_ = duration.count();
            
            std::cout << "✅ YICA AllReduce 执行完成，耗时: " << duration.count() << "ms" << std::endl;
            return true;
        } else {
            std::cout << "❌ YICA AllReduce 执行失败" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ YICA AllReduce 执行异常: " << e.what() << std::endl;
        return false;
    }
}

std::vector<yis::Instruction> YICAAllReduceOp::generate_yis_instructions() {
    std::cout << "🔧 生成 YICA AllReduce YIS 指令..." << std::endl;
    
    std::vector<yis::Instruction> instructions;
    
    // 生成分层归约指令
    auto tree_instructions = generate_hierarchical_reduction_instructions();
    instructions.insert(instructions.end(), tree_instructions.begin(), tree_instructions.end());
    
    // 生成同步指令
    yis::Instruction sync_instr;
    sync_instr.type = yis::InstructionType::SYNC_BAR;
    sync_instr.sync_required = true;
    instructions.push_back(sync_instr);
    
    std::cout << "✅ 生成了 " << instructions.size() << " 条 YIS 指令" << std::endl;
    return instructions;
}

std::vector<yis::Instruction> YICAAllReduceOp::generate_hierarchical_reduction_instructions() {
    std::vector<yis::Instruction> instructions;
    
    // 计算归约层级数
    int num_levels = static_cast<int>(std::log2(get_hardware_config().num_cim_arrays));
    
    std::cout << "🔧 生成 " << num_levels << " 层树状归约指令..." << std::endl;
    
    // 为每一层生成归约指令
    for (int level = 0; level < num_levels; ++level) {
        int stride = 1 << level; // 2^level
        int num_ops = get_hardware_config().num_cim_arrays >> (level + 1); // 每层的操作数
        
        for (int op = 0; op < num_ops; ++op) {
            yis::Instruction reduce_instr;
            reduce_instr.type = yis::InstructionType::REDUCE;
            reduce_instr.cim_array_id = op * stride * 2;
            reduce_instr.size = total_elements_ * 4 / (1 << level); // 每层数据量递减
            reduce_instr.sync_required = (level == num_levels - 1); // 最后一层需要同步
            
            instructions.push_back(reduce_instr);
        }
        
        // 每层之间添加同步点
        if (level < num_levels - 1) {
            yis::Instruction sync_instr;
            sync_instr.type = yis::InstructionType::SYNC_BAR;
            sync_instr.sync_required = true;
            instructions.push_back(sync_instr);
        }
    }
    
    return instructions;
}

bool YICAAllReduceOp::execute_all_reduce_operation() {
    std::cout << "🔧 执行 AllReduce 操作..." << std::endl;
    
    // 第一阶段：数据分发到各个 CIM 阵列
    std::cout << "🔧 阶段1: 数据分发..." << std::endl;
    size_t elements_per_cim = total_elements_ / get_hardware_config().num_cim_arrays;
    
    for (uint32_t cim_id = 0; cim_id < get_hardware_config().num_cim_arrays; ++cim_id) {
        // 模拟数据分发到CIM阵列的SPM
        std::cout << "📡 分发数据到 CIM-" << cim_id 
                  << " (元素数: " << elements_per_cim << ")" << std::endl;
    }
    
    // 第二阶段：树状归约计算
    std::cout << "🔧 阶段2: 树状归约计算..." << std::endl;
    int num_levels = static_cast<int>(std::log2(get_hardware_config().num_cim_arrays));
    
    for (int level = 0; level < num_levels; ++level) {
        int stride = 1 << level;
        int num_active_cims = get_hardware_config().num_cim_arrays >> (level + 1);
        
        std::cout << "🔧 归约层级 " << level + 1 << "/" << num_levels 
                  << " (活跃CIM: " << num_active_cims << ")" << std::endl;
        
        // 模拟并行归约操作
        for (int cim = 0; cim < num_active_cims; ++cim) {
            int src_cim = cim * stride * 2;
            int dst_cim = src_cim + stride;
            
            // 根据归约操作类型执行不同的计算
            switch (reduction_op_) {
                case AllReduceOp::SUM:
                    // 模拟求和操作
                    break;
                case AllReduceOp::MAX:
                    // 模拟求最大值操作
                    break;
                case AllReduceOp::MIN:
                    // 模拟求最小值操作
                    break;
                case AllReduceOp::MEAN:
                    // 模拟求均值操作
                    break;
                case AllReduceOp::PROD:
                    // 模拟求乘积操作
                    break;
            }
        }
        
        // 模拟层级同步延迟
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    // 第三阶段：结果广播（如果不是就地操作）
    if (!inplace_) {
        std::cout << "🔧 阶段3: 结果广播..." << std::endl;
        // 模拟将归约结果广播到所有CIM阵列
        for (uint32_t cim_id = 0; cim_id < get_hardware_config().num_cim_arrays; ++cim_id) {
            // 模拟广播操作
        }
    }
    
    // 模拟总体计算延迟
    double compute_intensity = total_elements_ * num_levels; // 归约操作的计算强度
    double execution_time_us = compute_intensity / (get_hardware_config().cim_compute_throughput_tflops * 1e12 / 1e6);
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(execution_time_us)));
    
    std::cout << "✅ AllReduce 操作执行完成" << std::endl;
    return true;
}

bool YICAAllReduceOp::validate_reduction_parameters() const {
    // 验证输入张量有效性
    if (input_tensor_.num_dims == 0) {
        std::cout << "❌ 输入张量维度为0" << std::endl;
        return false;
    }
    
    // 验证CIM阵列数量是2的幂次（用于树状归约）
    if ((get_hardware_config().num_cim_arrays & (get_hardware_config().num_cim_arrays - 1)) != 0) {
        std::cout << "⚠️ CIM阵列数量不是2的幂次，可能影响归约效率" << std::endl;
    }
    
    // 验证数据量是否适合SPM容量
    size_t required_bytes = input_tensor_.data_size();
    size_t available_bytes = get_hardware_config().spm_size_mb * 1024 * 1024;
    
    if (required_bytes > available_bytes) {
        std::cout << "⚠️ SPM 容量不足: 需要 " << (required_bytes / 1024.0 / 1024.0) 
                  << "MB, 可用 " << get_hardware_config().spm_size_mb << "MB" << std::endl;
        return false;
    }
    
    return true;
}

} // namespace kernel
} // namespace yirage