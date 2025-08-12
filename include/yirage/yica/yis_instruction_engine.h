// Copyright 2024 CMU
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <atomic>
#include <cstdint>
#include "yirage/yica/config.h"
#include "yirage/yica/yis_instruction_set.h"

namespace yirage {
namespace yica {

// YISInstructionType is defined in yis_instruction_set.h

// YIS 指令结构 - 生产级别
struct YISInstruction {
    // 基本指令信息
    YISInstructionType type;
    std::string opcode;
    std::vector<std::string> operands;
    std::unordered_map<std::string, std::string> attributes;
    
    // 内存操作相关
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    size_t size = 0;
    
    // CIM 阵列相关
    uint32_t cim_array_id = 0;
    
    // 矩阵操作相关
    int matrix_m = 0;
    int matrix_n = 0; 
    int matrix_k = 0;
    
    // 操作类型和精度
    enum class YISOperation {
        MATRIX_MULTIPLY_ACCUMULATE,
        REDUCE_SUM,
        CONDITIONAL_BRANCH,
        LOOP_CONTROL,
        KERNEL_END
    } operation = YISOperation::MATRIX_MULTIPLY_ACCUMULATE;
    
    enum class YICAPrecision {
        FP32,
        FP16,
        INT8
    } precision = YICAPrecision::FP32;
    
    // 同步和控制
    bool sync_required = false;
    
    // 构造函数
    YISInstruction() : type(YISInstructionType::YISECOPY_G2S) {}
    YISInstruction(YISInstructionType t, const std::string& op) 
        : type(t), opcode(op) {}
};

// 执行统计结构
struct ExecutionStats {
    uint64_t total_instructions = 0;
    uint64_t failed_instructions = 0;
    double total_execution_time_us = 0.0;
    double memory_access_bytes = 0.0;
    double average_instruction_time_us = 0.0;
};

// YIS 指令引擎
class YISInstructionEngine {
public:
    explicit YISInstructionEngine(const YICAConfig& config);
    ~YISInstructionEngine();
    
    // 解析YIS代码
    std::vector<YISInstruction> parse_yis_code(const std::string& yis_code);
    
    // 执行YIS指令
    bool execute_instruction(const YISInstruction& instruction);
    
    // 批量执行指令
    bool execute_instructions(const std::vector<YISInstruction>& instructions);
    
    // 验证指令合法性
    bool validate_instruction(const YISInstruction& instruction);
    
    // 优化指令序列
    std::vector<YISInstruction> optimize_instructions(
        const std::vector<YISInstruction>& instructions);
    
    // 获取支持的指令集
    std::vector<std::string> get_supported_opcodes() const;
    
    // 重置引擎状态
    void reset();
    
    // 获取执行统计
    ExecutionStats get_execution_stats() const;
    
    // 重置统计信息
    void reset_stats();
    
private:
    YICAConfig config_;
    
    // 内部状态
    std::unordered_map<std::string, std::string> registers_;
    std::vector<std::string> execution_log_;
    
    // 核心组件 - 暂时简化实现
    // std::unique_ptr<class CIMArraySimulator> cim_simulator_;  // Phase 2
    // std::unique_ptr<class SPMMemoryManager> spm_manager_;      // Phase 2
    
    // 执行状态
    std::atomic<bool> is_running_;
    std::vector<YISInstruction> instruction_queue_;
    
    // 执行统计
    ExecutionStats execution_stats_;
    
    // 内部方法 - 指令执行
    bool execute_cim_instruction(const YISInstruction& instruction);
    bool execute_smp_instruction(const YISInstruction& instruction);
    bool execute_control_instruction(const YISInstruction& instruction);
    bool execute_copy_instruction(const YISInstruction& instruction);
    bool execute_compute_instruction(const YISInstruction& instruction);
    
    // 具体指令实现
    bool execute_external_copy(const YISInstruction& instruction);
    bool execute_internal_copy(const YISInstruction& instruction);
    bool execute_matrix_multiply(const YISInstruction& instruction);
    bool execute_synchronization(const YISInstruction& instruction);
    bool execute_control_flow(const YISInstruction& instruction);
    
    // 高级指令实现
    bool execute_mma_operation(const YISInstruction& instruction, void* cim_array);
    bool execute_reduce_operation(const YISInstruction& instruction, void* cim_array);
    bool execute_conditional_branch(const YISInstruction& instruction);
    bool execute_loop_control(const YISInstruction& instruction);
    
    // 指令解析辅助
    bool parse_single_instruction(const std::string& line, YISInstruction& instruction);
    YISInstructionType parse_instruction_type(const std::string& opcode);
    std::vector<std::string> parse_operands(const std::string& operand_str);
    
    // 指令优化
    void optimize_memory_copies(std::vector<YISInstruction>& instructions);
    void optimize_synchronization(std::vector<YISInstruction>& instructions);
    void optimize_instruction_scheduling(std::vector<YISInstruction>& instructions);
    
    // 性能监控
    void update_execution_stats(double execution_time_us, bool success);
    
    // 生命周期管理
    void start();
    void stop();
};

}  // namespace yica
}  // namespace yirage
