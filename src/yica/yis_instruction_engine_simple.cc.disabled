#include "yirage/yica/yis_instruction_engine.h"
#include "yirage/yica/yis_instruction_set.h"
#include "yirage/yica/engine/cim_array_simulator.h"
#include "yirage/yica/spm_memory_manager.h"
#include <algorithm>

namespace yirage {
namespace yica {

YISInstructionEngine::YISInstructionEngine(const YICAConfig& config)
    : config_(config) {
    
    // 初始化CIM阵列模拟器
    cim_simulator_ = std::make_unique<CIMArraySimulator>(config_);
    
    // 初始化SPM内存管理器  
    smp_manager_ = std::make_unique<SPMMemoryManager>(config_);
}

YISInstructionEngine::~YISInstructionEngine() = default;

std::vector<YISInstruction> YISInstructionEngine::parse_yis_code(const std::string& yis_code) {
    std::vector<YISInstruction> instructions;
    
    // 简化的解析：按行分割
    std::istringstream stream(yis_code);
    std::string line;
    
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#' || line.substr(0, 2) == "//") {
            continue; // 跳过空行和注释
        }
        
        YISInstruction instruction;
        instruction.opcode = line;
        instruction.type = parse_instruction_type(line);
        instructions.push_back(instruction);
    }
    
    return instructions;
}

bool YISInstructionEngine::execute_instruction(const YISInstruction& instruction) {
    execution_log_.push_back("Executing: " + instruction.opcode);
    
    switch (instruction.type) {
        case YISInstructionType::YISECOPY_G2S:
        case YISInstructionType::YISECOPY_S2G:
        case YISInstructionType::YISECOPY_G2G:
            return execute_copy_instruction(instruction);
            
        case YISInstructionType::YISGEMM:
        case YISInstructionType::YISCONV:
            return execute_compute_instruction(instruction);
            
        default:
            execution_log_.push_back("Unknown instruction type");
            return false;
    }
}

bool YISInstructionEngine::execute_instructions(const std::vector<YISInstruction>& instructions) {
    for (const auto& instruction : instructions) {
        if (!execute_instruction(instruction)) {
            return false;
        }
    }
    return true;
}

bool YISInstructionEngine::validate_instruction(const YISInstruction& instruction) {
    // 简单验证：检查opcode不为空
    return !instruction.opcode.empty();
}

std::vector<YISInstruction> YISInstructionEngine::optimize_instructions(
    const std::vector<YISInstruction>& instructions) {
    // 简化的优化：直接返回原指令序列
    return instructions;
}

std::vector<std::string> YISInstructionEngine::get_supported_opcodes() const {
    return {
        "YISECOPY_G2S", "YISECOPY_S2G", "YISECOPY_G2G",
        "YISGEMM", "YISCONV", "YISRELU", "YISSOFTMAX"
    };
}

void YISInstructionEngine::reset() {
    execution_log_.clear();
    registers_.clear();
}

// 私有方法实现
bool YISInstructionEngine::execute_cim_instruction(const YISInstruction& instruction) {
    // 简化实现
    execution_log_.push_back("CIM instruction executed: " + instruction.opcode);
    return true;
}

bool YISInstructionEngine::execute_smp_instruction(const YISInstruction& instruction) {
    // 简化实现
    execution_log_.push_back("SPM instruction executed: " + instruction.opcode);
    return true;
}

bool YISInstructionEngine::execute_control_instruction(const YISInstruction& instruction) {
    // 简化实现
    execution_log_.push_back("Control instruction executed: " + instruction.opcode);
    return true;
}

bool YISInstructionEngine::execute_copy_instruction(const YISInstruction& instruction) {
    // 简化实现
    execution_log_.push_back("Copy instruction executed: " + instruction.opcode);
    return true;
}

bool YISInstructionEngine::execute_compute_instruction(const YISInstruction& instruction) {
    // 简化实现
    execution_log_.push_back("Compute instruction executed: " + instruction.opcode);
    return true;
}

YISInstructionType YISInstructionEngine::parse_instruction_type(const std::string& opcode) {
    if (opcode.find("YISECOPY_G2S") != std::string::npos) {
        return YISInstructionType::YISECOPY_G2S;
    } else if (opcode.find("YISECOPY_S2G") != std::string::npos) {
        return YISInstructionType::YISECOPY_S2G;
    } else if (opcode.find("YISGEMM") != std::string::npos) {
        return YISInstructionType::YISGEMM;
    } else if (opcode.find("YISCONV") != std::string::npos) {
        return YISInstructionType::YISCONV;
    } else {
        return YISInstructionType::YISECOPY_G2S; // 默认类型
    }
}

std::vector<std::string> YISInstructionEngine::parse_operands(const std::string& operand_str) {
    std::vector<std::string> operands;
    std::istringstream stream(operand_str);
    std::string operand;
    
    while (stream >> operand) {
        operands.push_back(operand);
    }
    
    return operands;
}

}  // namespace yica
}  // namespace yirage
