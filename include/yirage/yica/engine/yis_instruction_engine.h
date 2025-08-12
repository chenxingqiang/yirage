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

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "yirage/yica/yis_instruction_set.h"

namespace yirage {
namespace yica {

// YIS Instruction Engine for executing YIS instructions on YICA hardware
class YISInstructionEngine {
public:
    YISInstructionEngine();
    ~YISInstructionEngine();
    
    // Instruction generation
    std::vector<YISInstruction> generate_instructions(
        const std::string& operation_type,
        const std::vector<uint32_t>& input_shapes,
        const std::vector<uint32_t>& output_shapes,
        const std::string& data_type = "float32"
    );
    
    // Instruction execution
    bool execute_instructions(const std::vector<YISInstruction>& instructions);
    
    // Hardware communication
    bool send_to_hardware(const std::vector<YISInstruction>& instructions);
    std::vector<uint8_t> receive_from_hardware();
    
    // Status and debugging
    bool is_hardware_connected() const;
    std::string get_last_error() const;
    void set_debug_mode(bool enabled);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// YIS Instruction structure (compatible with existing YIS instruction set)
struct YISInstruction {
    YISInstructionType opcode;
    uint32_t cim_array_id;
    uint64_t spm_a_offset;
    uint64_t spm_b_offset;  
    uint64_t spm_c_offset;
    std::vector<uint32_t> dimensions;
    std::string data_type;
    
    YISInstruction() : opcode(YISInstructionType::YISECOPY_G2S), cim_array_id(0), 
                      spm_a_offset(0), spm_b_offset(0), spm_c_offset(0) {}
};

// CIM Array States
enum class CIMArrayState : int32_t {
    IDLE = 0,
    COMPUTING = 1,
    MEMORY_TRANSFER = 2,
    ERROR = 3
};

// CIM Array Information
struct CIMArrayInfo {
    uint32_t array_id;
    CIMArrayState state;
    float utilization;
    uint64_t memory_usage;
    std::string current_operation;
};

} // namespace yica
} // namespace yirage
