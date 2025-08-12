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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "yirage/kernel/graph.h"
#include "yirage/kernel/device_tensor.h"

namespace yirage {
namespace kernel {

// Forward declarations
class Graph;
struct DTensor;

// YICA硬件配置
struct YICAHardwareConfig {
    // CIM (Compute-in-Memory) 配置
    uint32_t num_cim_arrays = 16;
    float cim_compute_throughput_tflops = 0.1f;  // 0.1 TFLOPS per CIM

    // SPM (Scratchpad Memory) 配置
    uint32_t spm_size_mb = 2;  // 2MB SPM per die
    float spm_bandwidth_tb_s = 2.0f;  // 2TB/s bandwidth
    bool enable_double_buffering = true;

    // 内存层次配置
    float dram_bandwidth_gb_s = 1024.0f;  // 1TB/s DRAM bandwidth

    // 向量化配置
    uint32_t vector_width = 16;  // SIMD width
    bool enable_mixed_precision = true;
};

// YIS指令类型 (简化统一版本)
namespace yis {
    enum class InstructionType {
        // 数据移动指令
        ECOPY_G2S,    // Global to SPM
        ECOPY_S2G,    // SPM to Global
        ICOPY_S2S,    // SPM to SPM
        ICOPY_BC,     // Broadcast

        // 计算指令
        MMA,          // Matrix multiply-accumulate
        REDUCE,       // Reduction operation
        ELEM_OP,      // Element-wise operation

        // 同步指令
        SYNC_BAR,     // Barrier sync
        SYNC_CIM,     // CIM array sync
    };

    struct Instruction {
        InstructionType type;
        std::string opcode;
        std::vector<std::string> operands;
        std::map<std::string, std::string> metadata;
        uint64_t src_addr = 0;
        uint64_t dst_addr = 0;
        size_t size = 0;
        int cim_array_id = -1;
        bool sync_required = false;

        // 矩阵操作相关字段
        int matrix_m = 0;
        int matrix_n = 0;
        int matrix_k = 0;
    };
}

// YICA Kernel基类
class YICAKernelBase {
public:
    YICAKernelBase(Graph* graph, const YICAHardwareConfig& config = YICAHardwareConfig{});
    virtual ~YICAKernelBase();

    // 核心接口
    virtual bool initialize() = 0;
    virtual bool execute() = 0;
    virtual std::vector<yis::Instruction> generate_yis_instructions() = 0;

    // 性能接口
    virtual double get_execution_time_ms() const { return execution_time_ms_; }
    virtual float get_cim_utilization() const { return cim_utilization_; }
    virtual float get_memory_efficiency() const { return memory_efficiency_; }

    // 配置接口
    void set_hardware_config(const YICAHardwareConfig& config) { hw_config_ = config; }
    const YICAHardwareConfig& get_hardware_config() const { return hw_config_; }

protected:
    // 辅助方法
    size_t calculate_spm_tiling(size_t data_size) const;
    int select_best_cim_array(size_t workload) const;
    std::vector<yis::Instruction> generate_data_movement_instructions(
        const DTensor& src, const DTensor& dst, bool use_double_buffer = false);

    // 成员变量
    Graph* graph_;
    YICAHardwareConfig hw_config_;
    std::vector<yis::Instruction> instruction_sequence_;

    // 性能指标
    double execution_time_ms_ = 0.0;
    float cim_utilization_ = 0.0f;
    float memory_efficiency_ = 0.0f;
    bool is_initialized_ = false;
};

} // namespace kernel
} // namespace yirage
