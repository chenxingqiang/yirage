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
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "../config.h"

namespace yirage {
namespace yica {

// CIM Array Simulator for yz-g100 hardware
class CIMArraySimulator {
public:
    explicit CIMArraySimulator(const YICAConfig& config);
    ~CIMArraySimulator();
    
    // Array management
    bool initialize_arrays();
    bool shutdown_arrays();
    
    // Simulation control
    void start_simulation();
    void stop_simulation();
    bool is_simulation_running() const;
    
    // Array operations
    bool execute_on_array(uint32_t array_id, const YISInstruction& instruction);
    // std::vector<CIMArrayInfo> get_array_states() const; // TODO: Define CIMArrayInfo
    
    // Performance metrics
    float get_total_utilization() const;
    uint64_t get_total_operations() const;
    double get_simulation_time_ms() const;
    
    // Configuration
    void set_array_count(uint32_t count);
    void set_memory_per_array(uint64_t memory_size);
    void enable_power_modeling(bool enabled);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace yica
} // namespace yirage
