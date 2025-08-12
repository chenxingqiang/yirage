# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
YICA Types Cython Header
Defines common YICA types and structures for Cython bindings
"""

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr
from libc.stdint cimport uint32_t, uint64_t, int64_t

# YICA Configuration Types
cdef extern from "yirage/yica/config.h" namespace "yirage::yica":
    cdef cppclass YICAConfig:
        uint32_t num_cim_arrays
        uint64_t spm_size_per_die
        bool enable_hardware_acceleration
        bool enable_profiling
        string hardware_target
        
# YICA Hardware Types
cdef enum YICAArchitecture:
    YICA_V1_0 = 100
    YICA_V1_1 = 101
    YICA_V2_0 = 200
    SIMULATION = 999
    UNKNOWN = -1

cdef enum YICAExecutionMode:
    HARDWARE = 0
    SIMULATION_MODE = 1
    HYBRID = 2

cdef enum CIMArrayType:
    STANDARD = 0
    HIGH_PRECISION = 1
    LOW_POWER = 2
    ADAPTIVE = 3

cdef enum MemoryLevel:
    REGISTER_FILE = 0
    SPM = 1
    DRAM = 2

# YICA Performance Types
cdef struct YICAPerformanceMetrics:
    float cim_utilization
    float spm_hit_rate
    float memory_bandwidth_usage
    float power_consumption
    int64_t instruction_throughput

# YICA Resource Types
cdef struct CIMResourceAllocation:
    vector[uint32_t] allocated_arrays
    uint64_t allocated_memory
    float utilization_target

cdef struct SPMMemoryPlan:
    uint64_t total_size
    uint64_t allocated_size
    vector[uint64_t] buffer_offsets

# YICA Instruction Types
cdef enum YISInstructionType:
    YISECOPY = 0    # External copy
    YISICOPY = 1    # Internal copy  
    YISMMA = 2      # Matrix multiply
    YISSYNC = 3     # Synchronization
    YISCONTROL = 4  # Control flow

cdef struct YISInstruction:
    YISInstructionType opcode
    uint32_t cim_array_id
    uint64_t smp_a_offset
    uint64_t smp_b_offset  
    uint64_t smp_c_offset
    vector[uint32_t] dimensions
    string data_type

# YICA Optimization Types
cdef struct YICAOptimizationResult:
    string yis_kernel_code
    string triton_kernel_code
    CIMResourceAllocation cim_allocation
    SPMMemoryPlan spm_memory_plan
    float estimated_speedup
    uint64_t memory_footprint
    vector[string] optimization_log

# YICA Error Types
cdef enum YICAErrorType:
    HARDWARE_TIMEOUT = 0
    COMMUNICATION_ERROR = 1
    MEMORY_ERROR = 2
    COMPUTATION_ERROR = 3

cdef struct YICAErrorContext:
    YICAErrorType error_type
    string error_message
    uint32_t error_code
    string context_info

# Helper type aliases for Python compatibility
ctypedef YICAConfig* YICAConfigPtr
ctypedef YICAPerformanceMetrics* YICAPerformanceMetricsPtr
ctypedef CIMResourceAllocation* CIMResourceAllocationPtr
ctypedef SPMMemoryPlan* SPMMemoryPlanPtr
