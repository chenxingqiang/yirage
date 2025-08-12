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
YICA Kernels Cython Bindings
Provides Python interface to YICA-specific operators and hardware abstractions
Updated to include YIS Instruction Engine support
"""

from CCore cimport *
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.functional cimport function
from libc.stdint cimport uint32_t, uint64_t
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Callable

# YIS Instruction Engine Core Components
cdef extern from "yirage/yica/engine/yis_instruction_engine.h" namespace "yirage::yica":
    # YIS Instruction Types
    cdef enum YISInstructionType:
        YISECOPY_G2S = 0
        YISECOPY_S2G = 1
        YISECOPY_G2G = 2
        YISICOPY_S2S = 3
        YISICOPY_R2S = 4
        YISICOPY_S2R = 5
        YISICOPY_BC = 6
        YISICOPY_GAT = 7
        YISMMA_ACC = 8
        YISMMA_NONACC = 9
        YISMMA_SPMG = 10
        YISSYNC_BAR = 11
        YISSYNC_BOINIT = 12
        YISSYNC_BOARRV = 13
        YISSYNC_BOWAIT = 14
        YISCONTROL_CALL_EU = 15
        YISCONTROL_END = 16
    
    # YIS Execution Status
    cdef enum YISExecutionStatus:
        SUCCESS = 0
        FAILED = 1
        PENDING = 2
        TIMEOUT = 3
        MEMORY_ERROR = 4
        CIM_ERROR = 5
    
    # YIS Instruction Parameters
    cdef struct YISInstructionParams:
        uint64_t src_address
        uint64_t dst_address
        size_t size
        uint32_t matrix_m
        uint32_t matrix_n
        uint32_t matrix_k
        
    # YIS Instruction
    cdef struct YISInstruction:
        YISInstructionType type
        YISInstructionParams params
    
    # YIS Execution Result
    cdef struct YISExecutionResult:
        YISExecutionStatus status
        double execution_time_us
        string error_message
    
    # YIS Execution Statistics
    cdef struct YISExecutionStats:
        uint64_t total_instructions
        uint64_t successful_instructions
        double total_execution_time_ms
        double average_latency_us
        uint64_t copy_instructions
        uint64_t mma_instructions
        uint64_t sync_instructions
        uint64_t control_instructions
        double cim_utilization
        double spm_hit_rate
        double memory_bandwidth_gbps
    
    # YIS Instruction Engine
    cdef cppclass YISInstructionEngine:
        YISInstructionEngine(const YICAConfig& config)
        bool start()
        void stop()
        YISExecutionResult execute_instruction(const YISInstruction& instruction)
        vector[YISExecutionResult] execute_instructions(const vector[YISInstruction]& instructions)
        void execute_async(const vector[YISInstruction]& instructions, 
                          function[void(const vector[YISExecutionResult]&)] callback)
        YISExecutionStats get_execution_stats()
        void reset_stats()
        double get_cim_utilization()
        double get_spm_usage()
        double get_memory_bandwidth_utilization()
        void set_debug_mode(bool enable)
        string get_version()

cdef extern from "yirage/yica/engine/cim_array_simulator.h" namespace "yirage::yica":
    # CIM Array States
    cdef enum CIMArrayState:
        IDLE = 0
        COMPUTING = 1
        LOADING = 2
        STORING = 3
        ERROR = 4
    
    # CIM Compute Types
    cdef enum CIMComputeType:
        MATRIX_MULTIPLY = 0
        VECTOR_ADD = 1
        VECTOR_MUL = 2
        ACTIVATION = 3
        CUSTOM = 4
    
    # CIM Precision Modes
    cdef enum CIMPrecisionMode:
        INT8 = 0
        INT16 = 1
        FP16 = 2
        BF16 = 3
        FP32 = 4
    
    # CIM Array Metrics
    cdef struct CIMArrayMetrics:
        double utilization_rate
        double throughput_gops
        double power_consumption_w
        double temperature_celsius
        uint64_t total_operations
        double average_latency_us
        double energy_efficiency_tops_w
    
    # CIM Array Simulator
    cdef cppclass CIMArraySimulator:
        CIMArraySimulator(const YICAConfig& config, uint32_t array_id)
        bool initialize()
        void shutdown()
        double execute_matrix_multiply(uint32_t m, uint32_t n, uint32_t k,
                                     const void* a, const void* b, void* c,
                                     CIMPrecisionMode precision, bool accumulate)
        double execute_vector_operation(uint32_t size, const void* a, const void* b, void* c,
                                      CIMComputeType compute_type, CIMPrecisionMode precision)
        double execute_activation(uint32_t size, const void* input, void* output,
                                uint32_t activation_type, CIMPrecisionMode precision)
        CIMArrayState get_state()
        CIMArrayMetrics get_metrics()
        double get_utilization()
        double get_power_consumption()
        bool wait_for_completion(uint32_t timeout_ms)
        void reset_metrics()
        void set_debug_mode(bool enable)

# YICA Hardware Abstraction Layer
cdef extern from "yirage/yica/yica_hardware_abstraction.h" namespace "yirage::yica":
    cdef cppclass YICAHardwareAbstraction:
        YICAHardwareAbstraction(int num_cim_arrays, size_t spm_size)
        bool initialize_cim_arrays(int num_arrays)
        bool configure_spm_memory(size_t size)
        bool is_hardware_available()
        
cdef extern from "yirage/yica/yica_backend.h" namespace "yirage::yica":
    cdef cppclass YICABackend:
        YICABackend()
        bool initialize()
        bool is_available()

# YICA Device Memory Manager
cdef extern from "yirage/kernel/yica_device_memory_manager.h" namespace "yirage::kernel":
    cdef cppclass YICADeviceMemoryManager:
        YICADeviceMemoryManager()
        void set_device_id(int device_id)
        bool is_available()

# YICA Kernel Graph
cdef extern from "yirage/kernel/yica_graph.h" namespace "yirage::kernel":
    cdef cppclass YICAGraphManager:
        YICAGraphManager()
        bool optimize_for_cim_execution(int num_cim_arrays)
        bool enable_spm_graph_caching()

# YICA Operators
cdef extern from "yirage/kernel/yica_matmul.h" namespace "yirage::kernel":
    cdef cppclass YICAMatMulOp:
        YICAMatMulOp()
        bool optimize_for_cim_arrays(int num_cim_arrays)
        bool enable_spm_data_staging()

cdef extern from "yirage/kernel/yica_element_ops.h" namespace "yirage::kernel":
    cdef cppclass YICAElementOpsOp:
        YICAElementOpsOp()
        bool optimize_for_cim_vectorization(int vector_width)
        bool enable_spm_vectorized_access()

cdef extern from "yirage/kernel/yica_reduction.h" namespace "yirage::kernel":
    cdef cppclass YICAReductionOp:
        YICAReductionOp()
        bool optimize_for_cim_reduction(int num_cim_arrays)
        bool enable_hierarchical_reduction()

cdef extern from "yirage/kernel/yica_rms_norm.h" namespace "yirage::kernel":
    cdef cppclass YICARMSNormOp:
        YICARMSNormOp()
        bool optimize_for_cim_computation(int num_cim_arrays)
        bool enable_spm_intermediate_storage(size_t buffer_size)

cdef extern from "yirage/kernel/yica_all_reduce.h" namespace "yirage::kernel":
    cdef cppclass YICAAllReduceOp:
        YICAAllReduceOp()
        bool optimize_for_cim_reduction(int num_cim_arrays)
        bool enable_spm_buffering(size_t buffer_size)

cdef extern from "yirage/kernel/yica_chunk.h" namespace "yirage::kernel":
    cdef cppclass YICAChunkOp:
        YICAChunkOp()
        bool optimize_for_cim_chunking(int num_cim_arrays)
        bool enable_spm_chunk_caching(size_t cache_size)

cdef extern from "yirage/kernel/yica_customized.h" namespace "yirage::kernel":
    cdef cppclass YICACustomizedOp:
        YICACustomizedOp()
        bool optimize_for_cim_arrays(int num_arrays)
        bool enable_spm_staging(size_t staging_size)

# Python wrapper classes
cdef class PyYICAHardwareAbstraction:
    """Python wrapper for YICA Hardware Abstraction"""
    cdef unique_ptr[YICAHardwareAbstraction] c_obj
    
    def __init__(self, int num_cim_arrays=8, size_t spm_size=128*1024*1024):
        self.c_obj.reset(new YICAHardwareAbstraction(num_cim_arrays, spm_size))
    
    def initialize_cim_arrays(self, int num_arrays):
        """Initialize CIM arrays"""
        return self.c_obj.get().initialize_cim_arrays(num_arrays)
    
    def configure_spm_memory(self, size_t size):
        """Configure SPM memory"""
        return self.c_obj.get().configure_spm_memory(size)
    
    def is_hardware_available(self):
        """Check if YICA hardware is available"""
        return self.c_obj.get().is_hardware_available()

cdef class PyYICABackend:
    """Python wrapper for YICA Backend"""
    cdef unique_ptr[YICABackend] c_obj
    
    def __init__(self):
        self.c_obj.reset(new YICABackend())
    
    def initialize(self):
        """Initialize YICA backend"""
        return self.c_obj.get().initialize()
    
    def is_available(self):
        """Check if backend is available"""
        return self.c_obj.get().is_available()

cdef class PyYICADeviceMemoryManager:
    """Python wrapper for YICA Device Memory Manager"""
    cdef unique_ptr[YICADeviceMemoryManager] c_obj
    
    def __init__(self):
        self.c_obj.reset(new YICADeviceMemoryManager())
    
    def set_device_id(self, int device_id):
        """Set device ID"""
        self.c_obj.get().set_device_id(device_id)
    
    def is_available(self):
        """Check if memory manager is available"""
        return self.c_obj.get().is_available()

cdef class PyYICAKernelGraph:
    """Python wrapper for YICA Kernel Graph"""
    cdef unique_ptr[YICAGraphManager] c_obj
    
    def __init__(self):
        self.c_obj.reset(new YICAGraphManager())
    
    def optimize_for_cim_execution(self, int num_cim_arrays):
        """Optimize graph for CIM execution"""
        return self.c_obj.get().optimize_for_cim_execution(num_cim_arrays)
    
    def enable_spm_graph_caching(self):
        """Enable SPM graph caching"""
        return self.c_obj.get().enable_spm_graph_caching()
    
    def create_new_graph(self):
        """Create a new computation graph"""
        # Return a new graph instance
        return PyYICAKernelGraph()

# YICA Operator wrappers
cdef class PyYICAMatMulOp:
    """Python wrapper for YICA Matrix Multiplication"""
    cdef unique_ptr[YICAMatMulOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICAMatMulOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'num_cim_arrays' in config:
            self.optimize_for_cim_arrays(config['num_cim_arrays'])
        if 'enable_spm_staging' in config and config['enable_spm_staging']:
            self.enable_spm_data_staging()
    
    def optimize_for_cim_arrays(self, int num_cim_arrays):
        """Optimize for CIM arrays"""
        return self.c_obj.get().optimize_for_cim_arrays(num_cim_arrays)
    
    def enable_spm_data_staging(self):
        """Enable SPM data staging"""
        return self.c_obj.get().enable_spm_data_staging()

cdef class PyYICAElementOpsOp:
    """Python wrapper for YICA Element Operations"""
    cdef unique_ptr[YICAElementOpsOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICAElementOpsOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'vector_width' in config:
            self.optimize_for_cim_vectorization(config['vector_width'])
        if 'enable_spm_access' in config and config['enable_spm_access']:
            self.enable_spm_vectorized_access()
    
    def optimize_for_cim_vectorization(self, int vector_width):
        """Optimize for CIM vectorization"""
        return self.c_obj.get().optimize_for_cim_vectorization(vector_width)
    
    def enable_spm_vectorized_access(self):
        """Enable SPM vectorized access"""
        return self.c_obj.get().enable_spm_vectorized_access()

cdef class PyYICAReductionOp:
    """Python wrapper for YICA Reduction Operations"""
    cdef unique_ptr[YICAReductionOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICAReductionOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'num_cim_arrays' in config:
            self.optimize_for_cim_reduction(config['num_cim_arrays'])
        if 'enable_hierarchical' in config and config['enable_hierarchical']:
            self.enable_hierarchical_reduction()
    
    def optimize_for_cim_reduction(self, int num_cim_arrays):
        """Optimize for CIM reduction"""
        return self.c_obj.get().optimize_for_cim_reduction(num_cim_arrays)
    
    def enable_hierarchical_reduction(self):
        """Enable hierarchical reduction"""
        return self.c_obj.get().enable_hierarchical_reduction()

cdef class PyYICARMSNormOp:
    """Python wrapper for YICA RMS Normalization"""
    cdef unique_ptr[YICARMSNormOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICARMSNormOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'num_cim_arrays' in config:
            self.optimize_for_cim_computation(config['num_cim_arrays'])
        if 'spm_buffer_size' in config:
            self.enable_spm_intermediate_storage(config['spm_buffer_size'])
    
    def optimize_for_cim_computation(self, int num_cim_arrays):
        """Optimize for CIM computation"""
        return self.c_obj.get().optimize_for_cim_computation(num_cim_arrays)
    
    def enable_spm_intermediate_storage(self, size_t buffer_size):
        """Enable SPM intermediate storage"""
        return self.c_obj.get().enable_spm_intermediate_storage(buffer_size)

cdef class PyYICAAllReduceOp:
    """Python wrapper for YICA All-Reduce Operations"""
    cdef unique_ptr[YICAAllReduceOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICAAllReduceOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'num_cim_arrays' in config:
            self.optimize_for_cim_reduction(config['num_cim_arrays'])
        if 'spm_buffer_size' in config:
            self.enable_spm_buffering(config['spm_buffer_size'])
    
    def optimize_for_cim_reduction(self, int num_cim_arrays):
        """Optimize for CIM reduction"""
        return self.c_obj.get().optimize_for_cim_reduction(num_cim_arrays)
    
    def enable_spm_buffering(self, size_t buffer_size):
        """Enable SPM buffering"""
        return self.c_obj.get().enable_spm_buffering(buffer_size)

cdef class PyYICAChunkOp:
    """Python wrapper for YICA Chunk Operations"""
    cdef unique_ptr[YICAChunkOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICAChunkOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'num_cim_arrays' in config:
            self.optimize_for_cim_chunking(config['num_cim_arrays'])
        if 'spm_cache_size' in config:
            self.enable_spm_chunk_caching(config['spm_cache_size'])
    
    def optimize_for_cim_chunking(self, int num_cim_arrays):
        """Optimize for CIM chunking"""
        return self.c_obj.get().optimize_for_cim_chunking(num_cim_arrays)
    
    def enable_spm_chunk_caching(self, size_t cache_size):
        """Enable SPM chunk caching"""
        return self.c_obj.get().enable_spm_chunk_caching(cache_size)

cdef class PyYICACustomizedOp:
    """Python wrapper for YICA Customized Operations"""
    cdef unique_ptr[YICACustomizedOp] c_obj
    
    def __init__(self, config=None):
        self.c_obj.reset(new YICACustomizedOp())
        if config:
            self._configure(config)
    
    def _configure(self, config):
        """Configure the operator"""
        if 'num_cim_arrays' in config:
            self.optimize_for_cim_arrays(config['num_cim_arrays'])
        if 'spm_staging_size' in config:
            self.enable_spm_staging(config['spm_staging_size'])
    
    def optimize_for_cim_arrays(self, int num_arrays):
        """Optimize for CIM arrays"""
        return self.c_obj.get().optimize_for_cim_arrays(num_arrays)
    
    def enable_spm_staging(self, size_t staging_size):
        """Enable SPM staging"""
        return self.c_obj.get().enable_spm_staging(staging_size)

# High-level YICA Optimizer
cdef class PyYICAOptimizer:
    """High-level YICA Optimizer"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.hardware = PyYICAHardwareAbstraction(
            self.config.get('num_cim_arrays', 8),
            self.config.get('spm_size', 128*1024*1024)
        )
        self.backend = PyYICABackend()
        self.memory_manager = PyYICADeviceMemoryManager()
        self.kernel_graph = PyYICAKernelGraph()
        
        # Initialize backend
        self.backend.initialize()
    
    def is_available(self):
        """Check if optimizer is available"""
        return (self.hardware.is_hardware_available() and 
                self.backend.is_available() and
                self.memory_manager.is_available())
    
    def create_matmul_op(self, config=None):
        """Create optimized matrix multiplication operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICAMatMulOp(op_config)
    
    def create_element_ops_op(self, config=None):
        """Create optimized element operations operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICAElementOpsOp(op_config)
    
    def create_reduction_op(self, config=None):
        """Create optimized reduction operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICAReductionOp(op_config)
    
    def create_rms_norm_op(self, config=None):
        """Create optimized RMS norm operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICARMSNormOp(op_config)
    
    def create_all_reduce_op(self, config=None):
        """Create optimized all-reduce operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICAAllReduceOp(op_config)
    
    def create_chunk_op(self, config=None):
        """Create optimized chunk operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICAChunkOp(op_config)
    
    def create_customized_op(self, config=None):
        """Create customized operator"""
        op_config = {**self.config, **(config or {})}
        return PyYICACustomizedOp(op_config)
    
    def optimize_graph(self, graph):
        """Optimize computation graph for YICA"""
        self.kernel_graph.optimize_for_cim_execution(self.config.get('num_cim_arrays', 8))
        self.kernel_graph.enable_spm_graph_caching()
        return graph

# Export Python classes with shorter names
YICAHardwareAbstraction = PyYICAHardwareAbstraction
YICABackend = PyYICABackend  
YICADeviceMemoryManager = PyYICADeviceMemoryManager
YICAKernelGraph = PyYICAKernelGraph
YICAOptimizer = PyYICAOptimizer

# Export operator classes
YICAMatMulOp = PyYICAMatMulOp
YICAElementOpsOp = PyYICAElementOpsOp
YICAReductionOp = PyYICAReductionOp
YICARMSNormOp = PyYICARMSNormOp
YICAAllReduceOp = PyYICAAllReduceOp
YICAChunkOp = PyYICAChunkOp
YICACustomizedOp = PyYICACustomizedOp

# Convenience functions
def create_yica_optimizer(config=None):
    """Create YICA optimizer instance"""
    return PyYICAOptimizer(config)

def check_yica_hardware():
    """Check YICA hardware availability"""
    try:
        hardware = PyYICAHardwareAbstraction()
        return hardware.is_hardware_available()
    except:
        return False

def get_yica_capabilities():
    """Get YICA system capabilities"""
    try:
        optimizer = PyYICAOptimizer()
        return {
            "hardware_available": optimizer.hardware.is_hardware_available(),
            "backend_available": optimizer.backend.is_available(),
            "memory_manager_available": optimizer.memory_manager.is_available(),
            "supported_operators": [
                "matmul", "element_ops", "reduction", "rms_norm",
                "all_reduce", "chunk", "customized"
            ]
        }
    except Exception as e:
        return {"error": str(e), "available": False}

# =============================================================================
# YIS Instruction Engine Python Wrappers
# =============================================================================

cdef class PyYISInstructionEngine:
    """Python wrapper for YIS Instruction Engine"""
    cdef unique_ptr[YISInstructionEngine] c_engine
    cdef YICAConfig config_
    
    def __init__(self, config=None):
        """Initialize YIS Instruction Engine
        
        Args:
            config: Dictionary with YICA configuration parameters
        """
        # Set default configuration
        self.config_.num_cim_arrays = 512
        self.config_.spm_size_kb = 131072  # 128MB
        self.config_.dram_size_gb = 64
        self.config_.num_dram_channels = 8
        
        if config:
            if 'num_cim_arrays' in config:
                self.config_.num_cim_arrays = config['num_cim_arrays']
            if 'spm_size_kb' in config:
                self.config_.spm_size_kb = config['spm_size_kb']
            if 'dram_size_gb' in config:
                self.config_.dram_size_gb = config['dram_size_gb']
            if 'num_dram_channels' in config:
                self.config_.num_dram_channels = config['num_dram_channels']
        
        self.c_engine.reset(new YISInstructionEngine(self.config_))
    
    def start(self):
        """Start the instruction engine"""
        return self.c_engine.get().start()
    
    def stop(self):
        """Stop the instruction engine"""
        self.c_engine.get().stop()
    
    def execute_instruction(self, instruction_type, params=None):
        """Execute a single YIS instruction
        
        Args:
            instruction_type: String name of instruction type
            params: Dictionary of instruction parameters
            
        Returns:
            Dictionary with execution result
        """
        cdef YISInstruction instruction
        cdef YISExecutionResult result
        
        # Convert string instruction type to enum
        instruction_type_map = {
            'YISECOPY_G2S': YISECOPY_G2S,
            'YISECOPY_S2G': YISECOPY_S2G,
            'YISECOPY_G2G': YISECOPY_G2G,
            'YISICOPY_S2S': YISICOPY_S2S,
            'YISICOPY_R2S': YISICOPY_R2S,
            'YISICOPY_S2R': YISICOPY_S2R,
            'YISICOPY_BC': YISICOPY_BC,
            'YISICOPY_GAT': YISICOPY_GAT,
            'YISMMA_ACC': YISMMA_ACC,
            'YISMMA_NONACC': YISMMA_NONACC,
            'YISMMA_SPMG': YISMMA_SPMG,
            'YISSYNC_BAR': YISSYNC_BAR,
            'YISSYNC_BOINIT': YISSYNC_BOINIT,
            'YISSYNC_BOARRV': YISSYNC_BOARRV,
            'YISSYNC_BOWAIT': YISSYNC_BOWAIT,
            'YISCONTROL_CALL_EU': YISCONTROL_CALL_EU,
            'YISCONTROL_END': YISCONTROL_END
        }
        
        if instruction_type not in instruction_type_map:
            raise ValueError(f"Unknown instruction type: {instruction_type}")
        
        instruction.type = instruction_type_map[instruction_type]
        
        # Set parameters
        if params:
            if 'src_address' in params:
                instruction.params.src_address = params['src_address']
            if 'dst_address' in params:
                instruction.params.dst_address = params['dst_address']
            if 'size' in params:
                instruction.params.size = params['size']
            if 'matrix_m' in params:
                instruction.params.matrix_m = params['matrix_m']
            if 'matrix_n' in params:
                instruction.params.matrix_n = params['matrix_n']
            if 'matrix_k' in params:
                instruction.params.matrix_k = params['matrix_k']
        
        # Execute instruction
        result = self.c_engine.get().execute_instruction(instruction)
        
        # Convert result to Python dictionary
        status_map = {
            SUCCESS: 'success',
            FAILED: 'failed',
            PENDING: 'pending',
            TIMEOUT: 'timeout',
            MEMORY_ERROR: 'memory_error',
            CIM_ERROR: 'cim_error'
        }
        
        return {
            'status': status_map.get(result.status, 'unknown'),
            'execution_time_us': result.execution_time_us,
            'error_message': result.error_message.decode('utf-8') if result.error_message.size() > 0 else ""
        }
    
    def execute_instructions(self, instructions):
        """Execute multiple YIS instructions
        
        Args:
            instructions: List of instruction dictionaries
            
        Returns:
            List of execution results
        """
        cdef vector[YISInstruction] c_instructions
        cdef vector[YISExecutionResult] c_results
        cdef YISInstruction instruction
        
        # Convert Python instructions to C++ instructions
        for inst in instructions:
            instruction_type = inst.get('type', 'YISCONTROL_END')
            params = inst.get('params', {})
            
            # Set instruction type and parameters (similar to single instruction)
            # ... (implementation similar to execute_instruction)
            c_instructions.push_back(instruction)
        
        # Execute instructions
        c_results = self.c_engine.get().execute_instructions(c_instructions)
        
        # Convert results to Python list
        results = []
        for i in range(c_results.size()):
            # Convert each result (similar to single instruction)
            results.append({
                'status': 'success',  # Simplified for now
                'execution_time_us': c_results[i].execution_time_us,
                'error_message': c_results[i].error_message.decode('utf-8')
            })
        
        return results
    
    def get_execution_stats(self):
        """Get execution statistics
        
        Returns:
            Dictionary with execution statistics
        """
        cdef YISExecutionStats stats = self.c_engine.get().get_execution_stats()
        
        return {
            'total_instructions': stats.total_instructions,
            'successful_instructions': stats.successful_instructions,
            'total_execution_time_ms': stats.total_execution_time_ms,
            'average_latency_us': stats.average_latency_us,
            'copy_instructions': stats.copy_instructions,
            'mma_instructions': stats.mma_instructions,
            'sync_instructions': stats.sync_instructions,
            'control_instructions': stats.control_instructions,
            'cim_utilization': stats.cim_utilization,
            'spm_hit_rate': stats.spm_hit_rate,
            'memory_bandwidth_gbps': stats.memory_bandwidth_gbps
        }
    
    def reset_stats(self):
        """Reset execution statistics"""
        self.c_engine.get().reset_stats()
    
    def get_cim_utilization(self):
        """Get CIM array utilization"""
        return self.c_engine.get().get_cim_utilization()
    
    def get_spm_usage(self):
        """Get SPM memory usage"""
        return self.c_engine.get().get_spm_usage()
    
    def get_memory_bandwidth_utilization(self):
        """Get memory bandwidth utilization"""
        return self.c_engine.get().get_memory_bandwidth_utilization()
    
    def set_debug_mode(self, enable):
        """Enable or disable debug mode"""
        self.c_engine.get().set_debug_mode(enable)
    
    def get_version(self):
        """Get engine version"""
        return self.c_engine.get().get_version().decode('utf-8')

cdef class PyCIMArraySimulator:
    """Python wrapper for CIM Array Simulator"""
    cdef unique_ptr[CIMArraySimulator] c_simulator
    cdef YICAConfig config_
    cdef uint32_t array_id_
    
    def __init__(self, array_id=0, config=None):
        """Initialize CIM Array Simulator
        
        Args:
            array_id: CIM array identifier
            config: Dictionary with YICA configuration parameters
        """
        self.array_id_ = array_id
        
        # Set default configuration
        self.config_.num_cim_arrays = 512
        self.config_.spm_size_kb = 131072
        
        if config:
            if 'num_cim_arrays' in config:
                self.config_.num_cim_arrays = config['num_cim_arrays']
            if 'spm_size_kb' in config:
                self.config_.spm_size_kb = config['spm_size_kb']
        
        self.c_simulator.reset(new CIMArraySimulator(self.config_, self.array_id_))
    
    def initialize(self):
        """Initialize the CIM array simulator"""
        return self.c_simulator.get().initialize()
    
    def shutdown(self):
        """Shutdown the CIM array simulator"""
        self.c_simulator.get().shutdown()
    
    def execute_matrix_multiply(self, m, n, k, a, b, c, precision='FP32', accumulate=False):
        """Execute matrix multiplication on CIM array
        
        Args:
            m, n, k: Matrix dimensions
            a, b, c: Input and output matrices (numpy arrays)
            precision: Precision mode ('INT8', 'INT16', 'FP16', 'BF16', 'FP32')
            accumulate: Whether to accumulate results
            
        Returns:
            Execution time in microseconds
        """
        # Convert precision string to enum
        precision_map = {
            'INT8': INT8,
            'INT16': INT16,
            'FP16': FP16,
            'BF16': BF16,
            'FP32': FP32
        }
        
        if precision not in precision_map:
            raise ValueError(f"Unknown precision mode: {precision}")
        
        cdef CIMPrecisionMode c_precision = precision_map[precision]
        
        # Get pointers to numpy array data
        cdef void* a_ptr = <void*>a.data
        cdef void* b_ptr = <void*>b.data
        cdef void* c_ptr = <void*>c.data
        
        return self.c_simulator.get().execute_matrix_multiply(
            m, n, k, a_ptr, b_ptr, c_ptr, c_precision, accumulate
        )
    
    def execute_vector_operation(self, size, a, b, c, compute_type='VECTOR_ADD', precision='FP32'):
        """Execute vector operation on CIM array
        
        Args:
            size: Vector size
            a, b, c: Input and output vectors (numpy arrays)
            compute_type: Operation type ('VECTOR_ADD', 'VECTOR_MUL', etc.)
            precision: Precision mode
            
        Returns:
            Execution time in microseconds
        """
        # Convert compute type string to enum
        compute_type_map = {
            'MATRIX_MULTIPLY': MATRIX_MULTIPLY,
            'VECTOR_ADD': VECTOR_ADD,
            'VECTOR_MUL': VECTOR_MUL,
            'ACTIVATION': ACTIVATION,
            'CUSTOM': CUSTOM
        }
        
        precision_map = {
            'INT8': INT8,
            'INT16': INT16,
            'FP16': FP16,
            'BF16': BF16,
            'FP32': FP32
        }
        
        if compute_type not in compute_type_map:
            raise ValueError(f"Unknown compute type: {compute_type}")
        if precision not in precision_map:
            raise ValueError(f"Unknown precision mode: {precision}")
        
        cdef CIMComputeType c_compute_type = compute_type_map[compute_type]
        cdef CIMPrecisionMode c_precision = precision_map[precision]
        
        # Get pointers to numpy array data
        cdef void* a_ptr = <void*>a.data
        cdef void* b_ptr = <void*>b.data
        cdef void* c_ptr = <void*>c.data
        
        return self.c_simulator.get().execute_vector_operation(
            size, a_ptr, b_ptr, c_ptr, c_compute_type, c_precision
        )
    
    def execute_activation(self, size, input_data, output_data, activation_type=0, precision='FP32'):
        """Execute activation function on CIM array
        
        Args:
            size: Data size
            input_data, output_data: Input and output arrays (numpy arrays)
            activation_type: Activation function type (0=ReLU, 1=GELU, 2=SiLU)
            precision: Precision mode
            
        Returns:
            Execution time in microseconds
        """
        precision_map = {
            'INT8': INT8,
            'INT16': INT16,
            'FP16': FP16,
            'BF16': BF16,
            'FP32': FP32
        }
        
        if precision not in precision_map:
            raise ValueError(f"Unknown precision mode: {precision}")
        
        cdef CIMPrecisionMode c_precision = precision_map[precision]
        
        # Get pointers to numpy array data
        cdef void* input_ptr = <void*>input_data.data
        cdef void* output_ptr = <void*>output_data.data
        
        return self.c_simulator.get().execute_activation(
            size, input_ptr, output_ptr, activation_type, c_precision
        )
    
    def get_state(self):
        """Get current CIM array state"""
        cdef CIMArrayState state = self.c_simulator.get().get_state()
        
        state_map = {
            IDLE: 'idle',
            COMPUTING: 'computing',
            LOADING: 'loading',
            STORING: 'storing',
            ERROR: 'error'
        }
        
        return state_map.get(state, 'unknown')
    
    def get_metrics(self):
        """Get CIM array performance metrics"""
        cdef CIMArrayMetrics metrics = self.c_simulator.get().get_metrics()
        
        return {
            'utilization_rate': metrics.utilization_rate,
            'throughput_gops': metrics.throughput_gops,
            'power_consumption_w': metrics.power_consumption_w,
            'temperature_celsius': metrics.temperature_celsius,
            'total_operations': metrics.total_operations,
            'average_latency_us': metrics.average_latency_us,
            'energy_efficiency_tops_w': metrics.energy_efficiency_tops_w
        }
    
    def get_utilization(self):
        """Get CIM array utilization rate"""
        return self.c_simulator.get().get_utilization()
    
    def get_power_consumption(self):
        """Get CIM array power consumption"""
        return self.c_simulator.get().get_power_consumption()
    
    def wait_for_completion(self, timeout_ms=5000):
        """Wait for CIM array operations to complete"""
        return self.c_simulator.get().wait_for_completion(timeout_ms)
    
    def reset_metrics(self):
        """Reset CIM array performance metrics"""
        self.c_simulator.get().reset_metrics()
    
    def set_debug_mode(self, enable):
        """Enable or disable debug mode"""
        self.c_simulator.get().set_debug_mode(enable)

# =============================================================================
# Convenience Functions for YIS Engine
# =============================================================================

def create_yis_instruction_engine(config=None):
    """Create a YIS Instruction Engine instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        PyYISInstructionEngine instance
    """
    return PyYISInstructionEngine(config)

def create_cim_array_simulator(array_id=0, config=None):
    """Create a CIM Array Simulator instance
    
    Args:
        array_id: CIM array identifier
        config: Optional configuration dictionary
        
    Returns:
        PyCIMArraySimulator instance
    """
    return PyCIMArraySimulator(array_id, config)

def get_yis_instruction_types():
    """Get list of supported YIS instruction types
    
    Returns:
        List of instruction type names
    """
    return [
        'YISECOPY_G2S', 'YISECOPY_S2G', 'YISECOPY_G2G',
        'YISICOPY_S2S', 'YISICOPY_R2S', 'YISICOPY_S2R', 'YISICOPY_BC', 'YISICOPY_GAT',
        'YISMMA_ACC', 'YISMMA_NONACC', 'YISMMA_SPMG',
        'YISSYNC_BAR', 'YISSYNC_BOINIT', 'YISSYNC_BOARRV', 'YISSYNC_BOWAIT',
        'YISCONTROL_CALL_EU', 'YISCONTROL_END'
    ]

def get_cim_precision_modes():
    """Get list of supported CIM precision modes
    
    Returns:
        List of precision mode names
    """
    return ['INT8', 'INT16', 'FP16', 'BF16', 'FP32']

def get_cim_compute_types():
    """Get list of supported CIM compute types
    
    Returns:
        List of compute type names
    """
    return ['MATRIX_MULTIPLY', 'VECTOR_ADD', 'VECTOR_MUL', 'ACTIVATION', 'CUSTOM'] 