#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace yirage {
namespace yica {

// YICA Architecture Configuration
struct YICAConfig {
    // CIM Array Configuration
    uint32_t num_cim_arrays = 4;
    uint32_t cim_array_rows = 256;
    uint32_t cim_array_cols = 256;
    float cim_frequency_ghz = 1.0f;
    
    // SPM (Scratchpad Memory) Configuration  
    uint64_t spm_size_kb = 512;
    uint64_t spm_size_per_die = 256;  // KB per die
    uint32_t spm_banks = 4;
    float spm_bandwidth_gbps = 500.0f;
    
    // Memory Hierarchy
    uint64_t l1_cache_size_kb = 64;
    uint64_t l2_cache_size_kb = 1024;
    float dram_bandwidth_gbps = 1000.0f;
    
    // Computation Configuration
    bool supports_fp16 = true;
    bool supports_int8 = true;
    bool supports_fp32 = true;
    bool supports_mixed_precision = true;
    
    // Performance Parameters
    float peak_tops = 100.0f;  // Tera Operations Per Second
    float power_budget_watts = 300.0f;
    
    // Optimization Preferences
    bool prefer_cim_utilization = true;
    bool prefer_memory_efficiency = true;
    bool enable_data_reuse = true;
    bool enable_pipeline_optimization = true;
    
    // Validation
    bool is_valid() const {
        return num_cim_arrays > 0 && 
               cim_array_rows > 0 && 
               cim_array_cols > 0 &&
               spm_size_kb > 0;
    }
};

// YICA Operation Types
enum class YICAOpType {
    MATMUL,           // Matrix multiplication (CIM-friendly)
    ELEMENT_WISE,     // Element-wise operations  
    REDUCTION,        // Reduction operations
    CONVOLUTION,      // Convolution operations
    ATTENTION,        // Attention mechanisms
    NORMALIZATION,    // Layer norm, batch norm
    ACTIVATION,       // ReLU, GELU, etc.
    MEMORY_COPY,      // Data movement
    UNKNOWN
};

// YICA Memory Access Pattern
enum class YICAMemoryPattern {
    SEQUENTIAL,       // Sequential access
    STRIDED,          // Strided access
    RANDOM,           // Random access
    BROADCAST,        // Broadcast pattern
    GATHER_SCATTER    // Gather/scatter pattern
};

// YICA Optimization Strategy
struct YICAOptimizationStrategy {
    // CIM Utilization Strategy
    bool maximize_cim_parallelism = true;
    bool balance_cim_load = true;
    float target_cim_utilization = 0.8f;
    
    // Memory Optimization Strategy
    bool optimize_spm_usage = true;
    bool enable_data_prefetch = true;
    bool minimize_dram_access = true;
    
    // Computation Strategy
    bool enable_mixed_precision = false;
    bool optimize_data_reuse = true;
    bool enable_operation_fusion = true;
    
    // Performance vs. Energy Trade-off
    float performance_weight = 0.7f;  // 0.0 = energy-first, 1.0 = performance-first
    float energy_weight = 0.3f;
};

// YICA Performance Metrics
struct YICAPerformanceMetrics {
    // Execution Time
    float execution_time_ms = 0.0f;
    float computation_time_ms = 0.0f;
    float memory_time_ms = 0.0f;
    
    // Utilization
    float cim_utilization = 0.0f;
    float spm_utilization = 0.0f;
    float memory_bandwidth_utilization = 0.0f;
    
    // Energy Consumption
    float total_energy_mj = 0.0f;
    float computation_energy_mj = 0.0f;
    float memory_energy_mj = 0.0f;
    
    // Throughput
    float effective_tops = 0.0f;
    float memory_throughput_gbps = 0.0f;
    
    // Quality Metrics
    float speedup_ratio = 1.0f;
    float energy_efficiency_ratio = 1.0f;
};

// YICA Architecture Analysis Result
struct YICAAnalysisResult {
    YICAOpType op_type = YICAOpType::UNKNOWN;
    YICAMemoryPattern memory_pattern = YICAMemoryPattern::SEQUENTIAL;
    
    // Friendliness Scores (0.0 - 1.0)
    float cim_friendliness = 0.0f;
    float memory_friendliness = 0.0f;
    float parallelization_potential = 0.0f;
    
    // Resource Requirements
    uint64_t required_spm_kb = 0;
    uint32_t required_cim_arrays = 0;
    float estimated_execution_time_ms = 0.0f;
    
    // Optimization Opportunities
    std::vector<std::string> optimization_opportunities;
    std::vector<std::string> potential_bottlenecks;
    
    // Overall Assessment
    float overall_suitability = 0.0f;  // 0.0 = poor fit, 1.0 = perfect fit
    bool is_yica_beneficial = false;
};

// YICA Search Space Configuration
struct YICASearchConfig {
    // Search Parameters
    uint32_t max_search_iterations = 1000;
    float convergence_threshold = 0.01f;
    uint32_t population_size = 50;
    
    // Optimization Objectives
    bool optimize_latency = true;
    bool optimize_throughput = true;
    bool optimize_energy = true;
    bool optimize_memory_usage = false;
    
    // Search Space Constraints
    uint32_t max_cim_arrays_per_op = 4;
    uint64_t max_spm_allocation_kb = 256;
    float max_acceptable_latency_ms = 100.0f;
    
    // Exploration Strategy
    float exploration_rate = 0.2f;
    float mutation_rate = 0.1f;
    bool enable_adaptive_search = true;
};

// YICA Runtime Configuration
struct YICARuntimeConfig {
    // Execution Mode
    bool simulation_mode = true;
    bool profiling_enabled = true;
    bool debug_mode = false;
    
    // Runtime Optimization
    bool enable_dynamic_scheduling = true;
    bool enable_load_balancing = true;
    bool enable_adaptive_precision = false;
    
    // Monitoring
    bool collect_performance_metrics = true;
    bool collect_energy_metrics = false;
    uint32_t monitoring_interval_ms = 100;
    
    // Error Handling
    bool strict_error_checking = true;
    bool fallback_to_cpu = false;
    uint32_t max_retry_attempts = 3;
};

// Forward declarations
class YICAOptimizer;
class YICAArchitectureAnalyzer;
class YICASearchSpace;
class CIMSimulator;
class SPMManager;

// Global configuration instance
extern YICAConfig g_yica_config;

// Utility functions
std::string yica_op_type_to_string(YICAOpType op_type);
YICAOpType string_to_yica_op_type(const std::string& str);
bool is_yica_operation_supported(YICAOpType op_type);
float estimate_yica_speedup(const YICAAnalysisResult& analysis);

} // namespace yica
} // namespace yirage 