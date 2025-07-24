#pragma once

#include "yirage/yica/config.h"
#include "yirage/search/graph.h"
#include "yirage/transpiler/transpiler.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace yirage {
namespace yica {

// Forward declarations from Yirage
namespace search {
    class Graph;
    class Operator;
}
namespace transpiler {
    class TranspileResult;
}

class YICAOptimizer {
public:
    explicit YICAOptimizer(const YICAConfig& config = g_yica_config);
    ~YICAOptimizer();

    // Core Optimization Interface
    YICAAnalysisResult analyze_graph(const search::Graph& graph);
    YICAPerformanceMetrics optimize_graph(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy = {}
    );
    
    // Triton Code Generation
    std::string generate_triton_kernel(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy = {}
    );
    
    // Advanced Optimization
    search::Graph apply_yica_transformations(const search::Graph& graph);
    std::vector<YICAAnalysisResult> analyze_operators(
        const std::vector<search::Operator*>& operators
    );
    
    // Configuration Management
    void set_config(const YICAConfig& config);
    const YICAConfig& get_config() const;
    void set_search_config(const YICASearchConfig& config);
    const YICASearchConfig& get_search_config() const;
    
    // Performance Profiling
    void enable_profiling(bool enable = true);
    YICAPerformanceMetrics get_last_performance_metrics() const;
    std::vector<YICAPerformanceMetrics> get_performance_history() const;
    
    // Runtime Management
    void initialize_runtime(const YICARuntimeConfig& runtime_config = {});
    void shutdown_runtime();
    bool is_runtime_initialized() const;
    
    // Debugging and Visualization
    std::string dump_analysis_report(const YICAAnalysisResult& result) const;
    std::string dump_optimization_summary(const YICAPerformanceMetrics& metrics) const;
    void export_optimization_trace(const std::string& filename) const;

private:
    // Core Components
    std::unique_ptr<YICAArchitectureAnalyzer> analyzer_;
    std::unique_ptr<YICASearchSpace> search_space_;
    std::unique_ptr<CIMSimulator> cim_simulator_;
    std::unique_ptr<SPMManager> spm_manager_;
    
    // Configuration
    YICAConfig config_;
    YICASearchConfig search_config_;
    YICARuntimeConfig runtime_config_;
    
    // State Management
    bool runtime_initialized_;
    bool profiling_enabled_;
    
    // Performance Tracking
    mutable YICAPerformanceMetrics last_metrics_;
    mutable std::vector<YICAPerformanceMetrics> performance_history_;
    
    // Internal Methods
    void validate_config() const;
    void initialize_components();
    YICAAnalysisResult analyze_single_operator(const search::Operator& op);
    std::string generate_cim_optimized_code(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy
    );
    std::string generate_spm_optimized_code(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy
    );
};

// YICA Architecture Analyzer
class YICAArchitectureAnalyzer {
public:
    explicit YICAArchitectureAnalyzer(const YICAConfig& config);
    
    // Analysis Methods
    YICAAnalysisResult analyze_operation(const search::Operator& op);
    YICAAnalysisResult analyze_graph(const search::Graph& graph);
    
    // Operator Classification
    YICAOpType classify_operator(const search::Operator& op);
    YICAMemoryPattern analyze_memory_pattern(const search::Operator& op);
    
    // Resource Estimation
    uint64_t estimate_spm_requirement(const search::Operator& op);
    uint32_t estimate_cim_requirement(const search::Operator& op);
    float estimate_execution_time(const search::Operator& op);
    
    // Optimization Opportunity Detection
    std::vector<std::string> identify_optimization_opportunities(
        const search::Operator& op
    );
    std::vector<std::string> identify_bottlenecks(const search::Operator& op);
    
    // Friendliness Scoring
    float compute_cim_friendliness(const search::Operator& op);
    float compute_memory_friendliness(const search::Operator& op);
    float compute_parallelization_potential(const search::Operator& op);

private:
    YICAConfig config_;
    
    // Analysis Helper Methods
    bool is_matrix_operation(const search::Operator& op);
    bool is_element_wise_operation(const search::Operator& op);
    bool has_regular_memory_access(const search::Operator& op);
    float estimate_computational_intensity(const search::Operator& op);
};

// YICA Search Space Manager
class YICASearchSpace {
public:
    explicit YICASearchSpace(const YICAConfig& config, const YICASearchConfig& search_config);
    
    // Search Space Generation
    std::vector<YICAOptimizationStrategy> generate_search_space(
        const YICAAnalysisResult& analysis
    );
    
    // Optimization Search
    YICAOptimizationStrategy find_optimal_strategy(
        const search::Graph& graph,
        const YICAAnalysisResult& analysis
    );
    
    // Strategy Evaluation
    float evaluate_strategy(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy
    );
    
    // Search Configuration
    void set_search_constraints(const YICASearchConfig& config);
    void set_optimization_objectives(
        bool latency, bool throughput, bool energy, bool memory
    );

private:
    YICAConfig config_;
    YICASearchConfig search_config_;
    
    // Search State
    std::vector<YICAOptimizationStrategy> candidate_strategies_;
    std::unordered_map<size_t, float> strategy_scores_;
    
    // Search Methods
    std::vector<YICAOptimizationStrategy> generate_initial_population(
        const YICAAnalysisResult& analysis
    );
    YICAOptimizationStrategy mutate_strategy(const YICAOptimizationStrategy& strategy);
    YICAOptimizationStrategy crossover_strategies(
        const YICAOptimizationStrategy& parent1,
        const YICAOptimizationStrategy& parent2
    );
    bool is_strategy_feasible(
        const YICAOptimizationStrategy& strategy,
        const YICAAnalysisResult& analysis
    );
};

// YICA Integration with Yirage Transpiler
class YICATritonTranspiler {
public:
    explicit YICATritonTranspiler(const YICAConfig& config);
    
    // Code Generation
    std::string transpile_to_yica_triton(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy
    );
    
    // Code Templates
    std::string generate_cim_matmul_kernel(
        const search::Operator& matmul_op,
        const YICAOptimizationStrategy& strategy
    );
    std::string generate_spm_optimized_kernel(
        const search::Graph& graph,
        const YICAOptimizationStrategy& strategy
    );
    
    // Optimization Passes
    std::string apply_cim_optimizations(const std::string& base_code);
    std::string apply_memory_optimizations(const std::string& base_code);
    std::string apply_parallelization_optimizations(const std::string& base_code);

private:
    YICAConfig config_;
    
    // Code Generation Helpers
    std::string generate_cim_array_dispatch(uint32_t num_arrays);
    std::string generate_spm_memory_management();
    std::string generate_load_balancing_code();
};

// Utility Functions
namespace utils {
    // Performance Estimation
    float estimate_yica_vs_gpu_speedup(const YICAAnalysisResult& analysis);
    float estimate_energy_efficiency_gain(const YICAAnalysisResult& analysis);
    
    // Configuration Helpers
    YICAConfig create_default_yica_config();
    YICAOptimizationStrategy create_performance_first_strategy();
    YICAOptimizationStrategy create_energy_efficient_strategy();
    
    // Validation
    bool validate_yica_config(const YICAConfig& config);
    bool validate_optimization_strategy(const YICAOptimizationStrategy& strategy);
    
    // Reporting
    std::string format_analysis_result(const YICAAnalysisResult& result);
    std::string format_performance_metrics(const YICAPerformanceMetrics& metrics);
}

} // namespace yica
} // namespace yirage 