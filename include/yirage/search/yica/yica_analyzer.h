#pragma once

#include "yirage/search/yica/yica_types.h"
#include "yirage/kernel/graph.h"
#include "yirage/kernel/device_tensor.h"
#include "yirage/kernel/operator.h"
#include "yirage/type.h"
#include <memory>
#include <map>

namespace yirage {
namespace kernel {
class Graph;
class DTensor;
}
}

namespace yirage {
namespace search {
namespace yica {

/**
 * YICA架构感知分析器
 * 
 * 该类负责分析计算图对YICA存算一体架构的适配性，
 * 包括CIM友好度评估、内存访问模式分析和并行化机会发现。
 */
class YICAArchitectureAnalyzer {
public:
    /**
     * 构造函数
     * @param config YICA架构配置参数
     */
    explicit YICAArchitectureAnalyzer(const YICAConfig& config);
    
    /**
     * 析构函数
     */
    ~YICAArchitectureAnalyzer();
    
    // 禁用拷贝构造和赋值
    YICAArchitectureAnalyzer(const YICAArchitectureAnalyzer&) = delete;
    YICAArchitectureAnalyzer& operator=(const YICAArchitectureAnalyzer&) = delete;
    
    /**
     * 分析计算图的YICA适配性
     * @param graph 要分析的kernel graph
     * @return 分析结果
     */
    AnalysisResult analyze_computation_pattern(const kernel::Graph& graph);
    
    /**
     * 识别CIM友好的操作
     * @param graph 输入计算图
     * @return CIM友好的操作列表
     */
    std::vector<kernel::DTensor*> identify_cim_operations(const kernel::Graph& graph);
    
    /**
     * 分析内存访问模式
     * @param graph 输入计算图
     * @return 内存局部性评分 [0-1]
     */
    float analyze_memory_access_pattern(const kernel::Graph& graph);
    
    /**
     * 发现并行化机会
     * @param graph 输入计算图
     * @return 并行化机会列表
     */
    std::vector<ParallelizationOpportunity> find_parallel_patterns(const kernel::Graph& graph);
    
    /**
     * 更新YICA配置
     * @param config 新的配置参数
     */
    void update_config(const YICAConfig& config);
    
    /**
     * 获取当前配置
     * @return 当前YICA配置
     */
    const YICAConfig& get_config() const;

private:
    /**
     * 计算单个操作的CIM友好度
     * @param op 要分析的操作
     * @return CIM友好度评分 [0-1]
     */
    float calculate_cim_friendliness(const kernel::KNOperator* op);
    
    /**
     * 估计张量的内存访问成本
     * @param tensor 要分析的张量
     * @return 内存访问成本（相对值）
     */
    float estimate_memory_cost(const kernel::DTensor* tensor);
    
    /**
     * 识别操作类型
     * @param op 要分析的操作
     * @return 操作类型
     */
    OpType identify_operation_type(const kernel::KNOperator* op);
    
    /**
     * 获取张量的数据类型
     * @param tensor 要分析的张量
     * @return 数据类型
     */
    DataType get_tensor_data_type(const kernel::DTensor* tensor);
    
    /**
     * 计算张量大小（字节数）
     * @param tensor 要分析的张量
     * @return 张量大小（字节）
     */
    size_t calculate_tensor_size(const kernel::DTensor* tensor);
    
    /**
     * 分析单个张量的内存访问模式
     * @param tensor 要分析的张量
     * @return 内存访问模式
     */
    MemoryAccessPattern analyze_tensor_access_pattern(const kernel::DTensor* tensor);
    
    /**
     * 估算操作的计算复杂度
     * @param op 要分析的操作
     * @return 计算复杂度（FLOPS）
     */
    size_t estimate_computation_complexity(const kernel::KNOperator* op);
    
    /**
     * 检查数据并行机会
     * @param graph 输入计算图
     * @return 数据并行机会列表
     */
    std::vector<ParallelizationOpportunity> find_data_parallel_opportunities(const kernel::Graph& graph);
    
    /**
     * 检查模型并行机会
     * @param graph 输入计算图
     * @return 模型并行机会列表
     */
    std::vector<ParallelizationOpportunity> find_model_parallel_opportunities(const kernel::Graph& graph);

private:
    YICAConfig config_;
    
    // 缓存分析结果以提高性能
    std::map<const kernel::KNOperator*, float> cim_friendliness_cache_;
    std::map<const kernel::DTensor*, MemoryAccessPattern> access_pattern_cache_;
};

} // namespace yica
} // namespace search
} // namespace yirage 