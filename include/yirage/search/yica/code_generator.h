#pragma once

#include "yirage/search/yica/yica_types.h"
#include "yirage/kernel/graph.h"
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace yirage {
namespace search {
namespace yica {

/**
 * 代码生成目标类型
 */
enum class CodeGenTarget {
    CUDA_KERNEL,        // CUDA内核
    YICA_NATIVE,        // YICA原生代码
    OPENCL_KERNEL,      // OpenCL内核
    CPU_SIMD           // CPU SIMD代码
};

/**
 * 优化级别
 */
enum class OptimizationLevel {
    O0,    // 无优化
    O1,    // 基本优化
    O2,    // 标准优化
    O3     // 激进优化
};

/**
 * 模板类型
 */
enum class TemplateType {
    CIM_KERNEL,         // CIM内核模板
    SPM_MANAGEMENT,     // SPM管理模板
    HOST_INTERFACE,     // 主机接口模板
    RUNTIME_SUPPORT     // 运行时支持模板
};

/**
 * 指令类型
 */
enum class InstructionType {
    CIM_MATMUL,         // CIM矩阵乘法
    CIM_ELEMENTWISE,    // CIM逐元素操作
    SPM_LOAD,           // SPM加载
    SPM_STORE,          // SPM存储
    SYNC,               // 同步
    BRANCH              // 分支
};

/**
 * 生成的文件信息
 */
struct GeneratedFile {
    std::string filename;           // 文件名
    std::string content;           // 文件内容
    std::string file_type;         // 文件类型 (.cu, .h, .cpp等)
    size_t size_bytes = 0;         // 文件大小
    std::string description;       // 文件描述
};

/**
 * CIM指令结构
 */
struct CIMInstruction {
    InstructionType type;                    // 指令类型
    std::vector<int> operand_addresses;     // 操作数地址
    int result_address = -1;                // 结果地址
    int target_cim_array = -1;             // 目标CIM阵列
    std::map<std::string, int> parameters;  // 指令参数
    std::string assembly_code;              // 汇编代码
};

/**
 * 生成上下文
 */
struct GenerationContext {
    YICAConfig target_config;              // 目标配置
    std::map<std::string, int> variable_map; // 变量映射
    std::vector<CIMInstruction> instructions; // 指令序列
    size_t current_spm_offset = 0;         // 当前SPM偏移
    int next_temp_var_id = 0;              // 下一个临时变量ID
};

/**
 * 代码生成配置
 */
struct GenerationConfig {
    YICAConfig target_config;              // 目标YICA配置
    CodeGenTarget target = CodeGenTarget::CUDA_KERNEL; // 生成目标
    OptimizationLevel opt_level = OptimizationLevel::O2; // 优化级别
    bool enable_debug_info = false;        // 调试信息
    std::string output_directory = "./";   // 输出目录
    bool enable_profiling = false;         // 性能分析
    bool enable_assertions = true;         // 断言检查
};

/**
 * 代码生成结果
 */
struct GenerationResult {
    bool success = false;
    std::vector<GeneratedFile> generated_files;
    std::string compilation_commands;       // 编译命令
    float estimated_performance_gain = 0.0f; // 预估性能提升
    std::string generation_log;            // 生成日志
    std::map<std::string, float> metrics;  // 生成指标
    
    // 性能预估
    float estimated_latency_ms = 0.0f;     // 预估延迟
    float estimated_energy_mj = 0.0f;      // 预估能耗
    float spm_utilization = 0.0f;          // SPM利用率
};

/**
 * 操作生成器基类
 */
class OperatorGenerator {
public:
    virtual ~OperatorGenerator() = default;
    
    /**
     * 检查是否可以生成指定操作的代码
     */
    virtual bool can_generate(const kernel::KNOperator* op) const = 0;
    
    /**
     * 生成操作的代码
     */
    virtual GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const = 0;
    
    /**
     * 获取生成器名称
     */
    virtual std::string get_name() const = 0;
};

/**
 * 代码模板
 */
struct CodeTemplate {
    std::string name;                       // 模板名称
    std::string content;                    // 模板内容
    std::vector<std::string> parameters;   // 参数列表
    TemplateType type;                      // 模板类型
    std::string description;                // 模板描述
};

/**
 * 代码模板管理器
 */
class CodeTemplateManager {
public:
    CodeTemplateManager();
    ~CodeTemplateManager() = default;
    
    /**
     * 注册模板
     */
    void register_template(const CodeTemplate& tmpl);
    
    /**
     * 实例化模板
     */
    std::string instantiate_template(
        const std::string& name,
        const std::map<std::string, std::string>& params) const;
    
    /**
     * 获取模板列表
     */
    std::vector<std::string> get_template_names() const;
    
    /**
     * 检查模板是否存在
     */
    bool has_template(const std::string& name) const;

private:
    std::map<std::string, CodeTemplate> templates_;
    void initialize_default_templates();
    std::string replace_parameters(
        const std::string& content,
        const std::map<std::string, std::string>& params) const;
};

/**
 * CIM代码生成算法
 */
class CIMCodeGenAlgorithm {
public:
    CIMCodeGenAlgorithm() = default;
    ~CIMCodeGenAlgorithm() = default;
    
    /**
     * 生成CIM指令序列
     */
    std::vector<CIMInstruction> generate_cim_instructions(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    /**
     * 优化指令序列
     */
    void optimize_instructions(
        std::vector<CIMInstruction>& instructions,
        OptimizationLevel level) const;
    
    /**
     * 生成指令的汇编代码
     */
    std::string generate_assembly(
        const std::vector<CIMInstruction>& instructions) const;

private:
    std::string instruction_to_assembly(const CIMInstruction& inst) const;
    void apply_instruction_scheduling(std::vector<CIMInstruction>& instructions) const;
    void apply_register_allocation(std::vector<CIMInstruction>& instructions) const;
};

/**
 * YICA代码生成器主类
 */
class YICACodeGenerator {
public:
    using OperatorGeneratorPtr = std::unique_ptr<OperatorGenerator>;
    
    /**
     * 构造函数
     */
    YICACodeGenerator();
    
    /**
     * 析构函数
     */
    ~YICACodeGenerator();
    
    // 禁用拷贝构造和赋值
    YICACodeGenerator(const YICACodeGenerator&) = delete;
    YICACodeGenerator& operator=(const YICACodeGenerator&) = delete;
    
    /**
     * 核心代码生成接口
     */
    GenerationResult generate_code(
        const kernel::Graph& optimized_graph,
        const AnalysisResult& analysis,
        const GenerationConfig& config);
    
    /**
     * 端到端YICA内核生成
     */
    GenerationResult generate_yica_kernel(
        const kernel::Graph& graph,
        const YICAConfig& config);
    
    /**
     * 分模块生成接口
     */
    GenerationResult generate_cim_kernels(
        const std::vector<kernel::KNOperator*>& ops,
        const GenerationConfig& config);
    
    GenerationResult generate_memory_management(
        const AnalysisResult& analysis,
        const GenerationConfig& config);
    
    GenerationResult generate_host_interface(
        const kernel::Graph& graph,
        const GenerationConfig& config);
    
    /**
     * 操作生成器管理
     */
    void register_operator_generator(OperatorGeneratorPtr generator);
    void unregister_operator_generator(const std::string& name);
    
    /**
     * 配置管理
     */
    void set_default_config(const GenerationConfig& config);
    GenerationConfig get_default_config() const;
    
    /**
     * 模板管理
     */
    CodeTemplateManager& get_template_manager();
    const CodeTemplateManager& get_template_manager() const;

private:
    std::map<std::string, OperatorGeneratorPtr> operator_generators_;
    std::unique_ptr<CodeTemplateManager> template_manager_;
    std::unique_ptr<CIMCodeGenAlgorithm> cim_algorithm_;
    GenerationConfig default_config_;
    
    // 内部生成方法
    GenerationResult generate_kernel_file(
        const kernel::Graph& graph,
        const GenerationContext& context);
    
    GenerationResult generate_header_file(
        const kernel::Graph& graph,
        const GenerationContext& context);
    
    GenerationResult generate_host_file(
        const kernel::Graph& graph,
        const GenerationContext& context);
    
    GenerationResult generate_makefile(
        const GenerationConfig& config);
    
    // 代码分析和优化
    void analyze_graph_for_generation(
        const kernel::Graph& graph,
        GenerationContext& context);
    
    void optimize_generated_code(
        std::vector<GeneratedFile>& files,
        OptimizationLevel level);
    
    // 性能估算
    float estimate_performance_gain(
        const kernel::Graph& graph,
        const GenerationResult& result) const;
    
    // 初始化
    void initialize_default_generators();
    void initialize_builtin_templates();
    
    // 验证和错误检查
    bool validate_generation_input(
        const kernel::Graph& graph,
        const GenerationConfig& config,
        std::string& error_message);
    
    bool validate_generated_code(
        const std::vector<GeneratedFile>& files,
        std::string& error_message);
};

} // namespace yica
} // namespace search
} // namespace yirage 