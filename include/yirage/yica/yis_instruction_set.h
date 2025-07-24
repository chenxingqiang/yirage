#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "yirage/kernel/graph.h"
#include "yirage/yica/config.h"

namespace yirage {
namespace yica {

// YIS 指令类型枚举
enum class YISInstructionType {
    // 外部拷贝指令
    YISECOPY_G2S,    // Global to SPM
    YISECOPY_S2G,    // SPM to Global  
    YISECOPY_G2G,    // Global to Global
    YISECOPY_IM2COL, // Im2Col transformation
    
    // 内部拷贝指令
    YISICOPY_S2S,    // SPM to SPM
    YISICOPY_R2S,    // Register to SPM
    YISICOPY_S2R,    // SPM to Register
    YISICOPY_BC,     // Broadcast
    YISICOPY_GAT,    // Gather
    
    // 矩阵乘法指令
    YISMMA_ACC,      // Accumulate mode
    YISMMA_NONACC,   // Non-accumulate mode
    YISMMA_SPMG,     // SPM Global mode
    
    // 同步指令
    YISSYNC_BAR,     // Barrier synchronization
    YISSYNC_BOINIT,  // Buffer object init
    YISSYNC_BOARRV,  // Buffer object arrive
    YISSYNC_BOWAIT,  // Buffer object wait
    
    // 控制指令
    YISCONTROL_CALL_EU, // Call execution unit
    YISCONTROL_END      // End kernel
};

// YIS 指令参数结构
struct YISInstructionParams {
    // 通用参数
    std::vector<int> dimensions;      // 操作维度
    std::string data_type;            // 数据类型
    std::string layout;               // 数据布局
    
    // 地址参数
    std::string src_address;          // 源地址
    std::string dst_address;          // 目标地址
    int offset = 0;                   // 地址偏移
    
    // 矩阵乘法参数
    struct MatrixParams {
        int M, N, K;                  // 矩阵维度
        bool accumulate = false;      // 是否累加
        std::string precision;        // 计算精度
    } matrix;
    
    // 同步参数
    struct SyncParams {
        int sync_id;                  // 同步ID
        std::string scope;            // 同步范围 (WG/CWG/SWG)
    } sync;
    
    // 控制参数
    struct ControlParams {
        int function_id;              // 函数ID
        std::vector<std::string> args; // 参数列表
    } control;
};

// YIS 指令生成器
class YISInstructionSet {
public:
    explicit YISInstructionSet(const YICAConfig& config);
    ~YISInstructionSet() = default;
    
    // 主要接口：为内核操作生成 YIS 指令
    std::string generate_instruction(
        YISInstructionType type,
        const YISInstructionParams& params
    );
    
    // 为 Yirage 操作生成 YIS 指令序列
    std::vector<std::string> generate_for_operation(
        const kernel::KNOperator* op
    );
    
    // 生成完整的 YIS 内核
    struct YISKernel {
        std::string kernel_name;
        std::vector<std::string> instructions;
        std::map<std::string, std::string> register_allocations;
        size_t spm_requirement;
        std::string metadata;
    };
    
    YISKernel generate_kernel(
        const kernel::Graph& graph,
        const std::string& kernel_name
    );
    
    // 优化 YIS 指令序列
    std::vector<std::string> optimize_instruction_sequence(
        const std::vector<std::string>& instructions
    );

private:
    YICAConfig config_;
    
    // 指令生成方法
    std::string generate_copy_instruction(
        YISInstructionType type,
        const YISInstructionParams& params
    );
    
    std::string generate_mma_instruction(
        const YISInstructionParams& params
    );
    
    std::string generate_sync_instruction(
        YISInstructionType type,
        const YISInstructionParams& params
    );
    
    std::string generate_control_instruction(
        YISInstructionType type,
        const YISInstructionParams& params
    );
    
    // 辅助方法
    std::string format_address(const std::string& base, int offset);
    std::string get_data_type_suffix(const std::string& data_type);
    std::string get_layout_suffix(const std::string& layout);
    std::string get_scope_suffix(const std::string& scope);
    
    // 寄存器分配
    class RegisterAllocator {
    public:
        std::string allocate_register(const std::string& purpose);
        void free_register(const std::string& reg_name);
        std::map<std::string, std::string> get_allocation_map();
        
    private:
        int next_register_id_ = 0;
        std::map<std::string, std::string> allocated_registers_;
        std::vector<std::string> free_registers_;
    };
    
    std::unique_ptr<RegisterAllocator> register_allocator_;
    
    // 指令优化器
    class InstructionOptimizer {
    public:
        std::vector<std::string> optimize(
            const std::vector<std::string>& instructions
        );
        
    private:
        // 优化策略
        std::vector<std::string> eliminate_redundant_copies(
            const std::vector<std::string>& instructions
        );
        
        std::vector<std::string> fuse_memory_operations(
            const std::vector<std::string>& instructions
        );
        
        std::vector<std::string> reorder_for_pipeline(
            const std::vector<std::string>& instructions
        );
    };
    
    std::unique_ptr<InstructionOptimizer> instruction_optimizer_;
};

// YIS 指令模板库
class YISInstructionTemplates {
public:
    // 预定义的指令模板
    static const std::map<YISInstructionType, std::string> INSTRUCTION_TEMPLATES;
    
    // 常用操作的指令序列模板
    struct OperationTemplate {
        std::string name;
        std::vector<YISInstructionType> instruction_sequence;
        std::map<std::string, std::string> parameter_mappings;
    };
    
    static const std::vector<OperationTemplate> OPERATION_TEMPLATES;
    
    // 获取指令模板
    static std::string get_instruction_template(YISInstructionType type);
    
    // 获取操作模板
    static const OperationTemplate* get_operation_template(const std::string& op_name);
    
private:
    static void initialize_templates();
    static bool templates_initialized_;
};

// Yirage 操作到 YIS 指令的映射器
class YirageToYISMapper {
public:
    explicit YirageToYISMapper(const YICAConfig& config);
    
    // 映射 Yirage 操作到 YIS 指令
    struct MappingResult {
        std::vector<YISInstructionType> instruction_types;
        std::vector<YISInstructionParams> instruction_params;
        bool mapping_successful;
        std::string error_message;
        float estimated_performance_gain;
    };
    
    MappingResult map_operation(const kernel::KNOperator* op);
    
    // 检查操作是否支持 YICA 优化
    bool is_yica_optimizable(const kernel::KNOperator* op);
    
    // 获取操作的 CIM 友好度评分
    float get_cim_friendliness_score(const kernel::KNOperator* op);

private:
    YICAConfig config_;
    
    // 具体操作映射方法
    MappingResult map_matmul_operation(const kernel::KNOperator* op);
    MappingResult map_elementwise_operation(const kernel::KNOperator* op);
    MappingResult map_reduction_operation(const kernel::KNOperator* op);
    MappingResult map_memory_operation(const kernel::KNOperator* op);
    
    // 参数提取和转换
    YISInstructionParams extract_params_from_operation(const kernel::KNOperator* op);
    std::string convert_yirage_dtype_to_yis(type::DataType dtype);
    std::string convert_yirage_layout_to_yis(const layout::DmemLayout& layout);
};

// YIS 代码生成上下文
struct YISGenerationContext {
    // 当前生成状态
    std::string current_kernel_name;
    std::map<std::string, std::string> tensor_to_address_map;
    std::vector<std::string> generated_instructions;
    
    // 资源使用情况
    size_t spm_usage = 0;
    size_t register_usage = 0;
    std::map<std::string, bool> cim_array_usage;
    
    // 性能统计
    size_t instruction_count = 0;
    float estimated_execution_time = 0.0f;
    
    // 调试信息
    std::vector<std::string> debug_comments;
    bool enable_debug_output = false;
};

} // namespace yica
} // namespace yirage 