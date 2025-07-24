#include "yirage/search/yica/code_generator.h"
#include "yirage/search/yica/operator_generators.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <regex>
#include <iomanip>

namespace yirage {
namespace search {
namespace yica {

// ===== CodeTemplateManager Implementation =====

CodeTemplateManager::CodeTemplateManager() {
    initialize_default_templates();
}

void CodeTemplateManager::initialize_default_templates() {
    // CIM内核模板
    CodeTemplate cim_kernel_template;
    cim_kernel_template.name = "cim_kernel_base";
    cim_kernel_template.type = TemplateType::CIM_KERNEL;
    cim_kernel_template.content = R"(
__global__ void {KERNEL_NAME}(
    {INPUT_PARAMETERS}
) {
    // CIM Array Configuration
    const int cim_array_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    
    // SPM allocation
    __shared__ float spm_buffer[{SPM_SIZE}];
    
    {KERNEL_BODY}
    
    __syncthreads();
}
)";
    cim_kernel_template.parameters = {"KERNEL_NAME", "INPUT_PARAMETERS", "SPM_SIZE", "KERNEL_BODY"};
    cim_kernel_template.description = "Base template for CIM kernel generation";
    register_template(cim_kernel_template);
    
    // SPM管理模板
    CodeTemplate spm_template;
    spm_template.name = "spm_management";
    spm_template.type = TemplateType::SPM_MANAGEMENT;
    spm_template.content = R"(
// SPM Memory Management
template<typename T>
__device__ void spm_load(T* spm_addr, const T* global_addr, int size) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = tid; i < size; i += stride) {
        spm_addr[i] = global_addr[i];
    }
    __syncthreads();
}

template<typename T>
__device__ void spm_store(T* global_addr, const T* spm_addr, int size) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = tid; i < size; i += stride) {
        global_addr[i] = spm_addr[i];
    }
    __syncthreads();
}
)";
    spm_template.parameters = {};
    spm_template.description = "SPM memory management functions";
    register_template(spm_template);
    
    // 主机接口模板
    CodeTemplate host_template;
    host_template.name = "host_interface";
    host_template.type = TemplateType::HOST_INTERFACE;
    host_template.content = R"(
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

class {CLASS_NAME} {
public:
    {CLASS_NAME}(const YICAConfig& config) : config_(config) {
        allocate_device_memory();
    }
    
    ~{CLASS_NAME}() {
        deallocate_device_memory();
    }
    
    void execute({INPUT_TYPES}) {
        // Copy input to device
        {COPY_TO_DEVICE}
        
        // Launch kernel
        {KERNEL_LAUNCH}
        
        // Copy output from device
        {COPY_FROM_DEVICE}
        
        // Synchronize
        cudaDeviceSynchronize();
    }

private:
    YICAConfig config_;
    {DEVICE_MEMORY_DECLARATIONS}
    
    void allocate_device_memory() {
        {MEMORY_ALLOCATION}
    }
    
    void deallocate_device_memory() {
        {MEMORY_DEALLOCATION}
    }
};
)";
    host_template.parameters = {"CLASS_NAME", "INPUT_TYPES", "COPY_TO_DEVICE", 
                               "KERNEL_LAUNCH", "COPY_FROM_DEVICE", 
                               "DEVICE_MEMORY_DECLARATIONS", "MEMORY_ALLOCATION", 
                               "MEMORY_DEALLOCATION"};
    host_template.description = "Host interface class template";
    register_template(host_template);
}

void CodeTemplateManager::register_template(const CodeTemplate& tmpl) {
    templates_[tmpl.name] = tmpl;
}

std::string CodeTemplateManager::instantiate_template(
    const std::string& name,
    const std::map<std::string, std::string>& params) const {
    
    auto it = templates_.find(name);
    if (it == templates_.end()) {
        return ""; // 模板不存在
    }
    
    return replace_parameters(it->second.content, params);
}

std::vector<std::string> CodeTemplateManager::get_template_names() const {
    std::vector<std::string> names;
    names.reserve(templates_.size());
    
    for (const auto& [name, tmpl] : templates_) {
        names.push_back(name);
    }
    
    return names;
}

bool CodeTemplateManager::has_template(const std::string& name) const {
    return templates_.find(name) != templates_.end();
}

std::string CodeTemplateManager::replace_parameters(
    const std::string& content,
    const std::map<std::string, std::string>& params) const {
    
    std::string result = content;
    
    for (const auto& [key, value] : params) {
        std::string placeholder = "{" + key + "}";
        size_t pos = 0;
        
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), value);
            pos += value.length();
        }
    }
    
    return result;
}

// ===== CIMCodeGenAlgorithm Implementation =====

std::vector<CIMInstruction> CIMCodeGenAlgorithm::generate_cim_instructions(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::vector<CIMInstruction> instructions;
    
    if (!op) {
        return instructions;
    }
    
    // 根据操作类型生成不同的指令序列
    if (op->op_type == kernel::KNOperatorType::KN_MATMUL_OP) {
        // 生成矩阵乘法指令
        CIMInstruction matmul_inst;
        matmul_inst.type = InstructionType::CIM_MATMUL;
        matmul_inst.target_cim_array = 0; // 简化：使用第一个CIM阵列
        matmul_inst.operand_addresses = {0, 1}; // 输入操作数地址
        matmul_inst.result_address = 2; // 输出地址
        matmul_inst.assembly_code = "cim.matmul r2, r0, r1";
        instructions.push_back(matmul_inst);
        
    } else if (op->op_type == kernel::KNOperatorType::KN_EW_ADD_OP ||
               op->op_type == kernel::KNOperatorType::KN_EW_MUL_OP) {
        // 生成逐元素操作指令
        CIMInstruction ew_inst;
        ew_inst.type = InstructionType::CIM_ELEMENTWISE;
        ew_inst.target_cim_array = 0;
        ew_inst.operand_addresses = {0, 1};
        ew_inst.result_address = 2;
        
        if (op->op_type == kernel::KNOperatorType::KN_EW_ADD_OP) {
            ew_inst.assembly_code = "cim.add r2, r0, r1";
        } else {
            ew_inst.assembly_code = "cim.mul r2, r0, r1";
        }
        instructions.push_back(ew_inst);
    }
    
    return instructions;
}

void CIMCodeGenAlgorithm::optimize_instructions(
    std::vector<CIMInstruction>& instructions,
    OptimizationLevel level) const {
    
    if (level == OptimizationLevel::O0) {
        return; // 无优化
    }
    
    // 基本优化：指令调度
    if (level >= OptimizationLevel::O1) {
        apply_instruction_scheduling(instructions);
    }
    
    // 标准优化：寄存器分配
    if (level >= OptimizationLevel::O2) {
        apply_register_allocation(instructions);
    }
    
    // 激进优化：更多优化
    if (level >= OptimizationLevel::O3) {
        // 可以添加更激进的优化，如指令融合等
    }
}

std::string CIMCodeGenAlgorithm::generate_assembly(
    const std::vector<CIMInstruction>& instructions) const {
    
    std::stringstream ss;
    ss << "// Generated CIM Assembly Code\n";
    ss << ".cim_kernel:\n";
    
    for (size_t i = 0; i < instructions.size(); ++i) {
        ss << "  " << instruction_to_assembly(instructions[i]) << "\n";
    }
    
    ss << "  ret\n";
    return ss.str();
}

std::string CIMCodeGenAlgorithm::instruction_to_assembly(const CIMInstruction& inst) const {
    if (!inst.assembly_code.empty()) {
        return inst.assembly_code;
    }
    
    // 生成默认汇编代码
    switch (inst.type) {
        case InstructionType::CIM_MATMUL:
            return "cim.matmul r" + std::to_string(inst.result_address) + 
                   ", r" + std::to_string(inst.operand_addresses[0]) +
                   ", r" + std::to_string(inst.operand_addresses[1]);
        
        case InstructionType::CIM_ELEMENTWISE:
            return "cim.ew r" + std::to_string(inst.result_address) + 
                   ", r" + std::to_string(inst.operand_addresses[0]) +
                   ", r" + std::to_string(inst.operand_addresses[1]);
        
        case InstructionType::SPM_LOAD:
            return "spm.load r" + std::to_string(inst.result_address) + 
                   ", [" + std::to_string(inst.operand_addresses[0]) + "]";
        
        case InstructionType::SPM_STORE:
            return "spm.store [" + std::to_string(inst.result_address) + 
                   "], r" + std::to_string(inst.operand_addresses[0]);
        
        case InstructionType::SYNC:
            return "sync.barrier";
        
        case InstructionType::BRANCH:
            return "branch L" + std::to_string(inst.operand_addresses[0]);
        
        default:
            return "nop";
    }
}

void CIMCodeGenAlgorithm::apply_instruction_scheduling(
    std::vector<CIMInstruction>& instructions) const {
    
    // 简化的指令调度：将内存操作提前
    std::stable_sort(instructions.begin(), instructions.end(),
                     [](const CIMInstruction& a, const CIMInstruction& b) {
                         int priority_a = (a.type == InstructionType::SPM_LOAD) ? 0 : 1;
                         int priority_b = (b.type == InstructionType::SPM_LOAD) ? 0 : 1;
                         return priority_a < priority_b;
                     });
}

void CIMCodeGenAlgorithm::apply_register_allocation(
    std::vector<CIMInstruction>& instructions) const {
    
    // 简化的寄存器分配：重用寄存器
    std::map<int, int> register_map;
    int next_register = 0;
    
    for (auto& inst : instructions) {
        // 为操作数分配寄存器
        for (auto& addr : inst.operand_addresses) {
            if (register_map.find(addr) == register_map.end()) {
                register_map[addr] = next_register++;
            }
            addr = register_map[addr];
        }
        
        // 为结果分配寄存器
        if (inst.result_address >= 0) {
            if (register_map.find(inst.result_address) == register_map.end()) {
                register_map[inst.result_address] = next_register++;
            }
            inst.result_address = register_map[inst.result_address];
        }
    }
}

// ===== YICACodeGenerator Implementation =====

YICACodeGenerator::YICACodeGenerator() 
    : template_manager_(std::make_unique<CodeTemplateManager>()),
      cim_algorithm_(std::make_unique<CIMCodeGenAlgorithm>()) {
    
    initialize_default_generators();
    initialize_builtin_templates();
}

YICACodeGenerator::~YICACodeGenerator() = default;

void YICACodeGenerator::initialize_default_generators() {
    // 注册默认的操作生成器
    register_operator_generator(std::make_unique<MatmulOperatorGenerator>());
    register_operator_generator(std::make_unique<ElementwiseOperatorGenerator>());
    register_operator_generator(std::make_unique<ConvolutionOperatorGenerator>());
    register_operator_generator(std::make_unique<NormalizationOperatorGenerator>());
    register_operator_generator(std::make_unique<ActivationOperatorGenerator>());
    register_operator_generator(std::make_unique<MemoryOperatorGenerator>());
}

void YICACodeGenerator::initialize_builtin_templates() {
    // 内置模板已在CodeTemplateManager中初始化
}

GenerationResult YICACodeGenerator::generate_code(
    const kernel::Graph& optimized_graph,
    const AnalysisResult& analysis,
    const GenerationConfig& config) {
    
    GenerationResult result;
    std::stringstream log;
    
    try {
        // 1. 验证输入
        std::string error_message;
        if (!validate_generation_input(optimized_graph, config, error_message)) {
            result.success = false;
            result.generation_log = "Input validation failed: " + error_message;
            return result;
        }
        
        log << "Starting code generation for graph with " 
            << optimized_graph.operators.size() << " operators\n";
        
        // 2. 创建生成上下文
        GenerationContext context;
        context.target_config = config.target_config;
        analyze_graph_for_generation(optimized_graph, context);
        
        // 3. 生成各个文件
        auto kernel_result = generate_kernel_file(optimized_graph, context);
        if (kernel_result.success) {
            result.generated_files.insert(result.generated_files.end(),
                                        kernel_result.generated_files.begin(),
                                        kernel_result.generated_files.end());
        }
        
        auto header_result = generate_header_file(optimized_graph, context);
        if (header_result.success) {
            result.generated_files.insert(result.generated_files.end(),
                                        header_result.generated_files.begin(),
                                        header_result.generated_files.end());
        }
        
        auto host_result = generate_host_file(optimized_graph, context);
        if (host_result.success) {
            result.generated_files.insert(result.generated_files.end(),
                                        host_result.generated_files.begin(),
                                        host_result.generated_files.end());
        }
        
        auto makefile_result = generate_makefile(config);
        if (makefile_result.success) {
            result.generated_files.insert(result.generated_files.end(),
                                        makefile_result.generated_files.begin(),
                                        makefile_result.generated_files.end());
        }
        
        // 4. 优化生成的代码
        if (config.opt_level > OptimizationLevel::O0) {
            optimize_generated_code(result.generated_files, config.opt_level);
        }
        
        // 5. 验证生成的代码
        if (!validate_generated_code(result.generated_files, error_message)) {
            result.success = false;
            result.generation_log = log.str() + "\nValidation failed: " + error_message;
            return result;
        }
        
        // 6. 计算性能估算
        result.estimated_performance_gain = estimate_performance_gain(optimized_graph, result);
        
        // 7. 生成编译命令
        result.compilation_commands = "nvcc -o yica_kernel yica_kernel.cu -lcuda";
        
        // 8. 填充结果信息
        result.success = true;
        result.metrics["total_files"] = static_cast<float>(result.generated_files.size());
        result.spm_utilization = analysis.spm_efficiency;
        
        log << "Code generation completed successfully\n";
        log << "Generated " << result.generated_files.size() << " files\n";
        
    } catch (const std::exception& e) {
        result.success = false;
        log << "Exception during code generation: " << e.what() << "\n";
    }
    
    result.generation_log = log.str();
    return result;
}

GenerationResult YICACodeGenerator::generate_yica_kernel(
    const kernel::Graph& graph,
    const YICAConfig& config) {
    
    GenerationConfig gen_config;
    gen_config.target_config = config;
    gen_config.target = CodeGenTarget::YICA_NATIVE;
    
    // 创建简化的分析结果
    AnalysisResult analysis;
    analysis.cim_friendliness_score = 0.8f; // 假设高CIM友好度
    analysis.spm_efficiency = 0.6f;
    
    return generate_code(graph, analysis, gen_config);
}

void YICACodeGenerator::register_operator_generator(OperatorGeneratorPtr generator) {
    if (generator) {
        std::string name = generator->get_name();
        operator_generators_[name] = std::move(generator);
    }
}

void YICACodeGenerator::unregister_operator_generator(const std::string& name) {
    operator_generators_.erase(name);
}

void YICACodeGenerator::set_default_config(const GenerationConfig& config) {
    default_config_ = config;
}

GenerationConfig YICACodeGenerator::get_default_config() const {
    return default_config_;
}

CodeTemplateManager& YICACodeGenerator::get_template_manager() {
    return *template_manager_;
}

const CodeTemplateManager& YICACodeGenerator::get_template_manager() const {
    return *template_manager_;
}

// 私有方法实现

GenerationResult YICACodeGenerator::generate_kernel_file(
    const kernel::Graph& graph,
    const GenerationContext& context) {
    
    GenerationResult result;
    
    // 生成内核文件
    std::stringstream kernel_content;
    kernel_content << "#include \"yica_runtime.h\"\n\n";
    
    // 使用模板生成内核
    std::map<std::string, std::string> params;
    params["KERNEL_NAME"] = "yica_compute_kernel";
    params["INPUT_PARAMETERS"] = "float* input, float* output, int size";
    params["SPM_SIZE"] = std::to_string(context.target_config.spm_size_kb * 256); // 转换为元素数
    
    std::stringstream kernel_body;
    kernel_body << "    // Generated kernel body\n";
    kernel_body << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    kernel_body << "    if (idx < size) {\n";
    kernel_body << "        output[idx] = input[idx] * 2.0f; // Placeholder computation\n";
    kernel_body << "    }\n";
    
    params["KERNEL_BODY"] = kernel_body.str();
    
    std::string kernel_code = template_manager_->instantiate_template("cim_kernel_base", params);
    kernel_content << kernel_code;
    
    GeneratedFile kernel_file;
    kernel_file.filename = "yica_kernel.cu";
    kernel_file.content = kernel_content.str();
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_file.content.size();
    kernel_file.description = "Generated YICA CUDA kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    
    return result;
}

GenerationResult YICACodeGenerator::generate_header_file(
    const kernel::Graph& graph,
    const GenerationContext& context) {
    
    GenerationResult result;
    
    std::stringstream header_content;
    header_content << "#pragma once\n\n";
    header_content << "#include <cuda_runtime.h>\n";
    header_content << "#include \"yirage/search/yica/yica_types.h\"\n\n";
    header_content << "// YICA Kernel Declarations\n";
    header_content << "__global__ void yica_compute_kernel(float* input, float* output, int size);\n\n";
    header_content << "// Host Interface\n";
    header_content << "class YICAKernelInterface {\n";
    header_content << "public:\n";
    header_content << "    void launch_kernel(float* input, float* output, int size);\n";
    header_content << "};\n";
    
    GeneratedFile header_file;
    header_file.filename = "yica_kernel.h";
    header_file.content = header_content.str();
    header_file.file_type = ".h";
    header_file.size_bytes = header_file.content.size();
    header_file.description = "Generated YICA kernel header";
    
    result.generated_files.push_back(header_file);
    result.success = true;
    
    return result;
}

GenerationResult YICACodeGenerator::generate_host_file(
    const kernel::Graph& graph,
    const GenerationContext& context) {
    
    GenerationResult result;
    
    std::stringstream host_content;
    host_content << "#include \"yica_kernel.h\"\n";
    host_content << "#include <iostream>\n\n";
    host_content << "void YICAKernelInterface::launch_kernel(float* input, float* output, int size) {\n";
    host_content << "    // Kernel launch configuration\n";
    host_content << "    int block_size = 256;\n";
    host_content << "    int grid_size = (size + block_size - 1) / block_size;\n\n";
    host_content << "    // Launch kernel\n";
    host_content << "    yica_compute_kernel<<<grid_size, block_size>>>(input, output, size);\n";
    host_content << "    \n";
    host_content << "    // Check for errors\n";
    host_content << "    cudaError_t err = cudaGetLastError();\n";
    host_content << "    if (err != cudaSuccess) {\n";
    host_content << "        std::cerr << \"CUDA error: \" << cudaGetErrorString(err) << std::endl;\n";
    host_content << "    }\n";
    host_content << "}\n";
    
    GeneratedFile host_file;
    host_file.filename = "yica_host.cpp";
    host_file.content = host_content.str();
    host_file.file_type = ".cpp";
    host_file.size_bytes = host_file.content.size();
    host_file.description = "Generated YICA host interface";
    
    result.generated_files.push_back(host_file);
    result.success = true;
    
    return result;
}

GenerationResult YICACodeGenerator::generate_makefile(const GenerationConfig& config) {
    GenerationResult result;
    
    std::stringstream makefile_content;
    makefile_content << "# Generated Makefile for YICA kernel\n\n";
    makefile_content << "NVCC = nvcc\n";
    makefile_content << "CXXFLAGS = -std=c++17 -O2\n";
    makefile_content << "NVCCFLAGS = -arch=sm_70\n\n";
    makefile_content << "TARGET = yica_kernel\n";
    makefile_content << "SOURCES = yica_kernel.cu yica_host.cpp\n\n";
    makefile_content << "$(TARGET): $(SOURCES)\n";
    makefile_content << "\t$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $^\n\n";
    makefile_content << "clean:\n";
    makefile_content << "\trm -f $(TARGET)\n\n";
    makefile_content << ".PHONY: clean\n";
    
    GeneratedFile makefile;
    makefile.filename = "Makefile";
    makefile.content = makefile_content.str();
    makefile.file_type = "";
    makefile.size_bytes = makefile.content.size();
    makefile.description = "Generated build script";
    
    result.generated_files.push_back(makefile);
    result.success = true;
    
    return result;
}

void YICACodeGenerator::analyze_graph_for_generation(
    const kernel::Graph& graph,
    GenerationContext& context) {
    
    // 分析图结构，为代码生成做准备
    context.instructions.clear();
    context.variable_map.clear();
    context.current_spm_offset = 0;
    context.next_temp_var_id = 0;
    
    // 为每个操作分配变量ID
    int var_id = 0;
    for (const auto& op : graph.operators) {
        for (auto* tensor : op->input_tensors) {
            std::string tensor_name = "tensor_" + std::to_string(reinterpret_cast<uintptr_t>(tensor));
            if (context.variable_map.find(tensor_name) == context.variable_map.end()) {
                context.variable_map[tensor_name] = var_id++;
            }
        }
        
        for (auto* tensor : op->output_tensors) {
            std::string tensor_name = "tensor_" + std::to_string(reinterpret_cast<uintptr_t>(tensor));
            if (context.variable_map.find(tensor_name) == context.variable_map.end()) {
                context.variable_map[tensor_name] = var_id++;
            }
        }
    }
}

void YICACodeGenerator::optimize_generated_code(
    std::vector<GeneratedFile>& files,
    OptimizationLevel level) {
    
    // 简化的代码优化
    for (auto& file : files) {
        if (file.file_type == ".cu") {
            // 对CUDA文件应用优化
            if (level >= OptimizationLevel::O2) {
                // 添加编译器优化指令
                file.content = "#pragma GCC optimize(\"O2\")\n" + file.content;
            }
        }
    }
}

float YICACodeGenerator::estimate_performance_gain(
    const kernel::Graph& graph,
    const GenerationResult& result) const {
    
    // 简化的性能估算
    float base_gain = 1.2f; // 基础20%提升
    
    // 基于操作数量调整
    float op_factor = std::min(static_cast<float>(graph.operators.size()) / 10.0f, 2.0f);
    
    // 基于生成文件数量调整
    float file_factor = std::min(static_cast<float>(result.generated_files.size()) / 5.0f, 1.5f);
    
    return base_gain * op_factor * file_factor;
}

bool YICACodeGenerator::validate_generation_input(
    const kernel::Graph& graph,
    const GenerationConfig& config,
    std::string& error_message) {
    
    if (graph.operators.empty()) {
        error_message = "Empty graph provided";
        return false;
    }
    
    if (config.target_config.num_cim_arrays == 0) {
        error_message = "Invalid CIM array configuration";
        return false;
    }
    
    return true;
}

bool YICACodeGenerator::validate_generated_code(
    const std::vector<GeneratedFile>& files,
    std::string& error_message) {
    
    if (files.empty()) {
        error_message = "No files generated";
        return false;
    }
    
    // 检查是否有基本的文件类型
    bool has_kernel = false;
    bool has_header = false;
    
    for (const auto& file : files) {
        if (file.file_type == ".cu") has_kernel = true;
        if (file.file_type == ".h") has_header = true;
    }
    
    if (!has_kernel) {
        error_message = "No kernel file generated";
        return false;
    }
    
    return true;
}

} // namespace yica
} // namespace search
} // namespace yirage 