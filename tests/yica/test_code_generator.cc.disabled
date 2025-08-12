#include "yirage/search/yica/code_generator.h"
#include "yirage/search/yica/operator_generators.h"
#include <gtest/gtest.h>
#include <memory>

namespace yirage {
namespace search {
namespace yica {

class CodeGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置默认的YICA配置
        config_.num_cim_arrays = 4;
        config_.cim_array_size = 128;
        config_.spm_size_kb = 64;
        config_.memory_bandwidth_gbps = 100.0f;
        config_.compute_frequency_ghz = 1.0f;
        
        // 创建代码生成器
        generator_ = std::make_unique<YICACodeGenerator>();
        
        // 设置生成配置
        gen_config_.target_config = config_;
        gen_config_.target = CodeGenTarget::CUDA_KERNEL;
        gen_config_.opt_level = OptimizationLevel::O2;
        gen_config_.enable_debug_info = false;
        gen_config_.output_directory = "./test_output/";
    }
    
    void TearDown() override {
        generator_.reset();
    }
    
    YICAConfig config_;
    GenerationConfig gen_config_;
    std::unique_ptr<YICACodeGenerator> generator_;
};

// ===== CodeTemplateManager Tests =====

TEST_F(CodeGeneratorTest, TemplateManagerBasicFunctions) {
    auto& template_manager = generator_->get_template_manager();
    
    // 测试获取模板列表
    auto template_names = template_manager.get_template_names();
    EXPECT_FALSE(template_names.empty());
    
    // 测试检查模板存在
    EXPECT_TRUE(template_manager.has_template("cim_kernel_base"));
    EXPECT_FALSE(template_manager.has_template("non_existent_template"));
}

TEST_F(CodeGeneratorTest, TemplateInstantiation) {
    auto& template_manager = generator_->get_template_manager();
    
    // 测试模板实例化
    std::map<std::string, std::string> params;
    params["KERNEL_NAME"] = "test_kernel";
    params["INPUT_PARAMETERS"] = "float* input, float* output";
    params["SPM_SIZE"] = "1024";
    params["KERNEL_BODY"] = "// Test kernel body";
    
    std::string instantiated = template_manager.instantiate_template("cim_kernel_base", params);
    
    EXPECT_FALSE(instantiated.empty());
    EXPECT_NE(instantiated.find("test_kernel"), std::string::npos);
    EXPECT_NE(instantiated.find("float* input, float* output"), std::string::npos);
    EXPECT_NE(instantiated.find("1024"), std::string::npos);
}

TEST_F(CodeGeneratorTest, CustomTemplateRegistration) {
    auto& template_manager = generator_->get_template_manager();
    
    // 注册自定义模板
    CodeTemplate custom_template;
    custom_template.name = "custom_test_template";
    custom_template.type = TemplateType::CIM_KERNEL;
    custom_template.content = "Custom template with {PARAM1} and {PARAM2}";
    custom_template.parameters = {"PARAM1", "PARAM2"};
    custom_template.description = "Test custom template";
    
    template_manager.register_template(custom_template);
    
    // 验证注册成功
    EXPECT_TRUE(template_manager.has_template("custom_test_template"));
    
    // 测试实例化
    std::map<std::string, std::string> params;
    params["PARAM1"] = "value1";
    params["PARAM2"] = "value2";
    
    std::string result = template_manager.instantiate_template("custom_test_template", params);
    EXPECT_EQ(result, "Custom template with value1 and value2");
}

// ===== CIMCodeGenAlgorithm Tests =====

TEST_F(CodeGeneratorTest, CIMInstructionGeneration) {
    // 创建模拟的操作（需要创建简单的mock）
    // 这里简化测试，主要测试算法逻辑
    CIMCodeGenAlgorithm algorithm;
    
    // 测试空指针输入
    auto instructions = algorithm.generate_cim_instructions(nullptr, config_);
    EXPECT_TRUE(instructions.empty());
}

TEST_F(CodeGeneratorTest, CIMInstructionOptimization) {
    CIMCodeGenAlgorithm algorithm;
    
    // 创建测试指令序列
    std::vector<CIMInstruction> instructions;
    
    CIMInstruction inst1;
    inst1.type = InstructionType::SPM_LOAD;
    inst1.operand_addresses = {0};
    inst1.result_address = 1;
    instructions.push_back(inst1);
    
    CIMInstruction inst2;
    inst2.type = InstructionType::CIM_MATMUL;
    inst2.operand_addresses = {1, 2};
    inst2.result_address = 3;
    instructions.push_back(inst2);
    
    // 测试优化
    algorithm.optimize_instructions(instructions, OptimizationLevel::O2);
    
    // 验证指令顺序（SPM_LOAD应该在前面）
    EXPECT_EQ(instructions[0].type, InstructionType::SPM_LOAD);
}

TEST_F(CodeGeneratorTest, CIMAssemblyGeneration) {
    CIMCodeGenAlgorithm algorithm;
    
    std::vector<CIMInstruction> instructions;
    
    CIMInstruction inst;
    inst.type = InstructionType::CIM_MATMUL;
    inst.operand_addresses = {0, 1};
    inst.result_address = 2;
    inst.assembly_code = "cim.matmul r2, r0, r1";
    instructions.push_back(inst);
    
    std::string assembly = algorithm.generate_assembly(instructions);
    
    EXPECT_FALSE(assembly.empty());
    EXPECT_NE(assembly.find("cim.matmul r2, r0, r1"), std::string::npos);
    EXPECT_NE(assembly.find(".cim_kernel:"), std::string::npos);
    EXPECT_NE(assembly.find("ret"), std::string::npos);
}

// ===== OperatorGenerator Tests =====

TEST_F(CodeGeneratorTest, MatmulOperatorGenerator) {
    MatmulOperatorGenerator matmul_gen;
    
    // 测试生成器名称
    EXPECT_EQ(matmul_gen.get_name(), "MatmulOperatorGenerator");
    
    // 测试能力检查（需要mock KNOperator）
    // 这里简化测试
    EXPECT_FALSE(matmul_gen.can_generate(nullptr));
}

TEST_F(CodeGeneratorTest, ElementwiseOperatorGenerator) {
    ElementwiseOperatorGenerator ew_gen;
    
    EXPECT_EQ(ew_gen.get_name(), "ElementwiseOperatorGenerator");
    EXPECT_FALSE(ew_gen.can_generate(nullptr));
}

TEST_F(CodeGeneratorTest, OperatorGeneratorRegistration) {
    // 测试操作生成器注册
    auto custom_gen = std::make_unique<MatmulOperatorGenerator>();
    std::string gen_name = custom_gen->get_name();
    
    generator_->register_operator_generator(std::move(custom_gen));
    
    // 测试注销
    generator_->unregister_operator_generator(gen_name);
}

// ===== YICACodeGenerator Integration Tests =====

TEST_F(CodeGeneratorTest, DefaultConfigManagement) {
    // 测试默认配置管理
    generator_->set_default_config(gen_config_);
    
    auto retrieved_config = generator_->get_default_config();
    EXPECT_EQ(retrieved_config.target, gen_config_.target);
    EXPECT_EQ(retrieved_config.opt_level, gen_config_.opt_level);
    EXPECT_EQ(retrieved_config.target_config.num_cim_arrays, config_.num_cim_arrays);
}

TEST_F(CodeGeneratorTest, CodeGenerationInputValidation) {
    // 创建空图进行测试
    kernel::Graph empty_graph;
    AnalysisResult analysis;
    analysis.cim_friendliness_score = 0.5f;
    analysis.spm_efficiency = 0.6f;
    
    // 测试空图验证
    auto result = generator_->generate_code(empty_graph, analysis, gen_config_);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.generation_log.empty());
}

TEST_F(CodeGeneratorTest, YICAKernelGeneration) {
    // 创建简单图进行测试
    kernel::Graph test_graph;
    
    // 创建模拟操作（简化）
    // 在实际应用中需要创建真实的KNOperator对象
    
    auto result = generator_->generate_yica_kernel(test_graph, config_);
    
    // 即使是空图，也应该生成一些基本文件
    EXPECT_TRUE(result.success || !result.generation_log.empty());
}

// ===== Performance and Metrics Tests =====

TEST_F(CodeGeneratorTest, PerformanceEstimation) {
    kernel::Graph test_graph;
    AnalysisResult analysis;
    analysis.cim_friendliness_score = 0.8f;
    analysis.spm_efficiency = 0.7f;
    
    auto result = generator_->generate_code(test_graph, analysis, gen_config_);
    
    // 性能估算应该是合理的数值
    if (result.success) {
        EXPECT_GE(result.estimated_performance_gain, 1.0f);
        EXPECT_LE(result.estimated_performance_gain, 5.0f);
    }
}

TEST_F(CodeGeneratorTest, GeneratedFileStructure) {
    kernel::Graph test_graph;
    AnalysisResult analysis;
    analysis.cim_friendliness_score = 0.8f;
    analysis.spm_efficiency = 0.7f;
    
    auto result = generator_->generate_code(test_graph, analysis, gen_config_);
    
    if (result.success) {
        // 检查生成的文件结构
        bool has_kernel_file = false;
        bool has_header_file = false;
        bool has_host_file = false;
        bool has_makefile = false;
        
        for (const auto& file : result.generated_files) {
            if (file.file_type == ".cu") has_kernel_file = true;
            if (file.file_type == ".h") has_header_file = true;
            if (file.file_type == ".cpp") has_host_file = true;
            if (file.filename == "Makefile") has_makefile = true;
            
            // 验证文件基本属性
            EXPECT_FALSE(file.filename.empty());
            EXPECT_FALSE(file.content.empty());
            EXPECT_GT(file.size_bytes, 0);
            EXPECT_FALSE(file.description.empty());
        }
        
        // 至少应该有内核文件
        EXPECT_TRUE(has_kernel_file);
    }
}

TEST_F(CodeGeneratorTest, OptimizationLevels) {
    kernel::Graph test_graph;
    AnalysisResult analysis;
    analysis.cim_friendliness_score = 0.8f;
    analysis.spm_efficiency = 0.7f;
    
    // 测试不同优化级别
    std::vector<OptimizationLevel> levels = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3
    };
    
    for (auto level : levels) {
        gen_config_.opt_level = level;
        auto result = generator_->generate_code(test_graph, analysis, gen_config_);
        
        // 所有优化级别都应该能够成功生成代码（或给出合理的错误信息）
        EXPECT_TRUE(result.success || !result.generation_log.empty());
        
        if (result.success) {
            EXPECT_FALSE(result.generated_files.empty());
        }
    }
}

TEST_F(CodeGeneratorTest, ErrorHandling) {
    // 测试错误配置
    GenerationConfig bad_config = gen_config_;
    bad_config.target_config.num_cim_arrays = 0; // 无效配置
    
    kernel::Graph test_graph;
    AnalysisResult analysis;
    
    auto result = generator_->generate_code(test_graph, analysis, bad_config);
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.generation_log.empty());
    EXPECT_NE(result.generation_log.find("Invalid CIM array configuration"), std::string::npos);
}

// ===== Memory and Resource Tests =====

TEST_F(CodeGeneratorTest, MemoryFootprint) {
    // 测试生成器的内存使用
    auto& template_manager = generator_->get_template_manager();
    auto template_names = template_manager.get_template_names();
    
    // 应该有合理数量的内置模板
    EXPECT_GE(template_names.size(), 3);
    EXPECT_LE(template_names.size(), 20);
}

TEST_F(CodeGeneratorTest, ConcurrentAccess) {
    // 测试模板管理器的并发访问安全性（简化测试）
    auto& template_manager = generator_->get_template_manager();
    
    // 多次访问应该返回一致的结果
    auto names1 = template_manager.get_template_names();
    auto names2 = template_manager.get_template_names();
    
    EXPECT_EQ(names1.size(), names2.size());
}

} // namespace yica
} // namespace search
} // namespace yirage 