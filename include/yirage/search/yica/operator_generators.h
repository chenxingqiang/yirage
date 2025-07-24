#pragma once

#include "yirage/search/yica/code_generator.h"

namespace yirage {
namespace search {
namespace yica {

/**
 * 矩阵乘法操作生成器
 */
class MatmulOperatorGenerator : public OperatorGenerator {
public:
    MatmulOperatorGenerator() = default;
    virtual ~MatmulOperatorGenerator() = default;
    
    bool can_generate(const kernel::KNOperator* op) const override;
    GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const override;
    std::string get_name() const override { return "MatmulOperatorGenerator"; }

private:
    std::string generate_cim_matmul_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    void optimize_matmul_tiling(
        int M, int N, int K,
        const YICAConfig& config,
        int& tile_m, int& tile_n, int& tile_k) const;
};

/**
 * 逐元素操作生成器
 */
class ElementwiseOperatorGenerator : public OperatorGenerator {
public:
    ElementwiseOperatorGenerator() = default;
    virtual ~ElementwiseOperatorGenerator() = default;
    
    bool can_generate(const kernel::KNOperator* op) const override;
    GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const override;
    std::string get_name() const override { return "ElementwiseOperatorGenerator"; }

private:
    std::string generate_elementwise_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    std::string get_elementwise_operation_code(const std::string& op_type) const;
};

/**
 * 卷积操作生成器
 */
class ConvolutionOperatorGenerator : public OperatorGenerator {
public:
    ConvolutionOperatorGenerator() = default;
    virtual ~ConvolutionOperatorGenerator() = default;
    
    bool can_generate(const kernel::KNOperator* op) const override;
    GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const override;
    std::string get_name() const override { return "ConvolutionOperatorGenerator"; }

private:
    std::string generate_conv_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    void optimize_conv_mapping(
        int input_h, int input_w, int kernel_h, int kernel_w,
        const YICAConfig& config,
        int& block_h, int& block_w) const;
};

/**
 * 归一化操作生成器
 */
class NormalizationOperatorGenerator : public OperatorGenerator {
public:
    NormalizationOperatorGenerator() = default;
    virtual ~NormalizationOperatorGenerator() = default;
    
    bool can_generate(const kernel::KNOperator* op) const override;
    GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const override;
    std::string get_name() const override { return "NormalizationOperatorGenerator"; }

private:
    std::string generate_layernorm_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    std::string generate_reduction_code(
        const std::string& reduction_type,
        const YICAConfig& config) const;
};

/**
 * 激活函数生成器
 */
class ActivationOperatorGenerator : public OperatorGenerator {
public:
    ActivationOperatorGenerator() = default;
    virtual ~ActivationOperatorGenerator() = default;
    
    bool can_generate(const kernel::KNOperator* op) const override;
    GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const override;
    std::string get_name() const override { return "ActivationOperatorGenerator"; }

private:
    std::string generate_activation_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    std::string get_activation_function_code(const std::string& activation_type) const;
};

/**
 * 内存操作生成器
 */
class MemoryOperatorGenerator : public OperatorGenerator {
public:
    MemoryOperatorGenerator() = default;
    virtual ~MemoryOperatorGenerator() = default;
    
    bool can_generate(const kernel::KNOperator* op) const override;
    GenerationResult generate(
        const kernel::KNOperator* op,
        const GenerationContext& context) const override;
    std::string get_name() const override { return "MemoryOperatorGenerator"; }

private:
    std::string generate_memory_copy_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
    
    std::string generate_memory_reshape_kernel(
        const kernel::KNOperator* op,
        const YICAConfig& config) const;
};

} // namespace yica
} // namespace search
} // namespace yirage 