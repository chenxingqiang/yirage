#include "yirage/search/yica/operator_generators.h"
#include <sstream>
#include <cmath>

namespace yirage {
namespace search {
namespace yica {

// ===== MatmulOperatorGenerator Implementation =====

bool MatmulOperatorGenerator::can_generate(const kernel::KNOperator* op) const {
    return op && op->op_type == kernel::KNOperatorType::KN_MATMUL_OP;
}

GenerationResult MatmulOperatorGenerator::generate(
    const kernel::KNOperator* op,
    const GenerationContext& context) const {
    
    GenerationResult result;
    
    if (!can_generate(op)) {
        result.success = false;
        return result;
    }
    
    // 生成矩阵乘法内核代码
    std::string kernel_code = generate_cim_matmul_kernel(op, context.target_config);
    
    GeneratedFile kernel_file;
    kernel_file.filename = "matmul_kernel.cu";
    kernel_file.content = kernel_code;
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_code.size();
    kernel_file.description = "CIM-optimized matrix multiplication kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    result.estimated_performance_gain = 1.5f; // 50%性能提升估算
    
    return result;
}

std::string MatmulOperatorGenerator::generate_cim_matmul_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    
    ss << "__global__ void cim_matmul_kernel(\n";
    ss << "    const float* A, const float* B, float* C,\n";
    ss << "    int M, int N, int K\n";
    ss << ") {\n";
    ss << "    // CIM array ID and thread configuration\n";
    ss << "    const int cim_array_id = blockIdx.x;\n";
    ss << "    const int thread_id = threadIdx.x;\n";
    ss << "    const int block_size = blockDim.x;\n\n";
    
    // 确定分块大小
    int tile_m, tile_n, tile_k;
    optimize_matmul_tiling(256, 256, 256, config, tile_m, tile_n, tile_k);
    
    ss << "    // Optimized tiling: " << tile_m << "x" << tile_n << "x" << tile_k << "\n";
    ss << "    const int TILE_M = " << tile_m << ";\n";
    ss << "    const int TILE_N = " << tile_n << ";\n";
    ss << "    const int TILE_K = " << tile_k << ";\n\n";
    
    ss << "    // SPM allocation for CIM computation\n";
    ss << "    __shared__ float spm_A[" << tile_m * tile_k << "];\n";
    ss << "    __shared__ float spm_B[" << tile_k * tile_n << "];\n";
    ss << "    __shared__ float spm_C[" << tile_m * tile_n << "];\n\n";
    
    ss << "    // Compute thread block position\n";
    ss << "    int block_row = blockIdx.y;\n";
    ss << "    int block_col = blockIdx.x;\n\n";
    
    ss << "    // Initialize accumulator\n";
    ss << "    float accumulator = 0.0f;\n\n";
    
    ss << "    // Main computation loop\n";
    ss << "    for (int k_block = 0; k_block < (K + TILE_K - 1) / TILE_K; ++k_block) {\n";
    ss << "        // Load data into SPM\n";
    ss << "        int a_row = block_row * TILE_M + thread_id / TILE_K;\n";
    ss << "        int a_col = k_block * TILE_K + thread_id % TILE_K;\n";
    ss << "        int b_row = k_block * TILE_K + thread_id / TILE_N;\n";
    ss << "        int b_col = block_col * TILE_N + thread_id % TILE_N;\n\n";
    
    ss << "        if (a_row < M && a_col < K) {\n";
    ss << "            spm_A[thread_id] = A[a_row * K + a_col];\n";
    ss << "        } else {\n";
    ss << "            spm_A[thread_id] = 0.0f;\n";
    ss << "        }\n\n";
    
    ss << "        if (b_row < K && b_col < N) {\n";
    ss << "            spm_B[thread_id] = B[b_row * N + b_col];\n";
    ss << "        } else {\n";
    ss << "            spm_B[thread_id] = 0.0f;\n";
    ss << "        }\n\n";
    
    ss << "        __syncthreads();\n\n";
    
    ss << "        // CIM computation within SPM\n";
    ss << "        for (int k = 0; k < TILE_K; ++k) {\n";
    ss << "            int row = thread_id / TILE_N;\n";
    ss << "            int col = thread_id % TILE_N;\n";
    ss << "            if (row < TILE_M && col < TILE_N) {\n";
    ss << "                accumulator += spm_A[row * TILE_K + k] * spm_B[k * TILE_N + col];\n";
    ss << "            }\n";
    ss << "        }\n\n";
    
    ss << "        __syncthreads();\n";
    ss << "    }\n\n";
    
    ss << "    // Store result\n";
    ss << "    int out_row = block_row * TILE_M + thread_id / TILE_N;\n";
    ss << "    int out_col = block_col * TILE_N + thread_id % TILE_N;\n";
    ss << "    if (out_row < M && out_col < N) {\n";
    ss << "        C[out_row * N + out_col] = accumulator;\n";
    ss << "    }\n";
    ss << "}\n";
    
    return ss.str();
}

void MatmulOperatorGenerator::optimize_matmul_tiling(
    int M, int N, int K,
    const YICAConfig& config,
    int& tile_m, int& tile_n, int& tile_k) const {
    
    // 基于CIM阵列大小和SPM容量优化分块
    int max_spm_elements = config.spm_size_kb * 256; // 假设float类型
    
    // 简化的分块策略
    tile_m = std::min(32, M);
    tile_n = std::min(32, N);
    tile_k = std::min(32, K);
    
    // 确保分块大小适合SPM
    while ((tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) > max_spm_elements) {
        tile_m = std::max(16, tile_m - 8);
        tile_n = std::max(16, tile_n - 8);
        tile_k = std::max(16, tile_k - 8);
    }
}

// ===== ElementwiseOperatorGenerator Implementation =====

bool ElementwiseOperatorGenerator::can_generate(const kernel::KNOperator* op) const {
    return op && (op->op_type == kernel::KNOperatorType::KN_EW_ADD_OP ||
                  op->op_type == kernel::KNOperatorType::KN_EW_MUL_OP);
}

GenerationResult ElementwiseOperatorGenerator::generate(
    const kernel::KNOperator* op,
    const GenerationContext& context) const {
    
    GenerationResult result;
    
    if (!can_generate(op)) {
        result.success = false;
        return result;
    }
    
    std::string kernel_code = generate_elementwise_kernel(op, context.target_config);
    
    GeneratedFile kernel_file;
    kernel_file.filename = "elementwise_kernel.cu";
    kernel_file.content = kernel_code;
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_code.size();
    kernel_file.description = "CIM-optimized elementwise operation kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    result.estimated_performance_gain = 1.3f;
    
    return result;
}

std::string ElementwiseOperatorGenerator::generate_elementwise_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    std::string op_code = get_elementwise_operation_code(
        op->op_type == kernel::KNOperatorType::KN_EW_ADD_OP ? "add" : "mul");
    
    ss << "__global__ void cim_elementwise_kernel(\n";
    ss << "    const float* input1, const float* input2, float* output,\n";
    ss << "    int size\n";
    ss << ") {\n";
    ss << "    const int cim_array_id = blockIdx.x;\n";
    ss << "    const int thread_id = threadIdx.x;\n";
    ss << "    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;\n\n";
    
    ss << "    // SPM buffer for vectorized operations\n";
    ss << "    __shared__ float spm_input1[256];\n";
    ss << "    __shared__ float spm_input2[256];\n";
    ss << "    __shared__ float spm_output[256];\n\n";
    
    ss << "    // Load data into SPM\n";
    ss << "    int base_idx = blockIdx.x * blockDim.x;\n";
    ss << "    if (base_idx + thread_id < size) {\n";
    ss << "        spm_input1[thread_id] = input1[base_idx + thread_id];\n";
    ss << "        spm_input2[thread_id] = input2[base_idx + thread_id];\n";
    ss << "    } else {\n";
    ss << "        spm_input1[thread_id] = 0.0f;\n";
    ss << "        spm_input2[thread_id] = 0.0f;\n";
    ss << "    }\n\n";
    
    ss << "    __syncthreads();\n\n";
    
    ss << "    // CIM elementwise computation\n";
    ss << "    " << op_code << "\n\n";
    
    ss << "    __syncthreads();\n\n";
    
    ss << "    // Store result\n";
    ss << "    if (base_idx + thread_id < size) {\n";
    ss << "        output[base_idx + thread_id] = spm_output[thread_id];\n";
    ss << "    }\n";
    ss << "}\n";
    
    return ss.str();
}

std::string ElementwiseOperatorGenerator::get_elementwise_operation_code(const std::string& op_type) const {
    if (op_type == "add") {
        return "spm_output[thread_id] = spm_input1[thread_id] + spm_input2[thread_id];";
    } else if (op_type == "mul") {
        return "spm_output[thread_id] = spm_input1[thread_id] * spm_input2[thread_id];";
    } else {
        return "spm_output[thread_id] = spm_input1[thread_id]; // Unsupported operation";
    }
}

// ===== ConvolutionOperatorGenerator Implementation =====

bool ConvolutionOperatorGenerator::can_generate(const kernel::KNOperator* op) const {
    return op && op->op_type == kernel::KNOperatorType::KN_CONV_2D_OP;
}

GenerationResult ConvolutionOperatorGenerator::generate(
    const kernel::KNOperator* op,
    const GenerationContext& context) const {
    
    GenerationResult result;
    
    if (!can_generate(op)) {
        result.success = false;
        return result;
    }
    
    std::string kernel_code = generate_conv_kernel(op, context.target_config);
    
    GeneratedFile kernel_file;
    kernel_file.filename = "conv_kernel.cu";
    kernel_file.content = kernel_code;
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_code.size();
    kernel_file.description = "CIM-optimized convolution kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    result.estimated_performance_gain = 1.8f;
    
    return result;
}

std::string ConvolutionOperatorGenerator::generate_conv_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    
    ss << "__global__ void cim_conv2d_kernel(\n";
    ss << "    const float* input, const float* weight, float* output,\n";
    ss << "    int batch_size, int in_channels, int in_height, int in_width,\n";
    ss << "    int out_channels, int out_height, int out_width,\n";
    ss << "    int kernel_h, int kernel_w, int stride_h, int stride_w\n";
    ss << ") {\n";
    ss << "    const int cim_array_id = blockIdx.x;\n";
    ss << "    const int thread_id = threadIdx.x;\n\n";
    
    ss << "    // SPM allocation for convolution tiles\n";
    ss << "    __shared__ float spm_input_tile[32*32];\n";
    ss << "    __shared__ float spm_weight_tile[16*16];\n";
    ss << "    __shared__ float spm_output_tile[16*16];\n\n";
    
    ss << "    // Compute output position\n";
    ss << "    int out_y = blockIdx.y * blockDim.y + threadIdx.y;\n";
    ss << "    int out_x = blockIdx.z * blockDim.z + threadIdx.z;\n";
    ss << "    int out_c = blockIdx.x;\n\n";
    
    ss << "    if (out_y >= out_height || out_x >= out_width || out_c >= out_channels) return;\n\n";
    
    ss << "    float accumulator = 0.0f;\n\n";
    
    ss << "    // Convolution computation\n";
    ss << "    for (int ic = 0; ic < in_channels; ++ic) {\n";
    ss << "        for (int ky = 0; ky < kernel_h; ++ky) {\n";
    ss << "            for (int kx = 0; kx < kernel_w; ++kx) {\n";
    ss << "                int in_y = out_y * stride_h + ky;\n";
    ss << "                int in_x = out_x * stride_w + kx;\n\n";
    
    ss << "                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {\n";
    ss << "                    int input_idx = ic * in_height * in_width + in_y * in_width + in_x;\n";
    ss << "                    int weight_idx = out_c * in_channels * kernel_h * kernel_w +\n";
    ss << "                                     ic * kernel_h * kernel_w + ky * kernel_w + kx;\n\n";
    
    ss << "                    // CIM-accelerated multiply-accumulate\n";
    ss << "                    accumulator += input[input_idx] * weight[weight_idx];\n";
    ss << "                }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n\n";
    
    ss << "    // Store result\n";
    ss << "    int output_idx = out_c * out_height * out_width + out_y * out_width + out_x;\n";
    ss << "    output[output_idx] = accumulator;\n";
    ss << "}\n";
    
    return ss.str();
}

void ConvolutionOperatorGenerator::optimize_conv_mapping(
    int input_h, int input_w, int kernel_h, int kernel_w,
    const YICAConfig& config,
    int& block_h, int& block_w) const {
    
    // 基于CIM阵列优化卷积映射
    block_h = std::min(16, input_h);
    block_w = std::min(16, input_w);
    
    // 考虑SPM容量限制
    int max_elements = config.spm_size_kb * 256;
    while ((block_h * block_w + kernel_h * kernel_w) > max_elements) {
        block_h = std::max(8, block_h - 4);
        block_w = std::max(8, block_w - 4);
    }
}

// ===== NormalizationOperatorGenerator Implementation =====

bool NormalizationOperatorGenerator::can_generate(const kernel::KNOperator* op) const {
    return op && op->op_type == kernel::KNOperatorType::KN_RMS_NORM_OP;
}

GenerationResult NormalizationOperatorGenerator::generate(
    const kernel::KNOperator* op,
    const GenerationContext& context) const {
    
    GenerationResult result;
    
    if (!can_generate(op)) {
        result.success = false;
        return result;
    }
    
    std::string kernel_code = generate_layernorm_kernel(op, context.target_config);
    
    GeneratedFile kernel_file;
    kernel_file.filename = "norm_kernel.cu";
    kernel_file.content = kernel_code;
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_code.size();
    kernel_file.description = "CIM-optimized normalization kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    result.estimated_performance_gain = 1.4f;
    
    return result;
}

std::string NormalizationOperatorGenerator::generate_layernorm_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    
    ss << "__global__ void cim_layernorm_kernel(\n";
    ss << "    const float* input, float* output,\n";
    ss << "    int batch_size, int hidden_size, float eps\n";
    ss << ") {\n";
    ss << "    const int batch_idx = blockIdx.x;\n";
    ss << "    const int thread_id = threadIdx.x;\n\n";
    
    ss << "    // SPM for reduction operations\n";
    ss << "    __shared__ float spm_data[1024];\n";
    ss << "    __shared__ float spm_mean;\n";
    ss << "    __shared__ float spm_variance;\n\n";
    
    ss << "    // Load data\n";
    ss << "    int base_idx = batch_idx * hidden_size;\n";
    ss << "    float local_sum = 0.0f;\n\n";
    
    ss << "    for (int i = thread_id; i < hidden_size; i += blockDim.x) {\n";
    ss << "        float val = input[base_idx + i];\n";
    ss << "        spm_data[i % 1024] = val;\n";
    ss << "        local_sum += val;\n";
    ss << "    }\n\n";
    
    ss << "    " << generate_reduction_code("sum", config) << "\n\n";
    
    ss << "    if (thread_id == 0) {\n";
    ss << "        spm_mean = local_sum / hidden_size;\n";
    ss << "    }\n";
    ss << "    __syncthreads();\n\n";
    
    ss << "    // Compute variance\n";
    ss << "    float local_var = 0.0f;\n";
    ss << "    for (int i = thread_id; i < hidden_size; i += blockDim.x) {\n";
    ss << "        float diff = input[base_idx + i] - spm_mean;\n";
    ss << "        local_var += diff * diff;\n";
    ss << "    }\n\n";
    
    ss << "    " << generate_reduction_code("sum", config) << "\n\n";
    
    ss << "    if (thread_id == 0) {\n";
    ss << "        spm_variance = sqrt(local_var / hidden_size + eps);\n";
    ss << "    }\n";
    ss << "    __syncthreads();\n\n";
    
    ss << "    // Normalize\n";
    ss << "    for (int i = thread_id; i < hidden_size; i += blockDim.x) {\n";
    ss << "        output[base_idx + i] = (input[base_idx + i] - spm_mean) / spm_variance;\n";
    ss << "    }\n";
    ss << "}\n";
    
    return ss.str();
}

std::string NormalizationOperatorGenerator::generate_reduction_code(
    const std::string& reduction_type,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    
    if (reduction_type == "sum") {
        ss << "    // CIM-accelerated reduction\n";
        ss << "    __shared__ float reduction_buffer[256];\n";
        ss << "    reduction_buffer[thread_id] = local_sum;\n";
        ss << "    __syncthreads();\n\n";
        
        ss << "    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {\n";
        ss << "        if (thread_id < stride) {\n";
        ss << "            reduction_buffer[thread_id] += reduction_buffer[thread_id + stride];\n";
        ss << "        }\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n\n";
        
        ss << "    if (thread_id == 0) {\n";
        ss << "        local_sum = reduction_buffer[0];\n";
        ss << "    }";
    }
    
    return ss.str();
}

// ===== ActivationOperatorGenerator Implementation =====

bool ActivationOperatorGenerator::can_generate(const kernel::KNOperator* op) const {
    return op && (op->op_type == kernel::KNOperatorType::KN_RELU_OP ||
                  op->op_type == kernel::KNOperatorType::KN_GELU_OP);
}

GenerationResult ActivationOperatorGenerator::generate(
    const kernel::KNOperator* op,
    const GenerationContext& context) const {
    
    GenerationResult result;
    
    if (!can_generate(op)) {
        result.success = false;
        return result;
    }
    
    std::string kernel_code = generate_activation_kernel(op, context.target_config);
    
    GeneratedFile kernel_file;
    kernel_file.filename = "activation_kernel.cu";
    kernel_file.content = kernel_code;
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_code.size();
    kernel_file.description = "CIM-optimized activation kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    result.estimated_performance_gain = 1.2f;
    
    return result;
}

std::string ActivationOperatorGenerator::generate_activation_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    std::string activation_type = (op->op_type == kernel::KNOperatorType::KN_RELU_OP) ? "relu" : "gelu";
    std::string activation_code = get_activation_function_code(activation_type);
    
    ss << "__global__ void cim_activation_kernel(\n";
    ss << "    const float* input, float* output, int size\n";
    ss << ") {\n";
    ss << "    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;\n\n";
    
    ss << "    if (global_id < size) {\n";
    ss << "        float val = input[global_id];\n";
    ss << "        " << activation_code << "\n";
    ss << "        output[global_id] = val;\n";
    ss << "    }\n";
    ss << "}\n";
    
    return ss.str();
}

std::string ActivationOperatorGenerator::get_activation_function_code(const std::string& activation_type) const {
    if (activation_type == "relu") {
        return "val = fmaxf(0.0f, val);";
    } else if (activation_type == "gelu") {
        return "val = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));";
    } else {
        return "// Unsupported activation function";
    }
}

// ===== MemoryOperatorGenerator Implementation =====

bool MemoryOperatorGenerator::can_generate(const kernel::KNOperator* op) const {
    return op && (op->op_type == kernel::KNOperatorType::KN_INPUT_OP ||
                  op->op_type == kernel::KNOperatorType::KN_RESHAPE_OP);
}

GenerationResult MemoryOperatorGenerator::generate(
    const kernel::KNOperator* op,
    const GenerationContext& context) const {
    
    GenerationResult result;
    
    if (!can_generate(op)) {
        result.success = false;
        return result;
    }
    
    std::string kernel_code;
    if (op->op_type == kernel::KNOperatorType::KN_INPUT_OP) {
        kernel_code = generate_memory_copy_kernel(op, context.target_config);
    } else {
        kernel_code = generate_memory_reshape_kernel(op, context.target_config);
    }
    
    GeneratedFile kernel_file;
    kernel_file.filename = "memory_kernel.cu";
    kernel_file.content = kernel_code;
    kernel_file.file_type = ".cu";
    kernel_file.size_bytes = kernel_code.size();
    kernel_file.description = "CIM-optimized memory operation kernel";
    
    result.generated_files.push_back(kernel_file);
    result.success = true;
    result.estimated_performance_gain = 1.1f;
    
    return result;
}

std::string MemoryOperatorGenerator::generate_memory_copy_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    
    ss << "__global__ void cim_memory_copy_kernel(\n";
    ss << "    const float* src, float* dst, int size\n";
    ss << ") {\n";
    ss << "    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    const int stride = blockDim.x * gridDim.x;\n\n";
    
    ss << "    // Vectorized memory copy using SPM\n";
    ss << "    __shared__ float spm_buffer[1024];\n\n";
    
    ss << "    for (int i = global_id; i < size; i += stride) {\n";
    ss << "        int local_idx = threadIdx.x;\n";
    ss << "        if (i < size) {\n";
    ss << "            spm_buffer[local_idx] = src[i];\n";
    ss << "        }\n";
    ss << "        __syncthreads();\n\n";
    
    ss << "        if (i < size) {\n";
    ss << "            dst[i] = spm_buffer[local_idx];\n";
    ss << "        }\n";
    ss << "        __syncthreads();\n";
    ss << "    }\n";
    ss << "}\n";
    
    return ss.str();
}

std::string MemoryOperatorGenerator::generate_memory_reshape_kernel(
    const kernel::KNOperator* op,
    const YICAConfig& config) const {
    
    std::stringstream ss;
    
    ss << "__global__ void cim_memory_reshape_kernel(\n";
    ss << "    const float* input, float* output,\n";
    ss << "    int* old_dims, int* new_dims, int ndims, int size\n";
    ss << ") {\n";
    ss << "    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;\n\n";
    
    ss << "    if (global_id < size) {\n";
    ss << "        // Direct copy for reshape (data layout doesn't change)\n";
    ss << "        output[global_id] = input[global_id];\n";
    ss << "    }\n";
    ss << "}\n";
    
    return ss.str();
}

} // namespace yica
} // namespace search
} // namespace yirage 