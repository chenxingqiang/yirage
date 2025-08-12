/**
 * @file yica_matmul.cc
 * @brief YICA 矩阵乘法内核实现 - 基于 Mirage μGraph CIM 优化
 */

#include "yirage/kernel/yica_matmul.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/yica/yis_instruction_engine.h"
#include <iostream>
#include <chrono>

namespace yirage {
namespace kernel {

YICAMatMulOp::YICAMatMulOp(Graph* graph, const DTensor& A, const DTensor& B, bool transpose_a, bool transpose_b, const YICAHardwareConfig& config)
    : YICAKernelBase(graph, config), A_(A), B_(B), transpose_a_(transpose_a), transpose_b_(transpose_b), alpha_(1.0f), beta_(0.0f) {
    
    std::cout << "🔧 初始化 YICA MatMul 内核..." << std::endl;
    
    // 获取矩阵维度
    M_ = A.dim[0];
    K_ = A.dim[1]; 
    N_ = B.dim[1];
    
    std::cout << "📊 矩阵维度: " << M_ << "x" << K_ << " @ " << K_ << "x" << N_ 
              << " = " << M_ << "x" << N_ << std::endl;
    
    std::cout << "✅ YICA MatMul 内核初始化完成" << std::endl;
}

bool YICAMatMulOp::execute() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "🚀 执行 YICA MatMul 计算..." << std::endl;
    
    try {
        // 1. 基于 Mirage μGraph 的 CIM 阵列优化
        bool success = execute_cim_matmul();
        
        if (success) {
            auto end_time = std::chrono::high_resolution_clock::now();
            execution_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            std::cout << "✅ YICA MatMul 执行完成，耗时: " << execution_time_ms_ << "ms" << std::endl;
            return true;
        } else {
            std::cout << "❌ YICA MatMul 执行失败" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ YICA MatMul 执行异常: " << e.what() << std::endl;
        return false;
    }
}

bool YICAMatMulOp::execute_cim_matmul() {
    std::cout << "🧮 执行 CIM 优化矩阵乘法..." << std::endl;
    
    // 获取设备内存管理器
    DeviceMemoryManager* dmm = DeviceMemoryManager::get_instance();
    if (!dmm) {
        std::cout << "❌ 设备内存管理器未初始化" << std::endl;
        return false;
    }
    
    char* base_ptr = dmm->data_base_ptr[dmm->gpu_id];
    if (!base_ptr) {
        std::cout << "❌ 设备内存基础指针为空" << std::endl;
        return false;
    }
    
    const DTensor& A = A_;
    const DTensor& B = B_;
    DTensor& C = C_;
    
    // 获取实际内存指针
    float* A_data = reinterpret_cast<float*>(base_ptr + A.data_offset);
    float* B_data = reinterpret_cast<float*>(base_ptr + B.data_offset);
    float* C_data = reinterpret_cast<float*>(base_ptr + C.data_offset);
    
    // 基于 Mirage μGraph 的 CIM 分块策略
    const int BLOCK_SIZE = 32; // CIM 阵列块大小
    
    std::cout << "📦 使用 CIM 分块策略: " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
    
    // CIM 并行矩阵乘法 (CPU 参考实现)
    for (int i = 0; i < M_; i += BLOCK_SIZE) {
        for (int j = 0; j < N_; j += BLOCK_SIZE) {
            for (int k = 0; k < K_; k += BLOCK_SIZE) {
                // 计算当前块的实际大小
                int block_m = std::min(BLOCK_SIZE, M_ - i);
                int block_n = std::min(BLOCK_SIZE, N_ - j);
                int block_k = std::min(BLOCK_SIZE, K_ - k);
                
                // CIM 阵列内部计算
                for (int bi = 0; bi < block_m; ++bi) {
                    for (int bj = 0; bj < block_n; ++bj) {
                        float sum = 0.0f;
                        for (int bk = 0; bk < block_k; ++bk) {
                            sum += A_data[(i + bi) * K_ + (k + bk)] * 
                                   B_data[(k + bk) * N_ + (j + bj)];
                        }
                        
                        if (k == 0) {
                            C_data[(i + bi) * N_ + (j + bj)] = sum;
                        } else {
                            C_data[(i + bi) * N_ + (j + bj)] += sum;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "✅ CIM 优化矩阵乘法完成" << std::endl;
  return true;
}

std::vector<yis::Instruction> YICAMatMulOp::generate_yis_instructions() {
    std::vector<yis::Instruction> instructions;
    
    // 基于 Mirage μGraph 的 MatMul YIS 指令序列
    
    // 1. 初始化 CIM 阵列
    yis::Instruction init_cim;
    init_cim.type = yis::InstructionType::SYNC_CIM;
    init_cim.opcode = "YISCONTROL_CALL_EU";
    init_cim.operands = {"cim_init", "4"};
    instructions.push_back(init_cim);
    
    // 2. 加载矩阵 A 到 SPM
    yis::Instruction load_a;
    load_a.type = yis::InstructionType::ECOPY_G2S;
    load_a.opcode = "YISECOPY_G2S";
    load_a.operands = {"matrix_a", "spm_a", std::to_string(M_ * K_ * sizeof(float))};
    instructions.push_back(load_a);
    
    // 3. 加载矩阵 B 到 SPM
    yis::Instruction load_b;
    load_b.type = yis::InstructionType::ECOPY_G2S;
    load_b.opcode = "YISECOPY_G2S";
    load_b.operands = {"matrix_b", "spm_b", std::to_string(K_ * N_ * sizeof(float))};
    instructions.push_back(load_b);
    
    // 4. CIM 矩阵乘法运算
    yis::Instruction matmul;
    matmul.type = yis::InstructionType::MMA;
    matmul.opcode = "YISMMA_ACC";
    matmul.operands = {"spm_a", "spm_b", "spm_c"};
    matmul.matrix_m = M_;
    matmul.matrix_n = N_;
    matmul.matrix_k = K_;
    instructions.push_back(matmul);
    
    // 5. 同步
    yis::Instruction sync;
    sync.type = yis::InstructionType::SYNC_BAR;
    sync.opcode = "YISSYNC_BAR";
    sync.operands = {"matmul_barrier"};
    instructions.push_back(sync);
    
    // 6. 结果写回
    yis::Instruction store_c;
    store_c.type = yis::InstructionType::ECOPY_S2G;
    store_c.opcode = "YISECOPY_S2G";
    store_c.operands = {"spm_c", "matrix_c", std::to_string(M_ * N_ * sizeof(float))};
    instructions.push_back(store_c);
    
    // 7. 结束
    yis::Instruction end;
    end.type = yis::InstructionType::SYNC_BAR;
    end.opcode = "YISCONTROL_END";
    instructions.push_back(end);
    
    return instructions;
}





}  // namespace kernel
}  // namespace yirage
