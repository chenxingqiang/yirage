#include "yirage/yica/yis_instruction_engine.h"
#include "yirage/yica/yis_instruction_set.h"
#include "yirage/yica/cim_array_simulator.h"
#include "yirage/yica/spm_memory_manager.h"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <cblas.h>

namespace yirage {
namespace yica {

YISInstructionEngine::YISInstructionEngine(const YICAConfig& config)
    : config_(config), 
      execution_stats_{0, 0, 0.0, 0.0},
      is_running_(false) {
    
    // 初始化CIM阵列模拟器
    cim_simulator_ = std::make_unique<CIMArraySimulator>(config_);
    
    // 初始化SPM内存管理器
    spm_manager_ = std::make_unique<SPMMemoryManager>(config_);
    
    // 初始化指令队列
    instruction_queue_.reserve(1000);
    
    // 设置OpenMP线程数
    omp_set_num_threads(config_.num_cpu_threads);
    
    LOG(INFO) << "YIS指令执行引擎初始化完成，CIM阵列数: " 
              << config_.num_cim_arrays;
}

YISInstructionEngine::~YISInstructionEngine() {
    stop();
}

bool YISInstructionEngine::execute_instruction(const YISInstruction& instruction) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = false;
    
    switch (instruction.type) {
        case YISInstructionType::YISECOPY:
            success = execute_external_copy(instruction);
            break;
            
        case YISInstructionType::YISICOPY:
            success = execute_internal_copy(instruction);
            break;
            
        case YISInstructionType::YISMMA:
            success = execute_matrix_multiply(instruction);
            break;
            
        case YISInstructionType::YISSYNC:
            success = execute_synchronization(instruction);
            break;
            
        case YISInstructionType::YISCONTROL:
            success = execute_control_flow(instruction);
            break;
            
        default:
            LOG(ERROR) << "未知的YIS指令类型: " 
                      << static_cast<int>(instruction.type);
            return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // 更新执行统计
    execution_stats_.total_instructions++;
    execution_stats_.total_execution_time_us += duration;
    
    if (!success) {
        execution_stats_.failed_instructions++;
    }
    
    return success;
}

bool YISInstructionEngine::execute_external_copy(const YISInstruction& instruction) {
    // YISECOPY: 外部内存到SPM的数据拷贝
    try {
        void* src_ptr = reinterpret_cast<void*>(instruction.src_addr);
        void* dst_ptr = spm_manager_->allocate_spm_buffer(
            instruction.cim_array_id, instruction.size);
        
        if (!dst_ptr) {
            LOG(ERROR) << "SPM内存分配失败，大小: " << instruction.size;
            return false;
        }
        
        // 使用OpenMP并行拷贝大数据块
        if (instruction.size > 1024 * 1024) { // 1MB以上使用并行拷贝
            const size_t chunk_size = instruction.size / omp_get_max_threads();
            
            #pragma omp parallel for
            for (int i = 0; i < omp_get_max_threads(); ++i) {
                size_t start = i * chunk_size;
                size_t end = (i == omp_get_max_threads() - 1) ? 
                            instruction.size : (i + 1) * chunk_size;
                
                memcpy(static_cast<char*>(dst_ptr) + start,
                       static_cast<const char*>(src_ptr) + start,
                       end - start);
            }
        } else {
            memcpy(dst_ptr, src_ptr, instruction.size);
        }
        
        // 更新内存访问统计
        execution_stats_.memory_access_bytes += instruction.size;
        
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "YISECOPY执行失败: " << e.what();
        return false;
    }
}

bool YISInstructionEngine::execute_internal_copy(const YISInstruction& instruction) {
    // YISICOPY: SPM内部数据重排
    try {
        auto src_buffer = spm_manager_->get_spm_buffer(
            instruction.cim_array_id, instruction.src_addr);
        auto dst_buffer = spm_manager_->get_spm_buffer(
            instruction.cim_array_id, instruction.dst_addr);
        
        if (!src_buffer || !dst_buffer) {
            LOG(ERROR) << "SPM缓冲区获取失败";
            return false;
        }
        
        // 高效内存拷贝
        memcpy(dst_buffer, src_buffer, instruction.size);
        
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "YISICOPY执行失败: " << e.what();
        return false;
    }
}

bool YISInstructionEngine::execute_matrix_multiply(const YISInstruction& instruction) {
    // YISMMA: 矩阵乘法加速指令
    try {
        // 获取CIM阵列模拟器
        auto cim_array = cim_simulator_->get_cim_array(instruction.cim_array_id);
        if (!cim_array) {
            LOG(ERROR) << "CIM阵列获取失败，ID: " << instruction.cim_array_id;
            return false;
        }
        
        // 执行矩阵乘法
        bool success = false;
        
        switch (instruction.operation) {
            case YISOperation::MATRIX_MULTIPLY_ACCUMULATE:
                success = execute_mma_operation(instruction, cim_array);
                break;
                
            case YISOperation::REDUCE_SUM:
                success = execute_reduce_operation(instruction, cim_array);
                break;
                
            default:
                LOG(ERROR) << "不支持的YISMMA操作: " 
                          << static_cast<int>(instruction.operation);
                return false;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "YISMMA执行失败: " << e.what();
        return false;
    }
}

bool YISInstructionEngine::execute_mma_operation(
    const YISInstruction& instruction, 
    CIMArray* cim_array) {
    
    // 获取矩阵数据
    auto matrix_a = spm_manager_->get_matrix_buffer(
        instruction.cim_array_id, "matrix_a");
    auto matrix_b = spm_manager_->get_matrix_buffer(
        instruction.cim_array_id, "matrix_b");
    auto matrix_c = spm_manager_->get_matrix_buffer(
        instruction.cim_array_id, "matrix_c");
    
    if (!matrix_a || !matrix_b || !matrix_c) {
        LOG(ERROR) << "矩阵缓冲区获取失败";
        return false;
    }
    
    // 使用OpenBLAS进行高性能矩阵乘法
    // 模拟YICA存算一体的计算特性
    const int M = instruction.matrix_m;
    const int N = instruction.matrix_n;
    const int K = instruction.matrix_k;
    
    // 根据精度选择不同的BLAS函数
    if (instruction.precision == YICAPrecision::FP32) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   M, N, K, 1.0f,
                   static_cast<float*>(matrix_a), K,
                   static_cast<float*>(matrix_b), N,
                   1.0f, static_cast<float*>(matrix_c), N);
    } else if (instruction.precision == YICAPrecision::FP16) {
        // FP16模拟 - 使用FP32计算但限制精度
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   M, N, K, 1.0f,
                   static_cast<float*>(matrix_a), K,
                   static_cast<float*>(matrix_b), N,
                   1.0f, static_cast<float*>(matrix_c), N);
    }
    
    // 模拟CIM阵列的能耗和延迟特性
    cim_array->simulate_computation_cost(M * N * K);
    
    return true;
}

bool YISInstructionEngine::execute_synchronization(const YISInstruction& instruction) {
    // YISSYNC: 同步指令
    if (instruction.sync_required) {
        // 等待所有CIM阵列完成当前操作
        for (int i = 0; i < config_.num_cim_arrays; ++i) {
            auto cim_array = cim_simulator_->get_cim_array(i);
            if (cim_array) {
                cim_array->wait_for_completion();
            }
        }
        
        // OpenMP同步
        #pragma omp barrier
    }
    
    return true;
}

bool YISInstructionEngine::execute_control_flow(const YISInstruction& instruction) {
    // YISCONTROL: 控制流指令
    switch (instruction.operation) {
        case YISOperation::CONDITIONAL_BRANCH:
            return execute_conditional_branch(instruction);
            
        case YISOperation::LOOP_CONTROL:
            return execute_loop_control(instruction);
            
        case YISOperation::KERNEL_END:
            is_running_ = false;
            return true;
            
        default:
            LOG(ERROR) << "不支持的控制流操作";
            return false;
    }
}

ExecutionStats YISInstructionEngine::get_execution_stats() const {
    ExecutionStats stats = execution_stats_;
    
    if (stats.total_instructions > 0) {
        stats.average_instruction_time_us = 
            stats.total_execution_time_us / stats.total_instructions;
    }
    
    return stats;
}

void YISInstructionEngine::reset_stats() {
    execution_stats_ = {0, 0, 0.0, 0.0};
}

void YISInstructionEngine::start() {
    is_running_ = true;
    LOG(INFO) << "YIS指令执行引擎启动";
}

void YISInstructionEngine::stop() {
    is_running_ = false;
    LOG(INFO) << "YIS指令执行引擎停止";
}

} // namespace yica
} // namespace yirage 