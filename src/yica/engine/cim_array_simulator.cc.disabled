/**
 * @file cim_array_simulator.cc
 * @brief CIM 阵列模拟器实现
 * 
 * 实现 YICA CIM 阵列的精确模拟
 */

#include "yirage/yica/engine/cim_array_simulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>

namespace yirage {
namespace yica {

CIMArraySimulator::CIMArraySimulator(const YICAConfig& config)
    : config_(config), array_id_(0), state_(CIMArrayState::IDLE),
      debug_mode_(false), shutdown_requested_(false) {
    
    // 初始化物理模型参数
    physical_model_.base_latency_us = 0.1;  // 100ns 基础延迟
    physical_model_.compute_latency_factor = 1.0;
    physical_model_.memory_latency_factor = 0.5;
    physical_model_.base_power_w = 2.0;  // 2W 基础功耗
    physical_model_.dynamic_power_factor = 1.5;
    physical_model_.temperature_factor = 0.8;
    
    // 初始化性能指标
    metrics_ = {};
    metrics_.utilization_rate = 0.0;
    metrics_.throughput_gops = 0.0;
    metrics_.power_consumption_w = physical_model_.base_power_w;
    metrics_.temperature_celsius = 25.0;  // 室温
    metrics_.total_operations = 0;
    metrics_.average_latency_us = 0.0;
    metrics_.energy_efficiency_tops_w = 0.0;
    
    last_activity_time_ = std::chrono::steady_clock::now();
    
    std::cout << "[CIM Array " << array_id_ << "] Initialized" << std::endl;
}

CIMArraySimulator::~CIMArraySimulator() {
    shutdown();
}

bool CIMArraySimulator::initialize() {
    if (state_.load() != CIMArrayState::IDLE) {
        return true;  // 已经初始化
    }
    
    std::cout << "[CIM Array " << array_id_ << "] Initializing..." << std::endl;
    
    // 分配内部缓冲区
    size_t buffer_size = 1024 * 1024;  // 1MB 缓冲区
    internal_buffer_a_.resize(buffer_size);
    internal_buffer_b_.resize(buffer_size);
    internal_buffer_c_.resize(buffer_size);
    
    // 启动工作线程
    size_t num_workers = std::min(4u, std::thread::hardware_concurrency());
    for (size_t i = 0; i < num_workers; ++i) {
        worker_threads_.emplace_back(&CIMArraySimulator::worker_thread_function, this);
    }
    
    state_.store(CIMArrayState::IDLE);
    std::cout << "[CIM Array " << array_id_ << "] Successfully initialized with " 
              << num_workers << " workers" << std::endl;
    
    return true;
}

void CIMArraySimulator::shutdown() {
    if (shutdown_requested_.load()) {
        return;
    }
    
    std::cout << "[CIM Array " << array_id_ << "] Shutting down..." << std::endl;
    
    shutdown_requested_.store(true);
    task_cv_.notify_all();
    
    // 等待所有工作线程结束
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    state_.store(CIMArrayState::IDLE);
    std::cout << "[CIM Array " << array_id_ << "] Shutdown complete" << std::endl;
}

bool CIMArraySimulator::submit_task(const CIMComputeTask& task) {
    std::unique_lock<std::mutex> lock(task_mutex_);
    
    if (task_queue_.size() >= MAX_TASK_QUEUE_SIZE) {
        return false;  // 队列已满
    }
    
    task_queue_.push(task);
    task_cv_.notify_one();
    
    return true;
}

double CIMArraySimulator::execute_matrix_multiply(
    uint32_t m, uint32_t n, uint32_t k,
    const void* a, const void* b, void* c,
    CIMPrecisionMode precision, bool accumulate) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (debug_mode_.load()) {
        std::cout << "[CIM Array " << array_id_ << "] Matrix multiply " 
                  << m << "x" << n << "x" << k << std::endl;
    }
    
    // 更新状态
    state_.store(CIMArrayState::COMPUTING);
    last_activity_time_ = std::chrono::steady_clock::now();
    
    // 计算延迟模型
    double latency_us = compute_matrix_multiply_latency(m, n, k, precision);
    
    // 模拟计算时间
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(latency_us)));
    
    // 简化的矩阵乘法实现（用于验证）
    if (precision == CIMPrecisionMode::FP32) {
        const float* fa = static_cast<const float*>(a);
        const float* fb = static_cast<const float*>(b);
        float* fc = static_cast<float*>(c);
        
        // 执行矩阵乘法 C = A * B (或 C += A * B 如果accumulate为true)
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                float sum = accumulate ? fc[i * n + j] : 0.0f;
                for (uint32_t l = 0; l < k; ++l) {
                    sum += fa[i * k + l] * fb[l * n + j];
                }
                fc[i * n + j] = sum;
            }
        }
    }
    
    // 更新性能指标
    auto end_time = std::chrono::high_resolution_clock::now();
    auto actual_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    update_metrics({}, actual_duration);
    update_power_consumption(CIMComputeType::MATRIX_MULTIPLY, actual_duration);
    
    state_.store(CIMArrayState::IDLE);
    
    return actual_duration;
}

double CIMArraySimulator::execute_vector_operation(
    uint32_t size,
    const void* a, const void* b, void* c,
    CIMComputeType compute_type,
    CIMPrecisionMode precision) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (debug_mode_.load()) {
        std::cout << "[CIM Array " << array_id_ << "] Vector operation, size=" << size << std::endl;
    }
    
    state_.store(CIMArrayState::COMPUTING);
    last_activity_time_ = std::chrono::steady_clock::now();
    
    // 计算延迟模型
    double latency_us = compute_vector_operation_latency(size, compute_type, precision);
    
    // 模拟计算时间
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(latency_us)));
    
    // 简化的向量运算实现
    if (precision == CIMPrecisionMode::FP32) {
        const float* fa = static_cast<const float*>(a);
        const float* fb = static_cast<const float*>(b);
        float* fc = static_cast<float*>(c);
        
        switch (compute_type) {
            case CIMComputeType::VECTOR_ADD:
                for (uint32_t i = 0; i < size; ++i) {
                    fc[i] = fa[i] + fb[i];
                }
                break;
                
            case CIMComputeType::VECTOR_MUL:
                for (uint32_t i = 0; i < size; ++i) {
                    fc[i] = fa[i] * fb[i];
                }
                break;
                
            default:
                break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto actual_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    update_metrics({}, actual_duration);
    update_power_consumption(compute_type, actual_duration);
    
    state_.store(CIMArrayState::IDLE);
    
    return actual_duration;
}

double CIMArraySimulator::execute_activation(
    uint32_t size,
    const void* input, void* output,
    uint32_t activation_type,
    CIMPrecisionMode precision) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    state_.store(CIMArrayState::COMPUTING);
    last_activity_time_ = std::chrono::steady_clock::now();
    
    // 计算延迟模型
    double latency_us = compute_activation_latency(size, activation_type, precision);
    
    // 模拟计算时间
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(latency_us)));
    
    // 简化的激活函数实现
    if (precision == CIMPrecisionMode::FP32) {
        const float* fin = static_cast<const float*>(input);
        float* fout = static_cast<float*>(output);
        
        switch (activation_type) {
            case 0:  // ReLU
                for (uint32_t i = 0; i < size; ++i) {
                    fout[i] = std::max(0.0f, fin[i]);
                }
                break;
                
            case 1:  // GELU
                for (uint32_t i = 0; i < size; ++i) {
                    float x = fin[i];
                    fout[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
                }
                break;
                
            case 2:  // SiLU
                for (uint32_t i = 0; i < size; ++i) {
                    float x = fin[i];
                    fout[i] = x / (1.0f + std::exp(-x));
                }
                break;
                
            default:
                // 默认为恒等函数
                for (uint32_t i = 0; i < size; ++i) {
                    fout[i] = fin[i];
                }
                break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto actual_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    update_metrics({}, actual_duration);
    update_power_consumption(CIMComputeType::ACTIVATION, actual_duration);
    
    state_.store(CIMArrayState::IDLE);
    
    return actual_duration;
}

CIMArrayState CIMArraySimulator::get_state() const {
    return state_.load();
}

CIMArrayMetrics CIMArraySimulator::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

double CIMArraySimulator::get_utilization() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_.utilization_rate;
}

double CIMArraySimulator::get_power_consumption() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_.power_consumption_w;
}

bool CIMArraySimulator::wait_for_completion(uint32_t timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    while (state_.load() != CIMArrayState::IDLE) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        
        if (elapsed >= timeout_ms) {
            return false;  // 超时
        }
    }
    
    return true;
}

void CIMArraySimulator::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.total_operations = 0;
    metrics_.average_latency_us = 0.0;
    metrics_.utilization_rate = 0.0;
    metrics_.throughput_gops = 0.0;
    
    last_activity_time_ = std::chrono::steady_clock::now();
}

void CIMArraySimulator::set_debug_mode(bool enable) {
    debug_mode_.store(enable);
}

// 私有方法实现

void CIMArraySimulator::worker_thread_function() {
    while (!shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(task_mutex_);
        
        task_cv_.wait(lock, [this] {
            return !task_queue_.empty() || shutdown_requested_.load();
        });
        
        if (shutdown_requested_.load()) {
            break;
        }
        
        if (task_queue_.empty()) {
            continue;
        }
        
        CIMComputeTask task = task_queue_.front();
        task_queue_.pop();
        lock.unlock();
        
        // 执行任务
        bool success = execute_task_internal(task);
        
        // 调用完成回调
        if (task.completion_callback) {
            task.completion_callback(success, metrics_.average_latency_us);
        }
    }
}

bool CIMArraySimulator::execute_task_internal(const CIMComputeTask& task) {
    switch (task.compute_type) {
        case CIMComputeType::MATRIX_MULTIPLY:
            execute_matrix_multiply(task.m, task.n, task.k, 
                                   task.input_a, task.input_b, task.output,
                                   task.precision, task.accumulate);
            return true;
            
        case CIMComputeType::VECTOR_ADD:
        case CIMComputeType::VECTOR_MUL:
            execute_vector_operation(task.m * task.n, 
                                   task.input_a, task.input_b, task.output,
                                   task.compute_type, task.precision);
            return true;
            
        case CIMComputeType::ACTIVATION:
            execute_activation(task.m * task.n,
                             task.input_a, task.output, 0,  // ReLU
                             task.precision);
            return true;
            
        default:
            return false;
    }
}

double CIMArraySimulator::compute_matrix_multiply_latency(uint32_t m, uint32_t n, uint32_t k, 
                                                        CIMPrecisionMode precision) const {
    // 基础延迟模型：基于矩阵大小和精度
    double ops = static_cast<double>(m) * n * k * 2;  // 乘法+加法
    double precision_factor = get_precision_factor(precision);
    
    // 基础延迟 + 计算延迟
    double latency = physical_model_.base_latency_us + 
                    (ops / 1000.0) * physical_model_.compute_latency_factor * precision_factor;
    
    return latency;
}

double CIMArraySimulator::compute_vector_operation_latency(uint32_t size, 
                                                         CIMComputeType type,
                                                         CIMPrecisionMode precision) const {
    double ops = static_cast<double>(size);
    double precision_factor = get_precision_factor(precision);
    
    // 向量操作相对简单，延迟较低
    double latency = physical_model_.base_latency_us + 
                    (ops / 10000.0) * physical_model_.compute_latency_factor * precision_factor;
    
    return latency;
}

double CIMArraySimulator::compute_activation_latency(uint32_t size, uint32_t activation_type,
                                                   CIMPrecisionMode precision) const {
    double ops = static_cast<double>(size);
    double precision_factor = get_precision_factor(precision);
    
    // 激活函数的复杂度因子
    double complexity_factor = 1.0;
    switch (activation_type) {
        case 0:  // ReLU
            complexity_factor = 0.5;
            break;
        case 1:  // GELU
            complexity_factor = 2.0;
            break;
        case 2:  // SiLU
            complexity_factor = 1.5;
            break;
        default:
            complexity_factor = 1.0;
            break;
    }
    
    double latency = physical_model_.base_latency_us + 
                    (ops / 5000.0) * physical_model_.compute_latency_factor * 
                    precision_factor * complexity_factor;
    
    return latency;
}

void CIMArraySimulator::update_power_consumption(CIMComputeType type, double duration_us) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // 动态功耗计算
    double dynamic_power = physical_model_.base_power_w * physical_model_.dynamic_power_factor;
    
    switch (type) {
        case CIMComputeType::MATRIX_MULTIPLY:
            dynamic_power *= 1.5;  // 矩阵乘法功耗较高
            break;
        case CIMComputeType::VECTOR_ADD:
        case CIMComputeType::VECTOR_MUL:
            dynamic_power *= 1.0;  // 向量操作功耗适中
            break;
        case CIMComputeType::ACTIVATION:
            dynamic_power *= 0.8;  // 激活函数功耗较低
            break;
        default:
            break;
    }
    
    metrics_.power_consumption_w = physical_model_.base_power_w + dynamic_power;
    
    // 更新温度
    update_temperature(metrics_.power_consumption_w, duration_us);
}

void CIMArraySimulator::update_temperature(double power_w, double duration_us) {
    // 简化的温度模型
    double heat_generated = power_w * (duration_us / 1000000.0);  // 转换为秒
    double temperature_rise = heat_generated * physical_model_.temperature_factor;
    
    metrics_.temperature_celsius += temperature_rise;
    
    // 温度衰减
    metrics_.temperature_celsius *= TEMPERATURE_DECAY_FACTOR;
    
    // 限制温度范围
    metrics_.temperature_celsius = std::max(25.0, 
                                          std::min(85.0, metrics_.temperature_celsius));
}

double CIMArraySimulator::get_precision_factor(CIMPrecisionMode precision) const {
    switch (precision) {
        case CIMPrecisionMode::INT8:
            return 0.25;
        case CIMPrecisionMode::INT16:
            return 0.5;
        case CIMPrecisionMode::FP16:
        case CIMPrecisionMode::BF16:
            return 0.5;
        case CIMPrecisionMode::FP32:
            return 1.0;
        default:
            return 1.0;
    }
}

void CIMArraySimulator::log_debug(const std::string& message) const {
    if (debug_mode_.load()) {
        std::cout << "[CIM Array " << array_id_ << "] " << message << std::endl;
    }
}

void CIMArraySimulator::update_metrics(const CIMComputeTask& task, double execution_time_us) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.total_operations++;
    
    // 更新平均延迟
    if (metrics_.total_operations == 1) {
        metrics_.average_latency_us = execution_time_us;
    } else {
        metrics_.average_latency_us = (metrics_.average_latency_us * (metrics_.total_operations - 1) + 
                                     execution_time_us) / metrics_.total_operations;
    }
    
    // 更新利用率（简化计算）
    auto now = std::chrono::steady_clock::now();
    auto idle_time = std::chrono::duration_cast<std::chrono::microseconds>(
        now - last_activity_time_).count();
    
    if (idle_time > 0) {
        metrics_.utilization_rate = execution_time_us / (execution_time_us + idle_time);
    }
    
    // 更新吞吐量
    if (execution_time_us > 0) {
        metrics_.throughput_gops = 1000.0 / execution_time_us;  // 简化计算
    }
    
    // 更新能效比
    if (metrics_.power_consumption_w > 0) {
        metrics_.energy_efficiency_tops_w = metrics_.throughput_gops / metrics_.power_consumption_w;
    }
}

} // namespace yica
} // namespace yirage 