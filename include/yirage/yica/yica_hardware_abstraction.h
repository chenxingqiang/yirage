/**
 * @file yica_hardware_abstraction.h
 * @brief YICA 硬件抽象层 (Hardware Abstraction Layer - HAL)
 *
 * 提供统一的硬件接口，支持不同版本的 YICA 硬件架构，
 * 包括模拟模式、真实硬件和未来的硬件版本。
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <mutex>

namespace yirage {
namespace yica {

// 前向声明
class YICADevice;
class YICAMemoryManager;
class YICAKernelExecutor;

/**
 * @brief YICA 硬件架构版本枚举
 */
enum class YICAArchitecture {
    YICA_V1_0 = 100,  ///< YICA 第一代架构
    YICA_V1_1 = 101,  ///< YICA 1.1 增强版
    YICA_V2_0 = 200,  ///< YICA 第二代架构
    SIMULATION = 999, ///< 模拟模式
    UNKNOWN = -1      ///< 未知架构
};

/**
 * @brief 硬件运行模式
 */
enum class YICAExecutionMode {
    HARDWARE,         ///< 真实硬件模式
    SIMULATION,       ///< 软件模拟模式
    HYBRID           ///< 混合模式（部分硬件 + 部分模拟）
};

/**
 * @brief CIM 阵列类型
 */
enum class CIMArrayType {
    STANDARD,        ///< 标准 CIM 阵列
    HIGH_PRECISION,  ///< 高精度 CIM 阵列
    LOW_POWER,       ///< 低功耗 CIM 阵列
    ADAPTIVE         ///< 自适应 CIM 阵列
};

/**
 * @brief 内存层次结构类型
 */
enum class MemoryLevel {
    REGISTER_FILE = 0,  ///< 寄存器文件
    SPM = 1,           ///< 暂存器内存 (Scratchpad Memory)
    DRAM = 2           ///< 主内存
};

/**
 * @brief 硬件能力描述符
 */
struct YICACapabilities {
    // 基础架构信息
    YICAArchitecture architecture;
    std::string version_string;
    uint32_t revision_number;

    // 计算能力
    uint32_t num_cim_arrays;
    uint32_t cim_array_size_x;
    uint32_t cim_array_size_y;
    std::vector<CIMArrayType> supported_cim_types;

    // 内存层次
    uint64_t register_file_size;
    uint64_t spm_size_per_die;
    uint64_t dram_total_size;
    uint32_t memory_bus_width;

    // 性能参数
    float peak_compute_tops;      ///< 峰值计算性能 (TOPS)
    float memory_bandwidth_gbps;  ///< 内存带宽 (GB/s)
    uint32_t max_frequency_mhz;   ///< 最大工作频率 (MHz)

    // 特性支持
    bool supports_mixed_precision;
    bool supports_sparsity;
    bool supports_compression;
    bool supports_distributed;
    bool supports_dynamic_voltage;

    // 指令集
    std::vector<std::string> supported_instruction_sets;
    uint32_t instruction_cache_size;

    // 互连拓扑
    std::string interconnect_topology;
    uint32_t num_dies;
    uint32_t dies_per_package;
};

/**
 * @brief 硬件状态信息
 */
struct YICAHardwareStatus {
    bool is_available;
    bool is_initialized;
    float temperature_celsius;
    float power_consumption_watts;
    float utilization_percentage;
    uint64_t total_operations_executed;
    uint64_t total_errors;
    std::string last_error_message;
};

/**
 * @brief 性能计数器
 */
struct YICAPerformanceCounters {
    uint64_t cim_operations;
    uint64_t memory_transactions;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t stall_cycles;
    uint64_t active_cycles;
    float energy_consumed_joules;
};

/**
 * @brief 抽象硬件接口基类
 */
class YICAHardwareInterface {
public:
    virtual ~YICAHardwareInterface() = default;

    // 基础硬件管理
    virtual bool initialize() = 0;
    virtual bool is_available() const = 0;
    virtual void shutdown() = 0;
    virtual void reset() = 0;

    // 硬件信息查询
    virtual YICACapabilities get_capabilities() const = 0;
    virtual YICAHardwareStatus get_status() const = 0;
    virtual YICAPerformanceCounters get_performance_counters() const = 0;
    virtual std::string get_device_info() const = 0;

    // 内存管理
    virtual void* allocate_memory(size_t size, MemoryLevel level) = 0;
    virtual void deallocate_memory(void* ptr, MemoryLevel level) = 0;
    virtual bool copy_memory(void* dst, const void* src, size_t size,
                           MemoryLevel dst_level, MemoryLevel src_level) = 0;

    // 内核执行
    virtual bool execute_kernel(const std::string& kernel_name,
                              const std::vector<void*>& args,
                              const std::vector<size_t>& arg_sizes) = 0;
    virtual bool load_kernel(const std::string& kernel_name,
                           const std::string& kernel_code) = 0;

    // 同步控制
    virtual void synchronize() = 0;
    virtual bool wait_for_completion(uint32_t timeout_ms = 0) = 0;

    // 错误处理
    virtual std::string get_last_error() const = 0;
    virtual void clear_errors() = 0;

    // 调试和分析
    virtual void enable_profiling(bool enable) = 0;
    virtual std::string get_profiling_data() const = 0;
};

/**
 * @brief 实际硬件实现类
 */
class YICARealHardware : public YICAHardwareInterface {
private:
    YICACapabilities capabilities_;
    YICAHardwareStatus status_;
    YICAPerformanceCounters counters_;
    std::string last_error_;
    bool profiling_enabled_;
    mutable std::mutex hardware_mutex_;

    // 硬件特定的私有数据
    void* hardware_context_;
    std::unordered_map<std::string, void*> loaded_kernels_;

public:
    YICARealHardware();
    ~YICARealHardware() override;

    // 硬件接口实现
    bool initialize() override;
    bool is_available() const override;
    void shutdown() override;
    void reset() override;

    YICACapabilities get_capabilities() const override;
    YICAHardwareStatus get_status() const override;
    YICAPerformanceCounters get_performance_counters() const override;
    std::string get_device_info() const override;

    void* allocate_memory(size_t size, MemoryLevel level) override;
    void deallocate_memory(void* ptr, MemoryLevel level) override;
    bool copy_memory(void* dst, const void* src, size_t size,
                   MemoryLevel dst_level, MemoryLevel src_level) override;

    bool execute_kernel(const std::string& kernel_name,
                      const std::vector<void*>& args,
                      const std::vector<size_t>& arg_sizes) override;
    bool load_kernel(const std::string& kernel_name,
                   const std::string& kernel_code) override;

    void synchronize() override;
    bool wait_for_completion(uint32_t timeout_ms = 0) override;

    std::string get_last_error() const override;
    void clear_errors() override;

    void enable_profiling(bool enable) override;
    std::string get_profiling_data() const override;

private:
    // 内部辅助方法
    bool detect_hardware();
    bool initialize_hardware_context();
    bool configure_memory_hierarchy();
    bool load_firmware();
    void update_performance_counters() const;
};

/**
 * @brief 模拟硬件实现类
 */
class YICASimulationHardware : public YICAHardwareInterface {
private:
    YICACapabilities capabilities_;
    YICAHardwareStatus status_;
    YICAPerformanceCounters counters_;
    std::string last_error_;
    bool profiling_enabled_;

    // 模拟相关的数据结构
    std::unordered_map<MemoryLevel, std::vector<uint8_t>> simulated_memory_;
    std::unordered_map<std::string, std::function<bool(const std::vector<void*>&)>> kernel_simulators_;

public:
    YICASimulationHardware();
    ~YICASimulationHardware() override = default;

    // 硬件接口实现（模拟版本）
    bool initialize() override;
    bool is_available() const override;
    void shutdown() override;
    void reset() override;

    YICACapabilities get_capabilities() const override;
    YICAHardwareStatus get_status() const override;
    YICAPerformanceCounters get_performance_counters() const override;
    std::string get_device_info() const override;

    void* allocate_memory(size_t size, MemoryLevel level) override;
    void deallocate_memory(void* ptr, MemoryLevel level) override;
    bool copy_memory(void* dst, const void* src, size_t size,
                   MemoryLevel dst_level, MemoryLevel src_level) override;

    bool execute_kernel(const std::string& kernel_name,
                      const std::vector<void*>& args,
                      const std::vector<size_t>& arg_sizes) override;
    bool load_kernel(const std::string& kernel_name,
                   const std::string& kernel_code) override;

    void synchronize() override;
    bool wait_for_completion(uint32_t timeout_ms = 0) override;

    std::string get_last_error() const override;
    void clear_errors() override;

    void enable_profiling(bool enable) override;
    std::string get_profiling_data() const override;

private:
    // 模拟器特定方法
    void initialize_simulation_capabilities();
    void register_default_kernel_simulators();
    bool simulate_kernel_execution(const std::string& kernel_name,
                                 const std::vector<void*>& args);
    void simulate_performance_impact(const std::string& operation, size_t data_size);
};

/**
 * @brief 硬件抽象层管理器
 */
class YICAHardwareManager {
private:
    static std::unique_ptr<YICAHardwareManager> instance_;
    static std::mutex instance_mutex_;

    std::vector<std::unique_ptr<YICAHardwareInterface>> hardware_interfaces_;
    YICAExecutionMode execution_mode_;
    uint32_t active_device_id_;

    YICAHardwareManager();

public:
    static YICAHardwareManager& getInstance();
    ~YICAHardwareManager() = default;

    // 禁止拷贝和移动
    YICAHardwareManager(const YICAHardwareManager&) = delete;
    YICAHardwareManager& operator=(const YICAHardwareManager&) = delete;
    YICAHardwareManager(YICAHardwareManager&&) = delete;
    YICAHardwareManager& operator=(YICAHardwareManager&&) = delete;

    // 硬件管理
    bool initialize();
    void shutdown();
    uint32_t get_device_count() const;

    // 设备选择和管理
    bool set_active_device(uint32_t device_id);
    uint32_t get_active_device() const;
    YICAHardwareInterface* get_active_hardware();
    YICAHardwareInterface* get_hardware(uint32_t device_id);

    // 执行模式管理
    YICAExecutionMode get_execution_mode() const;
    bool set_execution_mode(YICAExecutionMode mode);

    // 硬件发现和兼容性
    std::vector<YICACapabilities> enumerate_hardware();
    bool is_hardware_compatible(YICAArchitecture arch) const;
    YICAArchitecture detect_architecture() const;

    // 全局操作
    void synchronize_all_devices();
    std::vector<YICAHardwareStatus> get_all_device_status() const;

private:
    // 内部方法
    bool detect_and_initialize_hardware();
    void initialize_simulation_fallback();
    bool validate_hardware_compatibility();
    YICAExecutionMode determine_optimal_execution_mode();
};

/**
 * @brief 便利函数和宏定义
 */

// 获取硬件管理器实例
#define YICA_HAL() ::yirage::yica::YICAHardwareManager::getInstance()

// 获取当前活跃硬件接口
#define YICA_HARDWARE() YICA_HAL().get_active_hardware()

// 硬件操作宏
#define YICA_CHECK_HARDWARE() \
    do { \
        if (!YICA_HARDWARE() || !YICA_HARDWARE()->is_available()) { \
            throw std::runtime_error("YICA hardware not available"); \
        } \
    } while(0)

// 架构版本比较函数
constexpr bool is_architecture_newer(YICAArchitecture a, YICAArchitecture b) {
    return static_cast<int>(a) > static_cast<int>(b);
}

// 架构兼容性检查
bool is_architecture_compatible(YICAArchitecture target, YICAArchitecture available);

// 能力查询辅助函数
bool has_capability(const YICACapabilities& caps, const std::string& feature);
uint32_t get_optimal_tile_size(const YICACapabilities& caps, const std::string& operation);

} // namespace yica
} // namespace yirage