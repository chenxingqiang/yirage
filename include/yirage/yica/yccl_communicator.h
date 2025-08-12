#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <future>
#include <chrono>
#include "yirage/kernel/graph.h"
#include "yirage/yica/config.h"

namespace yirage {
namespace yica {

// YCCL 通信操作类型
enum class YCCLOperation {
    ALL_REDUCE,     // 全归约
    ALL_GATHER,     // 全收集
    REDUCE_SCATTER, // 归约分散
    BROADCAST,      // 广播
    REDUCE,         // 归约
    SCATTER,        // 分散
    GATHER,         // 收集
    ALL_TO_ALL,     // 全交换
    SEND,           // 点对点发送
    RECV            // 点对点接收
};

// YCCL 归约操作类型
enum class YCCLReduceOp {
    SUM,    // 求和
    PROD,   // 乘积
    MAX,    // 最大值
    MIN,    // 最小值
    AVG     // 平均值
};

// YCCL 数据类型
enum class YCCLDataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT16,
    INT8,
    UINT8,
    BFLOAT16
};

// YCCL 通信域
enum class YCCLCommScope {
    GLOBAL,     // 全局通信域
    NODE,       // 节点内通信域
    DIE,        // Die 内通信域
    CIM_ARRAY   // CIM 阵列内通信域
};

// Die 网格拓扑结构
struct DieMeshTopology {
    std::vector<int> mesh_dims;           // 网格维度 [x, y, z]
    int total_dies;                       // 总 Die 数量
    std::map<int, std::vector<int>> die_coords;  // Die ID 到坐标映射
    std::map<std::vector<int>, int> coord_to_die; // 坐标到 Die ID 映射
    
    // 网格邻居关系
    std::map<int, std::vector<int>> neighbors;   // Die 的邻居列表
    
    // 通信路径优化
    std::map<std::pair<int, int>, std::vector<int>> optimal_paths;
};

// YCCL 通信请求
struct YCCLRequest {
    uint64_t request_id;
    YCCLOperation operation;
    YCCLDataType data_type;
    size_t element_count;
    void* send_buffer;
    void* recv_buffer;
    int root_rank;  // 用于 broadcast/reduce
    YCCLReduceOp reduce_op;
    YCCLCommScope scope;
    
    // 异步执行状态
    bool is_completed;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::string error_message;
};

// YCCL 通信器
class YCCLCommunicator {
public:
    explicit YCCLCommunicator(const YICAConfig& config);
    ~YCCLCommunicator();
    
    // 初始化和销毁
    bool initialize(int world_size, int rank);
    void finalize();
    
    // 基本信息
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    bool is_initialized() const { return initialized_; }
    
    // 集合通信操作 (异步)
    std::future<bool> all_reduce_async(
        void* send_buffer, void* recv_buffer,
        size_t element_count, YCCLDataType data_type,
        YCCLReduceOp reduce_op, YCCLCommScope scope = YCCLCommScope::GLOBAL
    );
    
    std::future<bool> all_gather_async(
        void* send_buffer, void* recv_buffer,
        size_t element_count, YCCLDataType data_type,
        YCCLCommScope scope = YCCLCommScope::GLOBAL
    );
    
    std::future<bool> reduce_scatter_async(
        void* send_buffer, void* recv_buffer,
        size_t element_count, YCCLDataType data_type,
        YCCLReduceOp reduce_op, YCCLCommScope scope = YCCLCommScope::GLOBAL
    );
    
    std::future<bool> broadcast_async(
        void* buffer, size_t element_count, YCCLDataType data_type,
        int root_rank, YCCLCommScope scope = YCCLCommScope::GLOBAL
    );
    
    std::future<bool> all_to_all_async(
        void* send_buffer, void* recv_buffer,
        size_t element_count, YCCLDataType data_type,
        YCCLCommScope scope = YCCLCommScope::GLOBAL
    );
    
    // 点对点通信操作
    std::future<bool> send_async(
        void* buffer, size_t element_count, YCCLDataType data_type,
        int dest_rank, int tag = 0
    );
    
    std::future<bool> recv_async(
        void* buffer, size_t element_count, YCCLDataType data_type,
        int src_rank, int tag = 0
    );
    
    // 同步版本 (阻塞)
    bool all_reduce(void* send_buffer, void* recv_buffer,
                   size_t element_count, YCCLDataType data_type,
                   YCCLReduceOp reduce_op, YCCLCommScope scope = YCCLCommScope::GLOBAL);
    
    bool all_gather(void* send_buffer, void* recv_buffer,
                   size_t element_count, YCCLDataType data_type,
                   YCCLCommScope scope = YCCLCommScope::GLOBAL);
    
    bool broadcast(void* buffer, size_t element_count, YCCLDataType data_type,
                  int root_rank, YCCLCommScope scope = YCCLCommScope::GLOBAL);
    
    // 同步操作
    void synchronize();
    void barrier(YCCLCommScope scope = YCCLCommScope::GLOBAL);
    
    // 拓扑管理
    void set_mesh_topology(const DieMeshTopology& topology);
    DieMeshTopology get_mesh_topology() const { return mesh_topology_; }
    
    // 通信优化
    void enable_compression(bool enable) { compression_enabled_ = enable; }
    void set_bandwidth_limit(size_t bytes_per_second) { bandwidth_limit_ = bytes_per_second; }
    void enable_overlap_computation(bool enable) { overlap_enabled_ = enable; }
    
    // 性能监控
    struct CommunicationStats {
        size_t total_bytes_sent;
        size_t total_bytes_received;
        size_t total_operations;
        double total_communication_time;
        double average_bandwidth;
        std::map<YCCLOperation, size_t> operation_counts;
        std::map<YCCLOperation, double> operation_times;
    };
    
    CommunicationStats get_communication_stats() const;
    void reset_stats();
    
    // 错误处理
    enum class ErrorHandlingMode {
        NO_HANDLING,    // 不处理异步错误
        TEAR_DOWN,      // 遇到错误时拆除进程组
        CLEAN_UP_ONLY,  // 仅清理资源
        SKIP_CLEAN_UP   // 跳过清理
    };
    
    void set_error_handling_mode(ErrorHandlingMode mode) { error_handling_mode_ = mode; }
    
private:
    YICAConfig config_;
    int rank_;
    int world_size_;
    bool initialized_;
    
    // 网格拓扑
    DieMeshTopology mesh_topology_;
    
    // 通信优化参数
    bool compression_enabled_;
    size_t bandwidth_limit_;
    bool overlap_enabled_;
    
    // 错误处理
    ErrorHandlingMode error_handling_mode_;
    
    // 性能统计
    mutable CommunicationStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 请求管理
    uint64_t next_request_id_;
    std::map<uint64_t, std::unique_ptr<YCCLRequest>> active_requests_;
    std::mutex requests_mutex_;
    
    // 内部实现方法
    uint64_t generate_request_id();
    void update_stats(YCCLOperation op, size_t bytes, double time);
    
    // 拓扑相关方法
    std::vector<int> get_optimal_path(int src_die, int dest_die);
    void optimize_communication_schedule();
    
    // 通信算法实现
    bool execute_ring_all_reduce(void* send_buffer, void* recv_buffer,
                                size_t element_count, YCCLDataType data_type,
                                YCCLReduceOp reduce_op);
    
    bool execute_tree_broadcast(void* buffer, size_t element_count,
                               YCCLDataType data_type, int root_rank);
    
    bool execute_butterfly_all_to_all(void* send_buffer, void* recv_buffer,
                                     size_t element_count, YCCLDataType data_type);
    
    // 数据类型工具
    size_t get_data_type_size(YCCLDataType data_type);
    void apply_reduce_operation(void* a, void* b, void* result,
                               size_t count, YCCLDataType data_type,
                               YCCLReduceOp reduce_op);
};

// YCCL 通信组管理器
class YCCLCommGroup {
public:
    explicit YCCLCommGroup(const std::vector<int>& ranks);
    ~YCCLCommGroup();
    
    // 创建子通信组
    static std::unique_ptr<YCCLCommGroup> create_subgroup(
        const YCCLCommunicator& parent_comm,
        const std::vector<int>& ranks
    );
    
    // 通信组信息
    int get_group_size() const { return group_ranks_.size(); }
    int get_group_rank(int global_rank) const;
    std::vector<int> get_group_ranks() const { return group_ranks_; }
    
    // 通信组操作
    bool all_reduce_group(YCCLCommunicator& comm,
                         void* send_buffer, void* recv_buffer,
                         size_t element_count, YCCLDataType data_type,
                         YCCLReduceOp reduce_op);
    
    bool broadcast_group(YCCLCommunicator& comm,
                        void* buffer, size_t element_count,
                        YCCLDataType data_type, int root_group_rank);

private:
    std::vector<int> group_ranks_;
    std::map<int, int> global_to_group_rank_;
};

// YCCL 工具函数
namespace yccl_utils {
    
    // 拓扑构建
    DieMeshTopology create_2d_mesh_topology(int mesh_x, int mesh_y);
    DieMeshTopology create_3d_mesh_topology(int mesh_x, int mesh_y, int mesh_z);
    DieMeshTopology create_torus_topology(int mesh_x, int mesh_y);
    
    // 通信算法选择
    std::string select_optimal_algorithm(YCCLOperation op, int world_size,
                                        size_t message_size, const DieMeshTopology& topology);
    
    // 性能预测
    double estimate_communication_time(YCCLOperation op, size_t message_size,
                                      int world_size, const DieMeshTopology& topology);
    
    // 数据类型转换
    // YCCLDataType torch_dtype_to_yccl(torch::ScalarType dtype); // TODO: Add torch dependency
    // torch::ScalarType yccl_dtype_to_torch(YCCLDataType dtype); // TODO: Add torch dependency
    
    // 调试和诊断
    void print_topology_info(const DieMeshTopology& topology);
    void print_communication_stats(const YCCLCommunicator::CommunicationStats& stats);
    
} // namespace yccl_utils

} // namespace yica
} // namespace yirage 