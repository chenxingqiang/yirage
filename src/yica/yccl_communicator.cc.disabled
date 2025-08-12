#include "yirage/yica/yccl_communicator.h"
#include <algorithm>
#include <numeric>
#include <thread>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>

namespace yirage {
namespace yica {

YCCLCommunicator::YCCLCommunicator(const YICAConfig& config)
    : config_(config), rank_(-1), world_size_(0), initialized_(false),
      compression_enabled_(false), bandwidth_limit_(0), overlap_enabled_(true),
      error_handling_mode_(ErrorHandlingMode::CLEAN_UP_ONLY),
      next_request_id_(1) {
    
    // 初始化统计信息
    stats_.total_bytes_sent = 0;
    stats_.total_bytes_received = 0;
    stats_.total_operations = 0;
    stats_.total_communication_time = 0.0;
    stats_.average_bandwidth = 0.0;
}

YCCLCommunicator::~YCCLCommunicator() {
    if (initialized_) {
        finalize();
    }
}

bool YCCLCommunicator::initialize(int world_size, int rank) {
    if (initialized_) {
        return false;
    }
    
    world_size_ = world_size;
    rank_ = rank;
    
    // 创建默认的 2D 网格拓扑
    int mesh_x = static_cast<int>(std::sqrt(world_size));
    int mesh_y = (world_size + mesh_x - 1) / mesh_x;
    mesh_topology_ = yccl_utils::create_2d_mesh_topology(mesh_x, mesh_y);
    
    initialized_ = true;
    
    std::cout << "YCCL initialized: rank=" << rank_ << ", world_size=" << world_size_
              << ", mesh=" << mesh_x << "x" << mesh_y << std::endl;
    
    return true;
}

void YCCLCommunicator::finalize() {
    if (!initialized_) {
        return;
    }
    
    // 等待所有异步操作完成
    synchronize();
    
    // 清理活跃请求
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        active_requests_.clear();
    }
    
    initialized_ = false;
    std::cout << "YCCL finalized for rank " << rank_ << std::endl;
}

std::future<bool> YCCLCommunicator::all_reduce_async(
    void* send_buffer, void* recv_buffer, size_t element_count,
    YCCLDataType data_type, YCCLReduceOp reduce_op, YCCLCommScope scope) {
    
    auto request_id = generate_request_id();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 创建异步任务
    return std::async(std::launch::async, [=]() -> bool {
        try {
            bool success = execute_ring_all_reduce(send_buffer, recv_buffer,
                                                  element_count, data_type, reduce_op);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end_time - start_time).count();
            size_t bytes = element_count * get_data_type_size(data_type);
            
            update_stats(YCCLOperation::ALL_REDUCE, bytes, duration);
            
            return success;
        } catch (const std::exception& e) {
            std::cerr << "AllReduce async error: " << e.what() << std::endl;
            return false;
        }
    });
}

std::future<bool> YCCLCommunicator::all_gather_async(
    void* send_buffer, void* recv_buffer, size_t element_count,
    YCCLDataType data_type, YCCLCommScope scope) {
    
    auto request_id = generate_request_id();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    return std::async(std::launch::async, [=]() -> bool {
        try {
            // 实现环形 AllGather 算法
            size_t element_size = get_data_type_size(data_type);
            size_t chunk_size = element_count * element_size;
            
            // 复制自己的数据到接收缓冲区
            char* recv_ptr = static_cast<char*>(recv_buffer);
            std::memcpy(recv_ptr + rank_ * chunk_size, send_buffer, chunk_size);
            
            // 环形收集其他进程的数据
            for (int step = 1; step < world_size_; ++step) {
                int src_rank = (rank_ - step + world_size_) % world_size_;
                int dest_rank = (rank_ + step) % world_size_;
                
                // 模拟通信：在实际实现中，这里会使用底层通信原语
                // 这里使用简化的实现
                if (src_rank != rank_) {
                    // 模拟从 src_rank 接收数据
                    // 在实际实现中，这里会是真实的网络通信
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end_time - start_time).count();
            size_t bytes = element_count * element_size * world_size_;
            
            update_stats(YCCLOperation::ALL_GATHER, bytes, duration);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "AllGather async error: " << e.what() << std::endl;
            return false;
        }
    });
}

std::future<bool> YCCLCommunicator::broadcast_async(
    void* buffer, size_t element_count, YCCLDataType data_type,
    int root_rank, YCCLCommScope scope) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    return std::async(std::launch::async, [=]() -> bool {
        try {
            bool success = execute_tree_broadcast(buffer, element_count, data_type, root_rank);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end_time - start_time).count();
            size_t bytes = element_count * get_data_type_size(data_type);
            
            update_stats(YCCLOperation::BROADCAST, bytes, duration);
            
            return success;
        } catch (const std::exception& e) {
            std::cerr << "Broadcast async error: " << e.what() << std::endl;
            return false;
        }
    });
}

bool YCCLCommunicator::all_reduce(void* send_buffer, void* recv_buffer,
                                 size_t element_count, YCCLDataType data_type,
                                 YCCLReduceOp reduce_op, YCCLCommScope scope) {
    auto future = all_reduce_async(send_buffer, recv_buffer, element_count,
                                  data_type, reduce_op, scope);
    return future.get();
}

bool YCCLCommunicator::all_gather(void* send_buffer, void* recv_buffer,
                                 size_t element_count, YCCLDataType data_type,
                                 YCCLCommScope scope) {
    auto future = all_gather_async(send_buffer, recv_buffer, element_count,
                                  data_type, scope);
    return future.get();
}

bool YCCLCommunicator::broadcast(void* buffer, size_t element_count,
                                YCCLDataType data_type, int root_rank,
                                YCCLCommScope scope) {
    auto future = broadcast_async(buffer, element_count, data_type, root_rank, scope);
    return future.get();
}

void YCCLCommunicator::synchronize() {
    // 等待所有活跃的异步操作完成
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    // 在实际实现中，这里会等待所有异步请求完成
    // 现在使用简化的实现
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void YCCLCommunicator::barrier(YCCLCommScope scope) {
    // 实现栅栏同步
    // 在实际实现中，这里会使用真实的分布式栅栏算法
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 简化的栅栏实现：使用 AllReduce 实现
    int dummy_data = 1;
    int result;
    all_reduce(&dummy_data, &result, 1, YCCLDataType::INT32, YCCLReduceOp::SUM, scope);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "Barrier completed for rank " << rank_ << " in " << duration << "s" << std::endl;
}

void YCCLCommunicator::set_mesh_topology(const DieMeshTopology& topology) {
    mesh_topology_ = topology;
    optimize_communication_schedule();
}

YCCLCommunicator::CommunicationStats YCCLCommunicator::get_communication_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // 计算平均带宽
    CommunicationStats current_stats = stats_;
    if (current_stats.total_communication_time > 0) {
        size_t total_bytes = current_stats.total_bytes_sent + current_stats.total_bytes_received;
        current_stats.average_bandwidth = total_bytes / current_stats.total_communication_time / (1024 * 1024); // MB/s
    }
    
    return current_stats;
}

void YCCLCommunicator::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_bytes_sent = 0;
    stats_.total_bytes_received = 0;
    stats_.total_operations = 0;
    stats_.total_communication_time = 0.0;
    stats_.average_bandwidth = 0.0;
    stats_.operation_counts.clear();
    stats_.operation_times.clear();
}

uint64_t YCCLCommunicator::generate_request_id() {
    return next_request_id_++;
}

void YCCLCommunicator::update_stats(YCCLOperation op, size_t bytes, double time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_operations++;
    stats_.total_communication_time += time;
    stats_.total_bytes_sent += bytes;  // 简化：假设发送和接收字节数相同
    stats_.total_bytes_received += bytes;
    
    stats_.operation_counts[op]++;
    stats_.operation_times[op] += time;
}

bool YCCLCommunicator::execute_ring_all_reduce(void* send_buffer, void* recv_buffer,
                                               size_t element_count, YCCLDataType data_type,
                                               YCCLReduceOp reduce_op) {
    size_t element_size = get_data_type_size(data_type);
    size_t total_bytes = element_count * element_size;
    
    // 初始化接收缓冲区
    std::memcpy(recv_buffer, send_buffer, total_bytes);
    
    // 环形 AllReduce 算法实现
    // 将数据分成 world_size 个块
    size_t chunk_count = element_count / world_size_;
    size_t chunk_size = chunk_count * element_size;
    
    char* recv_ptr = static_cast<char*>(recv_buffer);
    
    // Reduce-Scatter 阶段
    for (int step = 0; step < world_size_ - 1; ++step) {
        int send_rank = (rank_ - step + world_size_) % world_size_;
        int recv_rank = (rank_ - step - 1 + world_size_) % world_size_;
        
        // 计算当前处理的块
        int chunk_idx = (rank_ - step - 1 + world_size_) % world_size_;
        
        // 模拟发送数据到下一个进程
        // 模拟从上一个进程接收数据并进行归约
        // 在实际实现中，这里会使用真实的网络通信和归约操作
        
        // 简化的归约操作模拟
        if (reduce_op == YCCLReduceOp::SUM) {
            // 模拟求和操作
        }
    }
    
    // AllGather 阶段
    for (int step = 0; step < world_size_ - 1; ++step) {
        int send_rank = (rank_ + 1) % world_size_;
        int recv_rank = (rank_ - 1 + world_size_) % world_size_;
        
        // 模拟数据传播
        // 在实际实现中，这里会传播已归约的数据块
    }
    
    return true;
}

bool YCCLCommunicator::execute_tree_broadcast(void* buffer, size_t element_count,
                                             YCCLDataType data_type, int root_rank) {
    if (rank_ == root_rank) {
        // 根进程：向子节点广播数据
        int tree_arity = 2;  // 二叉树
        
        for (int child = 1; child <= tree_arity; ++child) {
            int child_rank = root_rank * tree_arity + child;
            if (child_rank < world_size_) {
                // 模拟发送数据到子节点
                // 在实际实现中，这里会使用真实的网络发送
            }
        }
    } else {
        // 非根进程：从父节点接收数据，然后向子节点转发
        int parent_rank = (rank_ - 1) / 2;
        
        // 模拟从父节点接收数据
        // 在实际实现中，这里会使用真实的网络接收
        
        // 向子节点转发数据
        int tree_arity = 2;
        for (int child = 1; child <= tree_arity; ++child) {
            int child_rank = rank_ * tree_arity + child;
            if (child_rank < world_size_) {
                // 模拟发送数据到子节点
            }
        }
    }
    
    return true;
}

std::vector<int> YCCLCommunicator::get_optimal_path(int src_die, int dest_die) {
    // 在网格拓扑中找到最优路径
    auto it = mesh_topology_.optimal_paths.find({src_die, dest_die});
    if (it != mesh_topology_.optimal_paths.end()) {
        return it->second;
    }
    
    // 如果没有预计算的路径，使用简单的曼哈顿距离路径
    std::vector<int> path;
    
    if (src_die < mesh_topology_.die_coords.size() && 
        dest_die < mesh_topology_.die_coords.size()) {
        
        auto src_coord = mesh_topology_.die_coords[src_die];
        auto dest_coord = mesh_topology_.die_coords[dest_die];
        
        // 简化的路径计算：先沿 X 轴，再沿 Y 轴
        path.push_back(src_die);
        
        int current_x = src_coord[0], current_y = src_coord[1];
        int target_x = dest_coord[0], target_y = dest_coord[1];
        
        // 沿 X 轴移动
        while (current_x != target_x) {
            current_x += (target_x > current_x) ? 1 : -1;
            std::vector<int> coord = {current_x, current_y};
            if (mesh_topology_.coord_to_die.count(coord)) {
                path.push_back(mesh_topology_.coord_to_die[coord]);
            }
        }
        
        // 沿 Y 轴移动
        while (current_y != target_y) {
            current_y += (target_y > current_y) ? 1 : -1;
            std::vector<int> coord = {current_x, current_y};
            if (mesh_topology_.coord_to_die.count(coord)) {
                path.push_back(mesh_topology_.coord_to_die[coord]);
            }
        }
    }
    
    return path;
}

void YCCLCommunicator::optimize_communication_schedule() {
    // 预计算最优通信路径
    mesh_topology_.optimal_paths.clear();
    
    for (int src = 0; src < mesh_topology_.total_dies; ++src) {
        for (int dest = 0; dest < mesh_topology_.total_dies; ++dest) {
            if (src != dest) {
                auto path = get_optimal_path(src, dest);
                mesh_topology_.optimal_paths[{src, dest}] = path;
            }
        }
    }
    
    std::cout << "Communication schedule optimized for " << mesh_topology_.total_dies 
              << " dies" << std::endl;
}

size_t YCCLCommunicator::get_data_type_size(YCCLDataType data_type) {
    switch (data_type) {
        case YCCLDataType::FLOAT32: return 4;
        case YCCLDataType::FLOAT16: return 2;
        case YCCLDataType::BFLOAT16: return 2;
        case YCCLDataType::INT32: return 4;
        case YCCLDataType::INT16: return 2;
        case YCCLDataType::INT8: return 1;
        case YCCLDataType::UINT8: return 1;
        default: return 4;  // 默认 4 字节
    }
}

void YCCLCommunicator::apply_reduce_operation(void* a, void* b, void* result,
                                             size_t count, YCCLDataType data_type,
                                             YCCLReduceOp reduce_op) {
    // 根据数据类型和归约操作执行实际的归约计算
    switch (data_type) {
        case YCCLDataType::FLOAT32: {
            float* fa = static_cast<float*>(a);
            float* fb = static_cast<float*>(b);
            float* fresult = static_cast<float*>(result);
            
            for (size_t i = 0; i < count; ++i) {
                switch (reduce_op) {
                    case YCCLReduceOp::SUM:
                        fresult[i] = fa[i] + fb[i];
                        break;
                    case YCCLReduceOp::PROD:
                        fresult[i] = fa[i] * fb[i];
                        break;
                    case YCCLReduceOp::MAX:
                        fresult[i] = std::max(fa[i], fb[i]);
                        break;
                    case YCCLReduceOp::MIN:
                        fresult[i] = std::min(fa[i], fb[i]);
                        break;
                    case YCCLReduceOp::AVG:
                        fresult[i] = (fa[i] + fb[i]) / 2.0f;
                        break;
                }
            }
            break;
        }
        case YCCLDataType::INT32: {
            int* ia = static_cast<int*>(a);
            int* ib = static_cast<int*>(b);
            int* iresult = static_cast<int*>(result);
            
            for (size_t i = 0; i < count; ++i) {
                switch (reduce_op) {
                    case YCCLReduceOp::SUM:
                        iresult[i] = ia[i] + ib[i];
                        break;
                    case YCCLReduceOp::PROD:
                        iresult[i] = ia[i] * ib[i];
                        break;
                    case YCCLReduceOp::MAX:
                        iresult[i] = std::max(ia[i], ib[i]);
                        break;
                    case YCCLReduceOp::MIN:
                        iresult[i] = std::min(ia[i], ib[i]);
                        break;
                    case YCCLReduceOp::AVG:
                        iresult[i] = (ia[i] + ib[i]) / 2;
                        break;
                }
            }
            break;
        }
        // 其他数据类型的实现...
        default:
            std::memcpy(result, a, count * get_data_type_size(data_type));
            break;
    }
}

// ===== YCCLCommGroup Implementation =====

YCCLCommGroup::YCCLCommGroup(const std::vector<int>& ranks) : group_ranks_(ranks) {
    for (size_t i = 0; i < ranks.size(); ++i) {
        global_to_group_rank_[ranks[i]] = static_cast<int>(i);
    }
}

YCCLCommGroup::~YCCLCommGroup() = default;

std::unique_ptr<YCCLCommGroup> YCCLCommGroup::create_subgroup(
    const YCCLCommunicator& parent_comm, const std::vector<int>& ranks) {
    
    // 验证 ranks 的有效性
    for (int rank : ranks) {
        if (rank < 0 || rank >= parent_comm.get_world_size()) {
            return nullptr;
        }
    }
    
    return std::make_unique<YCCLCommGroup>(ranks);
}

int YCCLCommGroup::get_group_rank(int global_rank) const {
    auto it = global_to_group_rank_.find(global_rank);
    return (it != global_to_group_rank_.end()) ? it->second : -1;
}

bool YCCLCommGroup::all_reduce_group(YCCLCommunicator& comm,
                                    void* send_buffer, void* recv_buffer,
                                    size_t element_count, YCCLDataType data_type,
                                    YCCLReduceOp reduce_op) {
    // 在子组内执行 AllReduce
    // 这里需要实现子组通信的逻辑
    // 简化实现：直接调用全局 AllReduce
    return comm.all_reduce(send_buffer, recv_buffer, element_count, data_type, reduce_op);
}

// ===== YCCL Utils Implementation =====

namespace yccl_utils {

DieMeshTopology create_2d_mesh_topology(int mesh_x, int mesh_y) {
    DieMeshTopology topology;
    topology.mesh_dims = {mesh_x, mesh_y};
    topology.total_dies = mesh_x * mesh_y;
    
    // 构建坐标映射
    for (int y = 0; y < mesh_y; ++y) {
        for (int x = 0; x < mesh_x; ++x) {
            int die_id = y * mesh_x + x;
            std::vector<int> coord = {x, y};
            
            topology.die_coords[die_id] = coord;
            topology.coord_to_die[coord] = die_id;
        }
    }
    
    // 构建邻居关系
    for (int die_id = 0; die_id < topology.total_dies; ++die_id) {
        auto coord = topology.die_coords[die_id];
        int x = coord[0], y = coord[1];
        
        std::vector<int> neighbors;
        
        // 上下左右邻居
        if (x > 0) neighbors.push_back((y * mesh_x) + (x - 1));      // 左
        if (x < mesh_x - 1) neighbors.push_back((y * mesh_x) + (x + 1)); // 右
        if (y > 0) neighbors.push_back(((y - 1) * mesh_x) + x);     // 上
        if (y < mesh_y - 1) neighbors.push_back(((y + 1) * mesh_x) + x); // 下
        
        topology.neighbors[die_id] = neighbors;
    }
    
    return topology;
}

DieMeshTopology create_3d_mesh_topology(int mesh_x, int mesh_y, int mesh_z) {
    DieMeshTopology topology;
    topology.mesh_dims = {mesh_x, mesh_y, mesh_z};
    topology.total_dies = mesh_x * mesh_y * mesh_z;
    
    // 构建 3D 坐标映射
    for (int z = 0; z < mesh_z; ++z) {
        for (int y = 0; y < mesh_y; ++y) {
            for (int x = 0; x < mesh_x; ++x) {
                int die_id = z * mesh_x * mesh_y + y * mesh_x + x;
                std::vector<int> coord = {x, y, z};
                
                topology.die_coords[die_id] = coord;
                topology.coord_to_die[coord] = die_id;
            }
        }
    }
    
    // 构建 3D 邻居关系
    for (int die_id = 0; die_id < topology.total_dies; ++die_id) {
        auto coord = topology.die_coords[die_id];
        int x = coord[0], y = coord[1], z = coord[2];
        
        std::vector<int> neighbors;
        
        // 6 个方向的邻居
        if (x > 0) neighbors.push_back(z * mesh_x * mesh_y + y * mesh_x + (x - 1));
        if (x < mesh_x - 1) neighbors.push_back(z * mesh_x * mesh_y + y * mesh_x + (x + 1));
        if (y > 0) neighbors.push_back(z * mesh_x * mesh_y + (y - 1) * mesh_x + x);
        if (y < mesh_y - 1) neighbors.push_back(z * mesh_x * mesh_y + (y + 1) * mesh_x + x);
        if (z > 0) neighbors.push_back((z - 1) * mesh_x * mesh_y + y * mesh_x + x);
        if (z < mesh_z - 1) neighbors.push_back((z + 1) * mesh_x * mesh_y + y * mesh_x + x);
        
        topology.neighbors[die_id] = neighbors;
    }
    
    return topology;
}

std::string select_optimal_algorithm(YCCLOperation op, int world_size,
                                    size_t message_size, const DieMeshTopology& topology) {
    switch (op) {
        case YCCLOperation::ALL_REDUCE:
            if (message_size < 1024) {
                return "tree_all_reduce";
            } else if (world_size <= 8) {
                return "ring_all_reduce";
            } else {
                return "rabenseifner_all_reduce";
            }
            
        case YCCLOperation::BROADCAST:
            if (world_size <= 16) {
                return "tree_broadcast";
            } else {
                return "pipeline_broadcast";
            }
            
        case YCCLOperation::ALL_GATHER:
            return "ring_all_gather";
            
        case YCCLOperation::ALL_TO_ALL:
            if (world_size <= 8) {
                return "direct_all_to_all";
            } else {
                return "butterfly_all_to_all";
            }
            
        default:
            return "default_algorithm";
    }
}

double estimate_communication_time(YCCLOperation op, size_t message_size,
                                  int world_size, const DieMeshTopology& topology) {
    // 简化的性能模型
    double latency = 0.001;  // 1ms 基础延迟
    double bandwidth = 10.0 * 1024 * 1024 * 1024;  // 10 GB/s 带宽
    
    double transfer_time = static_cast<double>(message_size) / bandwidth;
    
    switch (op) {
        case YCCLOperation::ALL_REDUCE:
            return latency * std::log2(world_size) + transfer_time * 2 * (world_size - 1) / world_size;
            
        case YCCLOperation::BROADCAST:
            return latency * std::log2(world_size) + transfer_time;
            
        case YCCLOperation::ALL_GATHER:
            return latency * (world_size - 1) + transfer_time * (world_size - 1);
            
        case YCCLOperation::ALL_TO_ALL:
            return latency * world_size + transfer_time * world_size;
            
        default:
            return latency + transfer_time;
    }
}

void print_topology_info(const DieMeshTopology& topology) {
    std::cout << "=== YICA Die Mesh Topology ===" << std::endl;
    std::cout << "Dimensions: ";
    for (size_t i = 0; i < topology.mesh_dims.size(); ++i) {
        std::cout << topology.mesh_dims[i];
        if (i < topology.mesh_dims.size() - 1) std::cout << "x";
    }
    std::cout << std::endl;
    std::cout << "Total Dies: " << topology.total_dies << std::endl;
    
    std::cout << "Die Coordinates:" << std::endl;
    for (const auto& [die_id, coord] : topology.die_coords) {
        std::cout << "  Die " << die_id << ": (";
        for (size_t i = 0; i < coord.size(); ++i) {
            std::cout << coord[i];
            if (i < coord.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    std::cout << "Neighbor Relationships:" << std::endl;
    for (const auto& [die_id, neighbors] : topology.neighbors) {
        std::cout << "  Die " << die_id << " -> [";
        for (size_t i = 0; i < neighbors.size(); ++i) {
            std::cout << neighbors[i];
            if (i < neighbors.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void print_communication_stats(const YCCLCommunicator::CommunicationStats& stats) {
    std::cout << "=== YCCL Communication Statistics ===" << std::endl;
    std::cout << "Total Operations: " << stats.total_operations << std::endl;
    std::cout << "Total Bytes Sent: " << stats.total_bytes_sent / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Bytes Received: " << stats.total_bytes_received / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Communication Time: " << stats.total_communication_time << " seconds" << std::endl;
    std::cout << "Average Bandwidth: " << stats.average_bandwidth << " MB/s" << std::endl;
    
    std::cout << "Operation Breakdown:" << std::endl;
    for (const auto& [op, count] : stats.operation_counts) {
        std::cout << "  Operation " << static_cast<int>(op) << ": " << count << " times" << std::endl;
    }
}

} // namespace yccl_utils

} // namespace yica  
} // namespace yirage 