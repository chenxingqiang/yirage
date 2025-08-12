""" 
YICA-Yirage Global Configuration

全局配置管理，包括YICA设备、GPU设备、编译选项等
"""

import os
from typing import Dict, List, Optional, Any
from enum import Enum


class YICADeviceType(Enum):
    """YICA设备类型"""
    YICA_G100 = "yica_g100"      # YICA G100 存算一体芯片
    YICA_G200 = "yica_g200"      # YICA G200 升级版
    YICA_CLUSTER = "yica_cluster" # YICA集群
    YICA_SIM = "yica_simulator"   # YICA模拟器


class BackendType(Enum):
    """后端类型"""
    YICA = "yica"           # YICA原生后端
    CUDA = "cuda"           # CUDA后端
    CPU = "cpu"             # CPU后端
    TRITON = "triton"       # Triton后端
    AUTO = "auto"           # 自动选择


class OptimizationLevel(Enum):
    """优化级别"""
    O0 = "O0"  # 无优化
    O1 = "O1"  # 基础优化
    O2 = "O2"  # 标准优化
    O3 = "O3"  # 激进优化


class GlobalConfig:
    def __init__(self):
        # ============================================================================
        # 基础配置
        # ============================================================================
        self.verbose = os.getenv("YIRAGE_VERBOSE", "false").lower() == "true"
        self.debug = os.getenv("YIRAGE_DEBUG", "false").lower() == "true"
        self.bypass_compile_errors = os.getenv("YIRAGE_BYPASS_ERRORS", "false").lower() == "true"
        
        # ============================================================================
        # YICA设备配置
        # ============================================================================
        # YICA设备类型
        self.yica_device_type = YICADeviceType(
            os.getenv("YICA_DEVICE_TYPE", "yica_g100")
        )
        
        # YICA设备ID (支持多设备)
        self.yica_device_id = int(os.getenv("YICA_DEVICE_ID", "0"))
        self.yica_device_ids = self._parse_device_ids(
            os.getenv("YICA_DEVICE_IDS", "0")
        )
        
        # YICA集群配置
        self.yica_cluster_size = int(os.getenv("YICA_CLUSTER_SIZE", "1"))
        self.yica_node_id = int(os.getenv("YICA_NODE_ID", "0"))
        self.num_devices = int(os.getenv("YICA_NUM_DEVICES", "1"))  # YICA设备数量
        
        # YICA硬件特性
        self.yica_cim_arrays = int(os.getenv("YICA_CIM_ARRAYS", "64"))      # CIM阵列数量
        self.yica_spm_size_mb = int(os.getenv("YICA_SPM_SIZE_MB", "256"))   # SPM大小(MB)
        self.yica_bandwidth_gbps = float(os.getenv("YICA_BANDWIDTH_GBPS", "400.0"))  # 带宽
        self.yica_compute_capability = os.getenv("YICA_COMPUTE_CAPABILITY", "2.0")   # 计算能力版本
        
        # ============================================================================
        # GPU设备配置 (后备选项)
        # ============================================================================
        self.gpu_device_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        self.gpu_memory_limit_gb = float(os.getenv("GPU_MEMORY_LIMIT_GB", "0"))  # 0表示不限制
        
        # ============================================================================
        # 后端和优化配置
        # ============================================================================
        self.default_backend = BackendType(
            os.getenv("YIRAGE_BACKEND", "auto")
        )
        self.optimization_level = OptimizationLevel(
            os.getenv("YIRAGE_OPT_LEVEL", "O2")
        )
        
        # 编译配置
        self.enable_jit_compilation = os.getenv("YIRAGE_JIT", "true").lower() == "true"
        self.compilation_cache_dir = os.getenv("YIRAGE_CACHE_DIR", 
                                             os.path.expanduser("~/.yirage/cache"))
        self.max_compilation_threads = int(os.getenv("YIRAGE_COMPILE_THREADS", 
                                                   str(os.cpu_count())))
        
        # ============================================================================
        # 性能和调试配置
        # ============================================================================
        # 性能监控
        self.enable_profiling = os.getenv("YIRAGE_PROFILING", "false").lower() == "true"
        self.profiling_output_dir = os.getenv("YIRAGE_PROFILE_DIR", "./yirage_profiles")
        
        # 内存管理
        self.enable_memory_pool = os.getenv("YIRAGE_MEMORY_POOL", "true").lower() == "true"
        self.memory_pool_size_mb = int(os.getenv("YIRAGE_MEMORY_POOL_SIZE_MB", "1024"))
        
        # 并行配置
        self.max_parallel_graphs = int(os.getenv("YIRAGE_MAX_PARALLEL_GRAPHS", "4"))
        self.enable_graph_fusion = os.getenv("YIRAGE_GRAPH_FUSION", "true").lower() == "true"
        
        # ============================================================================
        # 分布式训练配置
        # ============================================================================
        self.distributed_backend = os.getenv("YIRAGE_DIST_BACKEND", "yccl")  # yccl, nccl, gloo
        self.distributed_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.distributed_rank = int(os.getenv("RANK", "0"))
        self.distributed_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
        # ============================================================================
        # 实验性功能
        # ============================================================================
        self.enable_experimental_features = os.getenv("YIRAGE_EXPERIMENTAL", "false").lower() == "true"
        self.enable_auto_tuning = os.getenv("YIRAGE_AUTO_TUNE", "true").lower() == "true"
        self.enable_mixed_precision = os.getenv("YIRAGE_MIXED_PRECISION", "true").lower() == "true"
        
        # 创建必要的目录
        self._ensure_directories()
    
    def _parse_device_ids(self, device_ids_str: str) -> List[int]:
        """解析设备ID字符串"""
        try:
            return [int(x.strip()) for x in device_ids_str.split(",")]
        except:
            return [0]
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.compilation_cache_dir,
            self.profiling_output_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_yica_device_info(self) -> Dict[str, Any]:
        """获取YICA设备信息"""
        return {
            "device_type": self.yica_device_type.value,
            "device_id": self.yica_device_id,
            "device_ids": self.yica_device_ids,
            "cluster_size": self.yica_cluster_size,
            "node_id": self.yica_node_id,
            "cim_arrays": self.yica_cim_arrays,
            "spm_size_mb": self.yica_spm_size_mb,
            "bandwidth_gbps": self.yica_bandwidth_gbps,
            "compute_capability": self.yica_compute_capability,
        }
    
    def set_yica_device(self, device_type: str, device_id: int = 0):
        """设置YICA设备"""
        self.yica_device_type = YICADeviceType(device_type)
        self.yica_device_id = device_id
        if self.verbose:
            print(f"YICA device set to {device_type}:{device_id}")
    
    def set_backend(self, backend: str):
        """设置默认后端"""
        self.default_backend = BackendType(backend)
        if self.verbose:
            print(f"Default backend set to {backend}")
    
    def set_optimization_level(self, level: str):
        """设置优化级别"""
        self.optimization_level = OptimizationLevel(level)
        if self.verbose:
            print(f"Optimization level set to {level}")
    
    def enable_yica_cluster(self, cluster_size: int, node_id: int = 0):
        """启用YICA集群模式"""
        self.yica_cluster_size = cluster_size
        self.yica_node_id = node_id
        self.yica_device_type = YICADeviceType.YICA_CLUSTER
        if self.verbose:
            print(f"YICA cluster enabled: {cluster_size} nodes, current node: {node_id}")
    
    def get_effective_backend(self) -> BackendType:
        """获取有效的后端类型"""
        if self.default_backend == BackendType.AUTO:
            # 自动选择后端
            if self.yica_device_type in [YICADeviceType.YICA_G100, YICADeviceType.YICA_G200]:
                return BackendType.YICA
            elif self.yica_device_type == YICADeviceType.YICA_CLUSTER:
                return BackendType.YICA
            elif self.yica_device_type == YICADeviceType.YICA_SIM:
                return BackendType.CUDA  # 模拟器使用CUDA后端
            else:
                return BackendType.CUDA
        else:
            return self.default_backend
    
    def is_yica_available(self) -> bool:
        """检查YICA设备是否可用"""
        # 这里应该实际检查YICA设备状态
        # 目前返回基于配置的判断
        return self.yica_device_type != YICADeviceType.YICA_SIM
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "version": "1.0.6",
            "yica_device": self.get_yica_device_info(),
            "backend": self.get_effective_backend().value,
            "optimization_level": self.optimization_level.value,
            "distributed": {
                "backend": self.distributed_backend,
                "world_size": self.distributed_world_size,
                "rank": self.distributed_rank,
            },
            "performance": {
                "profiling_enabled": self.enable_profiling,
                "memory_pool_enabled": self.enable_memory_pool,
                "jit_compilation": self.enable_jit_compilation,
            },
            "experimental": {
                "auto_tuning": self.enable_auto_tuning,
                "mixed_precision": self.enable_mixed_precision,
                "graph_fusion": self.enable_graph_fusion,
            }
        }
    
    def print_config(self):
        """打印当前配置"""
        config = self.get_config_summary()
        print("=" * 60)
        print("YICA-Yirage Configuration")
        print("=" * 60)
        
        print(f"Version: {config['version']}")
        print(f"Backend: {config['backend']} (optimization: {config['optimization_level']})")
        
        print("\nYICA Device:")
        yica_info = config['yica_device']
        print(f"  Type: {yica_info['device_type']}")
        print(f"  Device ID: {yica_info['device_id']}")
        print(f"  CIM Arrays: {yica_info['cim_arrays']}")
        print(f"  SPM Size: {yica_info['spm_size_mb']} MB")
        print(f"  Bandwidth: {yica_info['bandwidth_gbps']} Gbps")
        
        if config['distributed']['world_size'] > 1:
            print(f"\nDistributed Training:")
            print(f"  Backend: {config['distributed']['backend']}")
            print(f"  World Size: {config['distributed']['world_size']}")
            print(f"  Rank: {config['distributed']['rank']}")
        
        print(f"\nPerformance:")
        perf = config['performance']
        print(f"  Profiling: {'✓' if perf['profiling_enabled'] else '✗'}")
        print(f"  Memory Pool: {'✓' if perf['memory_pool_enabled'] else '✗'}")
        print(f"  JIT Compilation: {'✓' if perf['jit_compilation'] else '✗'}")
        
        print(f"\nExperimental Features:")
        exp = config['experimental']
        print(f"  Auto Tuning: {'✓' if exp['auto_tuning'] else '✗'}")
        print(f"  Mixed Precision: {'✓' if exp['mixed_precision'] else '✗'}")
        print(f"  Graph Fusion: {'✓' if exp['graph_fusion'] else '✗'}")
        
        print("=" * 60)


# 全局配置实例
global_config = GlobalConfig()


# 便捷函数
def set_yica_device(device_type: str, device_id: int = 0):
    """设置YICA设备的便捷函数"""
    global_config.set_yica_device(device_type, device_id)


def set_backend(backend: str):
    """设置后端的便捷函数"""
    global_config.set_backend(backend)


def enable_verbose():
    """启用详细输出"""
    global_config.verbose = True


def enable_debug():
    """启用调试模式"""
    global_config.debug = True
    global_config.verbose = True


def print_config():
    """打印配置的便捷函数"""
    global_config.print_config()
