"""
YICA Hardware Specialization Module

YICA-specific implementations and optimizations for compute-in-memory architecture.
"""

# 使用具体导入避免模块执行冲突
try:
    from .yica_backend_integration import (
        YICABackendIntegration, get_yica_backend, yica_matmul,
        yica_allreduce, yica_rmsnorm
    )
except ImportError:
    pass

# yica_real_optimizer 是可执行模块，不在包级别导入以避免执行冲突
# 需要时可以通过 from yirage.yica.yica_real_optimizer import ... 显式导入

try:
    from .yica_pytorch_backend import (
        YICABackend, is_available, device_count, optimize_model
    )
except ImportError:
    pass

try:
    from .yica_advanced import (
        YICAAnalyzer, YICAMemoryManager, YICAPerformanceMonitor,
        create_yica_system
    )
except ImportError:
    pass

try:
    from .yica_distributed import (
        YCCLCommunicator, YICADistributedOptimizer,
        YICADistributedTrainer, yica_distributed_context
    )
except ImportError:
    pass

__all__ = [
    # Backend Integration
    "YICABackendIntegration",
    "get_yica_backend",
    "yica_matmul",
    "yica_allreduce",
    "yica_rmsnorm",

    # PyTorch Backend
    "YICABackend",
    "is_available",
    "device_count",
    "optimize_model",

    # Advanced Features
    "YICAAnalyzer",
    "YICAMemoryManager",
    "YICAPerformanceMonitor",
    "create_yica_system",

    # Distributed Training
    "YCCLCommunicator",
    "YICADistributedOptimizer",
    "YICADistributedTrainer",
    "yica_distributed_context",
]

# 注意：yica_real_optimizer 作为可执行模块，通过 python -m yirage.yica.yica_real_optimizer 使用
