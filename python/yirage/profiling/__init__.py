"""
YICA Profiling Module

Performance profiling, monitoring, and tuning tools.
"""

# Import with graceful fallbacks
try:
    from .profiler import *
except ImportError:
    pass

try:
    from .triton_profiler import *
except ImportError:
    # Triton profiler not available, create dummy classes
    class TritonProfiler:
        def __init__(self, *args, **kwargs):
            print("⚠️ Triton not available, using dummy profiler")
    
    def profile_and_select_best_graph(*args, **kwargs):
        print("⚠️ Triton profiling not available")
        return None

try:
    from .yica_performance_monitor import *
except ImportError:
    pass

try:
    from .yica_performance_tuner import *
except ImportError:
    pass

__all__ = [
    # Basic Profiler
    "export_to_perfetto_trace",
    "EventType",
    
    # Triton Profiler
    "TritonProfiler", 
    "profile_and_select_best_graph",
    
    # Performance Monitor
    "YICAPerformanceMonitor",
    "MetricCollector",
    "AnomalyDetector",
    "PerformanceAnalyzer",
    "RealTimeVisualizer",
    "PerformanceMetric",
    "Alert",
    "MetricType",
    "AlertLevel",
    
    # Performance Tuner
    "YICAAutoTuner",
    "YICAConfig",
    "PerformanceMetrics",
    "YICAPerformanceEvaluator",
    "AutoTuner",
    "RandomSearchTuner",
]
