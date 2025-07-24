#!/usr/bin/env python3
"""
YICA-Yirage 高级性能监控模块

提供实时的 YICA 性能监控和分析，包括：
- 实时性能指标收集
- 异常检测和告警
- 性能瓶颈分析
- 自动化优化建议
- 性能可视化和报告
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import queue
from enum import Enum
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ENERGY = "energy"
    TEMPERATURE = "temperature"
    UTILIZATION = "utilization"
    CACHE_HIT_RATE = "cache_hit_rate"
    BANDWIDTH = "bandwidth"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    type: MetricType
    value: float
    unit: str
    timestamp: datetime
    source: str = "yica_device"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceThreshold:
    """性能阈值"""
    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison_type: str = "greater_than"  # greater_than, less_than, equal


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.collector_thread = None
        self.metrics_queue = queue.Queue()
        self.device_simulators = {}
    
    def start_collection(self):
        """开始指标收集"""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collect_metrics)
        self.collector_thread.daemon = True
        self.collector_thread.start()
        
        logger.info("性能指标收集已开始")
    
    def stop_collection(self):
        """停止指标收集"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join()
        
        logger.info("性能指标收集已停止")
    
    def _collect_metrics(self):
        """收集指标的主循环"""
        while self.running:
            try:
                # 收集各种性能指标
                metrics = self._collect_yica_metrics()
                
                for metric in metrics:
                    self.metrics_queue.put(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"指标收集出错: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_yica_metrics(self) -> List[PerformanceMetric]:
        """收集 YICA 设备指标"""
        metrics = []
        current_time = datetime.now()
        
        # 模拟 CIM 阵列利用率
        cim_utilization = np.random.normal(75, 10) if NUMPY_AVAILABLE else 75.0
        cim_utilization = max(0, min(100, cim_utilization))
        
        metrics.append(PerformanceMetric(
            name="cim_utilization",
            type=MetricType.UTILIZATION,
            value=cim_utilization,
            unit="percent",
            timestamp=current_time,
            metadata={"array_count": 16, "array_size": 64}
        ))
        
        # 模拟 SPM 缓存命中率
        spm_hit_rate = np.random.normal(85, 5) if NUMPY_AVAILABLE else 85.0
        spm_hit_rate = max(0, min(100, spm_hit_rate))
        
        metrics.append(PerformanceMetric(
            name="spm_cache_hit_rate",
            type=MetricType.CACHE_HIT_RATE,
            value=spm_hit_rate,
            unit="percent",
            timestamp=current_time,
            metadata={"spm_size_mb": 64}
        ))
        
        # 模拟内存使用率
        memory_usage = np.random.normal(60, 15) if NUMPY_AVAILABLE else 60.0
        memory_usage = max(0, min(100, memory_usage))
        
        metrics.append(PerformanceMetric(
            name="memory_usage",
            type=MetricType.MEMORY,
            value=memory_usage,
            unit="percent",
            timestamp=current_time
        ))
        
        # 模拟设备温度
        temperature = np.random.normal(45, 5) if NUMPY_AVAILABLE else 45.0
        temperature = max(20, min(85, temperature))
        
        metrics.append(PerformanceMetric(
            name="device_temperature",
            type=MetricType.TEMPERATURE,
            value=temperature,
            unit="celsius",
            timestamp=current_time
        ))
        
        # 模拟延迟
        latency = np.random.lognormal(2.0, 0.5) if NUMPY_AVAILABLE else 5.0
        
        metrics.append(PerformanceMetric(
            name="inference_latency",
            type=MetricType.LATENCY,
            value=latency,
            unit="milliseconds",
            timestamp=current_time
        ))
        
        # 模拟吞吐量
        throughput = np.random.normal(1000, 100) if NUMPY_AVAILABLE else 1000.0
        throughput = max(0, throughput)
        
        metrics.append(PerformanceMetric(
            name="inference_throughput",
            type=MetricType.THROUGHPUT,
            value=throughput,
            unit="ops_per_second",
            timestamp=current_time
        ))
        
        # 模拟能耗
        power_consumption = np.random.normal(150, 20) if NUMPY_AVAILABLE else 150.0
        power_consumption = max(50, power_consumption)
        
        metrics.append(PerformanceMetric(
            name="power_consumption",
            type=MetricType.ENERGY,
            value=power_consumption,
            unit="watts",
            timestamp=current_time
        ))
        
        return metrics
    
    def get_latest_metrics(self, count: int = 10) -> List[PerformanceMetric]:
        """获取最新的指标"""
        metrics = []
        for _ in range(min(count, self.metrics_queue.qsize())):
            try:
                metric = self.metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break
        
        return metrics


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.thresholds = {}
        self.anomaly_callbacks = []
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """添加性能阈值"""
        self.thresholds[threshold.metric_name] = threshold
    
    def add_anomaly_callback(self, callback: Callable[[Alert], None]):
        """添加异常回调"""
        self.anomaly_callbacks.append(callback)
    
    def check_metric(self, metric: PerformanceMetric) -> Optional[Alert]:
        """检查指标是否异常"""
        self.metric_history[metric.name].append(metric.value)
        
        # 检查阈值
        if metric.name in self.thresholds:
            alert = self._check_threshold(metric)
            if alert:
                self._trigger_alert(alert)
                return alert
        
        # 检查统计异常
        if len(self.metric_history[metric.name]) >= self.window_size:
            alert = self._check_statistical_anomaly(metric)
            if alert:
                self._trigger_alert(alert)
                return alert
        
        return None
    
    def _check_threshold(self, metric: PerformanceMetric) -> Optional[Alert]:
        """检查阈值异常"""
        threshold = self.thresholds[metric.name]
        
        if threshold.comparison_type == "greater_than":
            if metric.value >= threshold.critical_threshold:
                level = AlertLevel.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value >= threshold.error_threshold:
                level = AlertLevel.ERROR
                threshold_value = threshold.error_threshold
            elif metric.value >= threshold.warning_threshold:
                level = AlertLevel.WARNING
                threshold_value = threshold.warning_threshold
            else:
                return None
        else:  # less_than
            if metric.value <= threshold.critical_threshold:
                level = AlertLevel.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value <= threshold.error_threshold:
                level = AlertLevel.ERROR
                threshold_value = threshold.error_threshold
            elif metric.value <= threshold.warning_threshold:
                level = AlertLevel.WARNING
                threshold_value = threshold.warning_threshold
            else:
                return None
        
        alert_id = f"{metric.name}_{int(time.time())}"
        message = (f"{metric.name} 值 {metric.value:.2f} {metric.unit} "
                  f"超过 {level.value} 阈值 {threshold_value:.2f}")
        
        return Alert(
            alert_id=alert_id,
            level=level,
            metric_name=metric.name,
            current_value=metric.value,
            threshold=threshold_value,
            message=message,
            timestamp=metric.timestamp
        )
    
    def _check_statistical_anomaly(self, metric: PerformanceMetric) -> Optional[Alert]:
        """检查统计异常"""
        if not NUMPY_AVAILABLE:
            return None
        
        values = list(self.metric_history[metric.name])
        if len(values) < 10:
            return None
        
        mean = np.mean(values[:-1])
        std = np.std(values[:-1])
        
        # 使用 3-sigma 规则
        if abs(metric.value - mean) > 3 * std:
            alert_id = f"{metric.name}_anomaly_{int(time.time())}"
            message = (f"{metric.name} 值 {metric.value:.2f} {metric.unit} "
                      f"异常偏离历史均值 {mean:.2f} (±{3*std:.2f})")
            
            return Alert(
                alert_id=alert_id,
                level=AlertLevel.WARNING,
                metric_name=metric.name,
                current_value=metric.value,
                threshold=mean + 3 * std,
                message=message,
                timestamp=metric.timestamp
            )
        
        return None
    
    def _trigger_alert(self, alert: Alert):
        """触发告警"""
        logger.warning(f"性能告警: {alert.message}")
        
        for callback in self.anomaly_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """分析性能数据"""
        if not metrics:
            return {}
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "metric_count": len(metrics),
            "metric_summary": self._summarize_metrics(metrics),
            "bottlenecks": self._identify_bottlenecks(metrics),
            "recommendations": self._generate_recommendations(metrics),
            "efficiency_score": self._calculate_efficiency_score(metrics)
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _summarize_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """汇总指标"""
        summary = {}
        
        # 按类型分组
        by_type = defaultdict(list)
        for metric in metrics:
            by_type[metric.type.value].append(metric.value)
        
        for metric_type, values in by_type.items():
            if NUMPY_AVAILABLE and values:
                summary[metric_type] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            else:
                summary[metric_type] = {
                    "count": len(values),
                    "values": values
                }
        
        return summary
    
    def _identify_bottlenecks(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 查找高延迟
        latency_metrics = [m for m in metrics if m.type == MetricType.LATENCY]
        if latency_metrics:
            high_latency = [m for m in latency_metrics if m.value > 10.0]  # 10ms 阈值
            if high_latency:
                bottlenecks.append({
                    "type": "high_latency",
                    "severity": "warning",
                    "affected_metrics": [m.name for m in high_latency],
                    "description": f"检测到 {len(high_latency)} 个高延迟指标"
                })
        
        # 查找低缓存命中率
        cache_metrics = [m for m in metrics if m.type == MetricType.CACHE_HIT_RATE]
        if cache_metrics:
            low_cache = [m for m in cache_metrics if m.value < 80.0]  # 80% 阈值
            if low_cache:
                bottlenecks.append({
                    "type": "low_cache_hit_rate",
                    "severity": "warning",
                    "affected_metrics": [m.name for m in low_cache],
                    "description": f"检测到 {len(low_cache)} 个低缓存命中率指标"
                })
        
        # 查找高内存使用
        memory_metrics = [m for m in metrics if m.type == MetricType.MEMORY]
        if memory_metrics:
            high_memory = [m for m in memory_metrics if m.value > 90.0]  # 90% 阈值
            if high_memory:
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "severity": "error",
                    "affected_metrics": [m.name for m in high_memory],
                    "description": f"检测到 {len(high_memory)} 个高内存使用指标"
                })
        
        # 查找高温度
        temp_metrics = [m for m in metrics if m.type == MetricType.TEMPERATURE]
        if temp_metrics:
            high_temp = [m for m in temp_metrics if m.value > 70.0]  # 70°C 阈值
            if high_temp:
                bottlenecks.append({
                    "type": "high_temperature",
                    "severity": "critical",
                    "affected_metrics": [m.name for m in high_temp],
                    "description": f"检测到 {len(high_temp)} 个高温度指标"
                })
        
        return bottlenecks
    
    def _generate_recommendations(self, metrics: List[PerformanceMetric]) -> List[Dict[str, str]]:
        """生成优化建议"""
        recommendations = []
        
        # 基于指标值生成建议
        for metric in metrics:
            if metric.type == MetricType.CACHE_HIT_RATE and metric.value < 85.0:
                recommendations.append({
                    "category": "memory_optimization",
                    "priority": "medium",
                    "title": "提升缓存命中率",
                    "description": f"{metric.name} 命中率为 {metric.value:.1f}%，建议优化数据访问模式或增加 SPM 大小"
                })
            
            elif metric.type == MetricType.UTILIZATION and metric.value < 60.0:
                recommendations.append({
                    "category": "resource_utilization",
                    "priority": "medium",
                    "title": "提升资源利用率",
                    "description": f"{metric.name} 利用率为 {metric.value:.1f}%，建议增加批次大小或启用算子融合"
                })
            
            elif metric.type == MetricType.LATENCY and metric.value > 15.0:
                recommendations.append({
                    "category": "performance_optimization",
                    "priority": "high",
                    "title": "降低推理延迟",
                    "description": f"{metric.name} 延迟为 {metric.value:.1f}ms，建议启用管道优化或调整模型精度"
                })
            
            elif metric.type == MetricType.TEMPERATURE and metric.value > 65.0:
                recommendations.append({
                    "category": "thermal_management",
                    "priority": "high",
                    "title": "加强散热管理",
                    "description": f"{metric.name} 温度为 {metric.value:.1f}°C，建议降低计算频率或改善散热条件"
                })
        
        # 去重
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            key = (rec["category"], rec["title"])
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_efficiency_score(self, metrics: List[PerformanceMetric]) -> float:
        """计算效率评分 (0-100)"""
        if not metrics:
            return 0.0
        
        scores = []
        
        # 评估各类指标
        for metric in metrics:
            if metric.type == MetricType.UTILIZATION:
                # 利用率越高越好，但不要超过95%
                score = min(metric.value, 95.0) / 95.0 * 100
                scores.append(score)
            
            elif metric.type == MetricType.CACHE_HIT_RATE:
                # 缓存命中率越高越好
                score = metric.value
                scores.append(score)
            
            elif metric.type == MetricType.LATENCY:
                # 延迟越低越好，以5ms为基准
                score = max(0, 100 - metric.value * 10)
                scores.append(score)
            
            elif metric.type == MetricType.TEMPERATURE:
                # 温度适中最好，40-50°C为最优
                if 40 <= metric.value <= 50:
                    score = 100
                elif metric.value < 40:
                    score = 90 - (40 - metric.value) * 2
                else:
                    score = max(0, 100 - (metric.value - 50) * 3)
                scores.append(score)
        
        if scores:
            return sum(scores) / len(scores)
        else:
            return 50.0  # 默认中等评分


class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.metric_data = defaultdict(lambda: {"x": deque(maxlen=max_points), 
                                              "y": deque(maxlen=max_points)})
        self.fig = None
        self.axes = {}
        self.animation = None
    
    def setup_realtime_plot(self, metric_names: List[str]):
        """设置实时绘图"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib 不可用，跳过实时可视化")
            return
        
        plt.ion()  # 开启交互模式
        
        # 创建子图
        n_metrics = len(metric_names)
        rows = (n_metrics + 1) // 2
        cols = 2 if n_metrics > 1 else 1
        
        self.fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]
        
        for i, metric_name in enumerate(metric_names):
            if i < len(axes):
                self.axes[metric_name] = axes[i]
                self.axes[metric_name].set_title(metric_name)
                self.axes[metric_name].set_xlabel('时间')
                self.axes[metric_name].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def update_plot(self, metrics: List[PerformanceMetric]):
        """更新绘图"""
        if not MATPLOTLIB_AVAILABLE or not self.fig:
            return
        
        current_time = time.time()
        
        for metric in metrics:
            if metric.name in self.axes:
                self.metric_data[metric.name]["x"].append(current_time)
                self.metric_data[metric.name]["y"].append(metric.value)
                
                # 更新图表
                ax = self.axes[metric.name]
                ax.clear()
                
                x_data = list(self.metric_data[metric.name]["x"])
                y_data = list(self.metric_data[metric.name]["y"])
                
                if x_data and y_data:
                    ax.plot(x_data, y_data, 'b-', linewidth=2)
                    ax.set_title(f"{metric.name} ({metric.value:.2f} {metric.unit})")
                    ax.set_xlabel('时间')
                    ax.grid(True, alpha=0.3)
                    
                    # 设置x轴显示最近的时间
                    if len(x_data) > 1:
                        ax.set_xlim(x_data[0], x_data[-1])
        
        plt.pause(0.01)  # 短暂暂停以更新图表


class YICAPerformanceMonitor:
    """YICA 性能监控主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.collector = MetricCollector(
            collection_interval=self.config.get("collection_interval", 1.0)
        )
        self.detector = AnomalyDetector(
            window_size=self.config.get("window_size", 100)
        )
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = RealTimeVisualizer(
            max_points=self.config.get("max_plot_points", 100)
        )
        
        self.alerts = []
        self.monitoring_active = False
        self.analysis_interval = self.config.get("analysis_interval", 10.0)
        self.last_analysis_time = 0
        
        # 设置默认阈值
        self._setup_default_thresholds()
        
        # 设置告警回调
        self.detector.add_anomaly_callback(self._handle_alert)
    
    def _setup_default_thresholds(self):
        """设置默认性能阈值"""
        thresholds = [
            PerformanceThreshold("cim_utilization", 90, 95, 98),
            PerformanceThreshold("memory_usage", 80, 90, 95),
            PerformanceThreshold("device_temperature", 60, 70, 80),
            PerformanceThreshold("inference_latency", 10, 20, 50),
            PerformanceThreshold("spm_cache_hit_rate", 70, 60, 50, "less_than"),
        ]
        
        for threshold in thresholds:
            self.detector.add_threshold(threshold)
    
    def _handle_alert(self, alert: Alert):
        """处理告警"""
        self.alerts.append(alert)
        
        # 保持最近1000个告警
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def start_monitoring(self, enable_visualization: bool = False, 
                        monitored_metrics: List[str] = None):
        """开始性能监控"""
        if self.monitoring_active:
            logger.warning("性能监控已在运行")
            return
        
        logger.info("启动 YICA 性能监控...")
        
        # 启动指标收集
        self.collector.start_collection()
        
        # 设置可视化
        if enable_visualization and monitored_metrics:
            self.visualizer.setup_realtime_plot(monitored_metrics)
        
        self.monitoring_active = True
        
        # 启动监控循环
        self._monitoring_loop(enable_visualization)
    
    def stop_monitoring(self):
        """停止性能监控"""
        if not self.monitoring_active:
            return
        
        logger.info("停止 YICA 性能监控...")
        
        self.monitoring_active = False
        self.collector.stop_collection()
        
        # 关闭可视化
        if MATPLOTLIB_AVAILABLE and self.visualizer.fig:
            plt.close(self.visualizer.fig)
    
    def _monitoring_loop(self, enable_visualization: bool):
        """监控主循环"""
        while self.monitoring_active:
            try:
                # 获取最新指标
                metrics = self.collector.get_latest_metrics()
                
                # 异常检测
                for metric in metrics:
                    self.detector.check_metric(metric)
                
                # 更新可视化
                if enable_visualization and metrics:
                    self.visualizer.update_plot(metrics)
                
                # 定期性能分析
                current_time = time.time()
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    self.analyzer.analyze_performance(metrics)
                    self.last_analysis_time = current_time
                
                time.sleep(0.1)  # 短暂休眠
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(1.0)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前监控状态"""
        recent_metrics = self.collector.get_latest_metrics(50)
        recent_alerts = [a for a in self.alerts if not a.resolved][-10:]
        
        status = {
            "monitoring_active": self.monitoring_active,
            "total_metrics_collected": len(recent_metrics),
            "active_alerts": len(recent_alerts),
            "recent_metrics": [asdict(m) for m in recent_metrics[-5:]],
            "recent_alerts": [asdict(a) for a in recent_alerts],
        }
        
        if recent_metrics:
            latest_analysis = self.analyzer.analyze_performance(recent_metrics)
            status["latest_analysis"] = latest_analysis
        
        return status
    
    def generate_report(self, output_file: str = None) -> str:
        """生成监控报告"""
        if output_file is None:
            output_file = f"yica_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "monitoring_config": self.config,
                "report_version": "1.0"
            },
            "monitoring_summary": self.get_current_status(),
            "analysis_history": self.analyzer.analysis_history[-10:],  # 最近10次分析
            "alert_summary": {
                "total_alerts": len(self.alerts),
                "alerts_by_level": self._summarize_alerts_by_level(),
                "recent_alerts": [asdict(a) for a in self.alerts[-20:]]
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"监控报告已保存: {output_file}")
        return output_file
    
    def _summarize_alerts_by_level(self) -> Dict[str, int]:
        """按级别汇总告警"""
        summary = {level.value: 0 for level in AlertLevel}
        
        for alert in self.alerts:
            summary[alert.level.value] += 1
        
        return summary


def main():
    """演示性能监控功能"""
    print("🔍 YICA 高级性能监控演示")
    
    # 创建性能监控器
    config = {
        "collection_interval": 0.5,
        "analysis_interval": 5.0,
        "window_size": 50
    }
    
    monitor = YICAPerformanceMonitor(config)
    
    try:
        # 启动监控
        monitored_metrics = [
            "cim_utilization", 
            "memory_usage", 
            "device_temperature", 
            "inference_latency"
        ]
        
        print("🚀 启动性能监控...")
        monitor.start_monitoring(
            enable_visualization=False,  # 设置为True可启用实时可视化
            monitored_metrics=monitored_metrics
        )
        
        # 运行监控一段时间
        print("📊 监控运行中... (按 Ctrl+C 停止)")
        
        for i in range(30):  # 运行30秒
            time.sleep(1)
            
            if i % 10 == 0:
                status = monitor.get_current_status()
                print(f"  监控状态: {status['total_metrics_collected']} 个指标, "
                      f"{status['active_alerts']} 个活跃告警")
        
        # 生成报告
        report_file = monitor.generate_report()
        print(f"📋 监控报告: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断监控")
    
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("✅ 性能监控演示完成")


if __name__ == "__main__":
    main() 