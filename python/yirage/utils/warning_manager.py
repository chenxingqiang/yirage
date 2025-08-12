"""
YICA-Yirage Warning Manager

统一管理警告信息，避免重复警告和提供用户友好的错误信息
"""

import os
import warnings
from typing import Set, Dict, Any
from enum import Enum


class WarningLevel(Enum):
    """警告级别"""
    SILENT = 0      # 静默，不显示警告
    ESSENTIAL = 1   # 仅显示关键警告
    NORMAL = 2      # 显示常规警告
    VERBOSE = 3     # 显示所有警告包括调试信息


class YirageWarningManager:
    """YICA-Yirage警告管理器"""
    
    def __init__(self):
        self._shown_warnings: Set[str] = set()
        self._warning_level = self._get_warning_level()
        
        # 依赖包安装建议
        self._dependency_suggestions = {
            "graphviz": "pip install graphviz",
            "tg4perfetto": "pip install tg4perfetto", 
            "torch": "pip install torch",
            "numpy": "pip install numpy",
            "z3-solver": "pip install z3-solver",
        }
        
        # 警告分类
        self._warning_categories = {
            "dependency": WarningLevel.NORMAL,
            "cython": WarningLevel.VERBOSE,
            "import": WarningLevel.VERBOSE,
            "performance": WarningLevel.ESSENTIAL,
            "hardware": WarningLevel.ESSENTIAL,
        }
    
    def _get_warning_level(self) -> WarningLevel:
        """从环境变量获取警告级别"""
        level_str = os.getenv("YIRAGE_WARNING_LEVEL", "normal").lower()
        
        if level_str == "silent":
            return WarningLevel.SILENT
        elif level_str == "essential":
            return WarningLevel.ESSENTIAL
        elif level_str == "verbose":
            return WarningLevel.VERBOSE
        else:
            return WarningLevel.NORMAL
    
    def should_show_warning(self, category: str, warning_id: str) -> bool:
        """判断是否应该显示警告"""
        if self._warning_level == WarningLevel.SILENT:
            return False
        
        # 避免重复警告
        if warning_id in self._shown_warnings:
            return False
        
        # 根据类别和级别判断
        required_level = self._warning_categories.get(category, WarningLevel.NORMAL)
        return self._warning_level.value >= required_level.value
    
    def warn_dependency_missing(self, package_name: str, feature_description: str = None):
        """依赖包缺失警告"""
        warning_id = f"dependency_{package_name}"
        
        if not self.should_show_warning("dependency", warning_id):
            return
        
        self._shown_warnings.add(warning_id)
        
        suggestion = self._dependency_suggestions.get(package_name, f"pip install {package_name}")
        feature_msg = f" for {feature_description}" if feature_description else ""
        
        print(f"📦 Optional dependency '{package_name}' not found{feature_msg}")
        print(f"   Install with: {suggestion}")
    
    def warn_cython_unavailable(self, module_name: str):
        """Cython模块不可用警告"""
        warning_id = f"cython_{module_name}"
        
        if not self.should_show_warning("cython", warning_id):
            return
        
        self._shown_warnings.add(warning_id)
        
        if self._warning_level == WarningLevel.VERBOSE:
            print(f"🔧 Cython module '{module_name}' not available, using Python fallback")
    
    def warn_import_failed(self, module_name: str, fallback_available: bool = True):
        """导入失败警告"""
        warning_id = f"import_{module_name}"
        
        if not self.should_show_warning("import", warning_id):
            return
        
        self._shown_warnings.add(warning_id)
        
        if self._warning_level == WarningLevel.VERBOSE:
            fallback_msg = " (using fallback)" if fallback_available else ""
            print(f"⚠️  Module '{module_name}' not available{fallback_msg}")
    
    def warn_performance_issue(self, message: str, suggestion: str = None):
        """性能问题警告"""
        warning_id = f"performance_{hash(message)}"
        
        if not self.should_show_warning("performance", warning_id):
            return
        
        self._shown_warnings.add(warning_id)
        
        print(f"🚀 Performance: {message}")
        if suggestion:
            print(f"   Suggestion: {suggestion}")
    
    def warn_hardware_issue(self, message: str, suggestion: str = None):
        """硬件问题警告"""
        warning_id = f"hardware_{hash(message)}"
        
        if not self.should_show_warning("hardware", warning_id):
            return
        
        self._shown_warnings.add(warning_id)
        
        print(f"🔌 Hardware: {message}")
        if suggestion:
            print(f"   Suggestion: {suggestion}")
    
    def get_missing_dependencies(self) -> Dict[str, str]:
        """获取缺失的依赖包列表"""
        missing = {}
        
        # 检查常用依赖
        try:
            import torch
        except ImportError:
            missing["torch"] = self._dependency_suggestions["torch"]
        
        try:
            import numpy
        except ImportError:
            missing["numpy"] = self._dependency_suggestions["numpy"]
        
        try:
            import z3
        except ImportError:
            missing["z3-solver"] = self._dependency_suggestions["z3-solver"]
        
        try:
            import graphviz
        except ImportError:
            missing["graphviz"] = self._dependency_suggestions["graphviz"]
        
        try:
            import tg4perfetto
        except ImportError:
            missing["tg4perfetto"] = self._dependency_suggestions["tg4perfetto"]
        
        return missing
    
    def print_dependency_summary(self):
        """打印依赖包摘要"""
        if self._warning_level == WarningLevel.SILENT:
            return
        
        missing = self.get_missing_dependencies()
        
        if missing:
            print("\n📋 Optional Dependencies Status:")
            print("   The following packages are not installed but could enhance functionality:")
            for package, command in missing.items():
                print(f"   • {package:<15} : {command}")
            print("\n   Note: These are optional. YICA-Yirage will work without them.")
        elif self._warning_level == WarningLevel.VERBOSE:
            print("✅ All optional dependencies are available")
    
    def reset_warnings(self):
        """重置警告状态（用于测试）"""
        self._shown_warnings.clear()


# 全局警告管理器实例
warning_manager = YirageWarningManager()


# 便捷函数
def warn_dependency_missing(package_name: str, feature_description: str = None):
    """依赖包缺失警告的便捷函数"""
    warning_manager.warn_dependency_missing(package_name, feature_description)


def warn_cython_unavailable(module_name: str):
    """Cython不可用警告的便捷函数"""
    warning_manager.warn_cython_unavailable(module_name)


def warn_import_failed(module_name: str, fallback_available: bool = True):
    """导入失败警告的便捷函数"""
    warning_manager.warn_import_failed(module_name, fallback_available)


def warn_performance_issue(message: str, suggestion: str = None):
    """性能问题警告的便捷函数"""
    warning_manager.warn_performance_issue(message, suggestion)


def warn_hardware_issue(message: str, suggestion: str = None):
    """硬件问题警告的便捷函数"""
    warning_manager.warn_hardware_issue(message, suggestion)


def print_dependency_summary():
    """打印依赖摘要的便捷函数"""
    warning_manager.print_dependency_summary()


def set_warning_level(level: str):
    """设置警告级别"""
    os.environ["YIRAGE_WARNING_LEVEL"] = level
    # 重新创建管理器实例
    global warning_manager
    warning_manager = YirageWarningManager()


# 在模块导入时显示依赖摘要
if os.getenv("YIRAGE_SHOW_DEPENDENCY_SUMMARY", "false").lower() == "true":
    print_dependency_summary()
