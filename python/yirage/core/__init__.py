"""
YICA-Yirage Core Module

Core functionality and interfaces for YICA computing framework.
"""

from .core import *
from .kernel import *
from .threadblock import *
from .global_config import global_config
from .version import __version__

__all__ = [
    # Core interfaces
    "YICACore",
    "get_yica_core",
    "initialize_yica",
    "is_yica_available",
    "get_yica_info",

    # Kernel management
    "KNGraph",
    "Handle",

    # Threadblock operations
    "TBGraph",
    "DTensor",
    "STensor",

    # Configuration
    "global_config",
    "__version__",
]
