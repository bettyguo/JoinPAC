"""Hardware and OS information capture."""

from __future__ import annotations

import platform
from typing import Any

import psutil


def capture_system_info() -> dict[str, Any]:
    """Capture comprehensive system information."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["cuda_available"] = False
    return info
