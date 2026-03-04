"""System profiling utilities."""

from __future__ import annotations

import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class SystemProfiler:
    """Profiles system resources during experiments."""

    def __init__(self) -> None:
        self._snapshots: list[dict[str, Any]] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Start profiling."""
        self._start_time = time.time()
        self._snapshots = []

    def snapshot(self) -> dict[str, Any]:
        """Take a system resource snapshot."""
        snap = {
            "timestamp": time.time() - (self._start_time or time.time()),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        }
        self._snapshots.append(snap)
        return snap

    def get_summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        if not self._snapshots:
            return {}
        cpu_values = [s["cpu_percent"] for s in self._snapshots]
        mem_values = [s["memory_used_gb"] for s in self._snapshots]
        return {
            "num_snapshots": len(self._snapshots),
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_gb": sum(mem_values) / len(mem_values),
            "max_memory_gb": max(mem_values),
        }
