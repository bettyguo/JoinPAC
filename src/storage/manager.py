"""Storage manager for benchmark data and intermediate results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages data storage and retrieval for benchmarks."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._cache: dict[str, Any] = {}

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a database table as a DataFrame."""
        if table_name in self._cache:
            return self._cache[table_name]

        csv_path = self.data_dir / f"{table_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self._cache[table_name] = df
            logger.info("Loaded table %s: %d rows", table_name, len(df))
            return df

        raise FileNotFoundError(f"Table data not found: {csv_path}")

    def get_table_sizes(self) -> dict[str, int]:
        """Return sizes of all loaded tables."""
        return {name: len(df) for name, df in self._cache.items()}

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
