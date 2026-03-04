"""Query execution and cardinality collection."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class QueryExecutor:
    """Executes queries and collects cardinality information."""

    def __init__(self, connection_string: str | None = None) -> None:
        self.connection_string = connection_string
        self._connection: Any = None

    def execute_query(self, sql: str) -> dict[str, Any]:
        """Execute a query and return results with timing."""
        start = time.perf_counter()
        try:
            if self._connection is not None:
                cursor = self._connection.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                elapsed = time.perf_counter() - start
                return {
                    "cardinality": len(rows),
                    "execution_time_ms": elapsed * 1000,
                    "success": True,
                }
        except Exception as e:
            logger.error("Query execution failed: %s", e)

        elapsed = time.perf_counter() - start
        return {
            "cardinality": 0,
            "execution_time_ms": elapsed * 1000,
            "success": False,
        }

    def get_true_cardinality(self, sql: str) -> int:
        """Get the true cardinality of a query result."""
        result = self.execute_query(f"SELECT COUNT(*) FROM ({sql}) AS subq")
        return result.get("cardinality", 0)
