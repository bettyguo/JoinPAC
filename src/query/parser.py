"""Query parsing and structure extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

from pac_ce.storage.structures import JoinEdge, JoinTree, QueryStructure

logger = logging.getLogger(__name__)


def parse_query(sql: str) -> QueryStructure:
    """Parse a SQL query into a QueryStructure."""
    tables = _extract_tables(sql)
    joins = _extract_joins(sql)
    predicates = _extract_predicates(sql, tables)
    join_tree = _build_join_tree(tables, joins)

    return QueryStructure(
        tables=tables,
        joins=joins,
        predicates=predicates,
        join_tree=join_tree,
    )


def _extract_tables(sql: str) -> list[str]:
    """Extract table names from SQL."""
    pattern = r"FROM\s+(\w+)|JOIN\s+(\w+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    tables = []
    for m in matches:
        table = m[0] or m[1]
        if table and table not in tables:
            tables.append(table)
    return tables


def _extract_joins(sql: str) -> list[JoinEdge]:
    """Extract join conditions from SQL."""
    pattern = r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)"
    matches = re.findall(pattern, sql)
    edges = []
    for left_table, left_attr, right_table, right_attr in matches:
        edges.append(JoinEdge(
            left_table=left_table,
            right_table=right_table,
            join_attributes=[left_attr],
        ))
    return edges


def _extract_predicates(
    sql: str, tables: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Extract selection predicates from SQL."""
    predicates: dict[str, list[dict[str, Any]]] = {t: [] for t in tables}
    pattern = r"(\w+)\.(\w+)\s*(=|<|>|<=|>=|LIKE|BETWEEN)\s*(.+?)(?:\s+AND|\s+OR|\s*$)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    for table, attr, op, value in matches:
        if table in predicates:
            predicates[table].append({
                "attribute": attr,
                "operator": op,
                "value": value.strip().rstrip(")"),
            })
    return predicates


def _build_join_tree(
    tables: list[str], joins: list[JoinEdge]
) -> JoinTree | None:
    """Build a join tree from tables and join edges."""
    if not tables:
        return None

    root = tables[0]
    children: dict[str, list[str]] = {t: [] for t in tables}
    visited = {root}

    for edge in joins:
        if edge.left_table in visited and edge.right_table not in visited:
            children[edge.left_table].append(edge.right_table)
            visited.add(edge.right_table)
        elif edge.right_table in visited and edge.left_table not in visited:
            children[edge.right_table].append(edge.left_table)
            visited.add(edge.left_table)

    return JoinTree(root=root, children=children, edges=joins)
