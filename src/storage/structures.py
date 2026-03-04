"""Data structures for join graph representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JoinEdge:
    """Represents a join edge between two relations."""
    left_table: str
    right_table: str
    join_attributes: list[str]


@dataclass
class JoinTree:
    """Represents an acyclic join tree."""
    root: str
    children: dict[str, list[str]] = field(default_factory=dict)
    edges: list[JoinEdge] = field(default_factory=list)
    _treewidth: int | None = None

    @property
    def num_joins(self) -> int:
        """Number of join edges."""
        return len(self.edges)

    @property
    def num_relations(self) -> int:
        """Number of relations in the join tree."""
        all_tables = {self.root}
        for edge in self.edges:
            all_tables.add(edge.left_table)
            all_tables.add(edge.right_table)
        return len(all_tables)

    def is_attribute_disjoint(self) -> bool:
        """Check if the join tree satisfies the attribute-disjoint condition."""
        for parent, kids in self.children.items():
            parent_edges = [e for e in self.edges if e.left_table == parent]
            all_attrs: list[set[str]] = []
            for edge in parent_edges:
                attr_set = set(edge.join_attributes)
                for prev in all_attrs:
                    if attr_set & prev:
                        return False
                all_attrs.append(attr_set)
        return True

    def is_acyclic(self) -> bool:
        """Check if the join graph is acyclic."""
        visited: set[str] = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node in visited:
                return False
            visited.add(node)
            stack.extend(self.children.get(node, []))
        return True

    @property
    def depth(self) -> int:
        """Compute tree depth."""
        def _depth(node: str) -> int:
            kids = self.children.get(node, [])
            if not kids:
                return 0
            return 1 + max(_depth(c) for c in kids)
        return _depth(self.root)


@dataclass
class QueryStructure:
    """Represents a parsed query structure."""
    tables: list[str]
    joins: list[JoinEdge]
    predicates: dict[str, list[dict[str, Any]]]
    join_tree: JoinTree | None = None

    @property
    def num_tables(self) -> int:
        return len(self.tables)

    @property
    def num_joins(self) -> int:
        return len(self.joins)

    @property
    def num_predicates(self) -> int:
        return sum(len(preds) for preds in self.predicates.values())

    @property
    def query_encoding_complexity(self) -> int:
        """Compute kappa(Q) = |V| + |E| + sum of predicates per vertex."""
        return self.num_tables + self.num_joins + self.num_predicates
