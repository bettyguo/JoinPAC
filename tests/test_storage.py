"""Tests for storage structures."""

import pytest
from pac_ce.storage.structures import JoinEdge, JoinTree, QueryStructure


class TestJoinTree:
    def test_attribute_disjoint(self, sample_join_tree):
        assert sample_join_tree.is_attribute_disjoint()

    def test_non_attribute_disjoint(self):
        tree = JoinTree(
            root="A",
            children={"A": ["B", "C"]},
            edges=[
                JoinEdge("A", "B", ["shared_attr"]),
                JoinEdge("A", "C", ["shared_attr"]),
            ],
        )
        assert not tree.is_attribute_disjoint()

    def test_acyclic(self, sample_join_tree):
        assert sample_join_tree.is_acyclic()

    def test_num_joins(self, sample_join_tree):
        assert sample_join_tree.num_joins == 2

    def test_num_relations(self, sample_join_tree):
        assert sample_join_tree.num_relations == 3


class TestQueryStructure:
    def test_num_tables(self, sample_query_structure):
        assert sample_query_structure.num_tables == 3

    def test_num_joins(self, sample_query_structure):
        assert sample_query_structure.num_joins == 2

    def test_num_predicates(self, sample_query_structure):
        assert sample_query_structure.num_predicates == 2

    def test_encoding_complexity(self, sample_query_structure):
        kappa = sample_query_structure.query_encoding_complexity
        assert kappa == 3 + 2 + 2  # tables + joins + predicates
