"""Shared test fixtures for PAC-CE."""

import numpy as np
import pytest

from pac_ce.core.config import PACCEConfig
from pac_ce.storage.structures import JoinEdge, JoinTree, QueryStructure


@pytest.fixture
def default_config():
    """Default test configuration."""
    return PACCEConfig()


@pytest.fixture
def sample_join_tree():
    """Sample attribute-disjoint acyclic join tree."""
    return JoinTree(
        root="title",
        children={"title": ["movie_info", "keyword"]},
        edges=[
            JoinEdge("title", "movie_info", ["movie_id"]),
            JoinEdge("title", "keyword", ["keyword_id"]),
        ],
    )


@pytest.fixture
def sample_query_structure(sample_join_tree):
    """Sample query structure."""
    return QueryStructure(
        tables=["title", "movie_info", "keyword"],
        joins=sample_join_tree.edges,
        predicates={
            "title": [{"attribute": "production_year", "operator": ">", "value": "2000"}],
            "movie_info": [{"attribute": "info_type", "operator": "=", "value": "genres"}],
            "keyword": [],
        },
        join_tree=sample_join_tree,
    )


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.RandomState(42)
