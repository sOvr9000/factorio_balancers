"""
Factorio Belt Balancer Analysis and Generation Library

This module provides tools to work with belt balancers in the game Factorio.
It allows for representing belt balancers as directed graphs, analyzing their flow
properties, creating and manipulating balancer layouts, and converting between
Factorio blueprint strings and graph representations.
"""

from .beltgraph import Evaluation, BeltGraph
from .util import (
    list_vertex_pairs,
    compact_string_to_edges_string,
    edges_string_to_compact_string,
    derive_graph,
    try_derive_graph,
    get_balancer,
    is_balancer_defined,
    common_balancers,
)
from .custom_types import (
    Scalar,
    FlowDict,
    LinearCombination,
)

__version__ = '1.0.0'
__all__ = [
    'Evaluation',
    'BeltGraph',
    'list_vertex_pairs',
    'compact_string_to_edges_string',
    'edges_string_to_compact_string',
    'derive_graph',
    'try_derive_graph',
    'get_balancer',
    'is_balancer_defined',
    'common_balancers',
]
