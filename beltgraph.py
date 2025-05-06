"""
Factorio Belt Balancer Analysis and Generation Library

This module provides tools to work with belt balancers in the game Factorio.
It allows for representing belt balancers as directed graphs, analyzing their flow
properties, creating and manipulating balancer layouts, and converting between
Factorio blueprint strings and graph representations.

The main class provided is BeltGraph, which represents a belt balancer as a directed
graph with designated inputs and outputs.

Key features:
- Convert Factorio blueprint strings to graph representations and vice versa
- Analyze balancer throughput and balance properties
- Create, modify and manipulate balancer structures
- Evaluate balancer performance
- Store and retrieve common balancer patterns
"""

import os
from copy import deepcopy
from graph_tools import Graph
from sympy import factorint
from factorio_blueprints import (
    load_blueprint_string_from_file,
    load_blueprint_string,
)
from .metrics import measure_accuracy
from .sparsesolver import calculate_flow_points, pretty_relations, construct_relations
from .util import edges_string_to_compact_string, compact_string_to_edges_string, try_derive_graph, common_balancers
from typing import Optional, Union, List, Dict, Tuple, Iterator

__all__ = [
    'Evaluation', 'BeltGraph'
]



class Evaluation(dict):
    """
    A dictionary-based class for storing evaluation metrics of a BeltGraph.
    
    This class extends dict to include metrics about a belt balancer's performance,
    such as accuracy, error, bottlenecks, and flow characteristics.
    
    Attributes:
        display_flow_points (bool): Flag to control whether flow points are displayed
            in the string representation.
    """
    display_flow_points = False
    
    def __repr__(self):
        """
        Create a string representation of the evaluation results.
        
        Returns:
            str: A formatted string showing accuracy, error, score, bottlenecks,
                 and flow information.
        """
        s = f'Accuracy: {self["accuracy"]:.5f}\nError: {self["error"]:.5f}\nScore: {self["score"]:.5f}\n'
        s += f'Bottlenecks: {self["bottlenecks"]}\n'
        s += f'Output flow:\n{pretty_relations(self["output_flow"])}\n'
        if self.display_flow_points:
            s += f'Flow points:\n{pretty_relations(self["flow_points"])}\n'
        return s



class BeltGraph(Graph):
    """
    A directed graph representation of a Factorio belt balancer.
    
    This class extends the Graph class to represent belt balancers, where vertices
    represent splitters, inputs, or outputs, and edges represent belt connections.
    
    Attributes:
        inputs (list): List of vertex IDs representing input belts
        outputs (list): List of vertex IDs representing output belts
    """
    
    def __init__(self):
        """Initialize a new BeltGraph with empty lists of inputs and outputs."""
        super().__init__(directed=True, multiedged=False)
        self.inputs = []
        self.outputs = []
    
    @property
    def num_inputs(self) -> int:
        """Return the number of input belts in the balancer."""
        return len(self.inputs)
    
    @property
    def num_outputs(self) -> int:
        """Return the number of output belts in the balancer."""
        return len(self.outputs)
    
    @property
    def num_vertices(self) -> int:
        """Return the total number of vertices in the graph."""
        return len(self.vertices())
    
    @property
    def num_internal_edges(self) -> int:
        """
        Return the number of belt connections between any two splitters in the balancer.
        
        This excludes connections to input or output vertices.
        """
        return self.num_edges - self.num_inputs - self.num_outputs
    
    @property
    def num_internal_vertices(self) -> int:
        """
        Return the number of splitters in the balancer.
        
        This excludes input and output vertices.
        """
        return self.num_vertices - self.num_inputs - self.num_outputs
    
    @property
    def num_edges(self) -> int:
        """Return the total number of edges (belt connections) in the graph."""
        return len(self.edges())
    
    @property
    def balancer_type(self) -> str:
        """
        Return the balancer type as a string in the format 'inputs-outputs'.
        
        For example, a balancer with 4 inputs and 4 outputs would be "4-4".
        """
        return f'{self.num_inputs}-{self.num_outputs}'
    
    @property
    def summary(self) -> str:
        """
        Return a summary string with balancer type, vertices and edges count.
        
        Format: "<inputs>-<outputs> [V:<internal_vertices> E:<internal_edges>]"
        """
        return f'{self.balancer_type} [V:{self.num_internal_vertices} E:{max(0,self.num_internal_edges)}]'
    
    @property
    def advanced_summary(self) -> str:
        """
        Return a detailed summary including prime factorization of inputs and outputs.
        
        This is useful for understanding the mathematical properties of the balancer.
        """
        in_factors = factorint(self.num_inputs)
        out_factors = factorint(self.num_outputs)
        in_str = ' * '.join(f'{k}^{v}' for k, v in in_factors.items())
        out_str = ' * '.join(f'{k}^{v}' for k, v in out_factors.items())

        # If the number of inputs or outputs is 1, simply write "1".
        if len(in_factors) == 0:
            in_str = '1'
        if len(out_factors) == 0:
            out_str = '1'

        factor_sum = sum(k*v for k, v in in_factors.items()) + sum(k*v for k, v in out_factors.items())
        s = self.summary
        return f'{s} [{in_str} -> {out_str}] [factor sum: {factor_sum}]'
    
    def internal_edges(self) -> Iterator[Tuple[int, int]]:
        """
        Iterate over edges where neither endpoint is an input or output vertex.
        
        Yields:
            tuple: (u, v) pairs representing internal edges
        """
        for u, v in self.edges():
            if self.is_input(u):
                continue
            if self.is_output(v):
                continue
            yield u, v
    
    def internal_vertices(self) -> Iterator[int]:
        """
        Iterate over vertices that are neither inputs nor outputs.
        
        Yields:
            int: Vertex IDs of internal vertices (splitters)
        """
        for u in self.vertices():
            if not self.is_input(u) and not self.is_output(u):
                yield u
    
    def set_input(self, u: int, flag: bool = True) -> None:
        """
        Define a vertex of the graph to be an input.

        If vertex `u` does not exist yet, then a new vertex is created with that id as an input vertex.
        
        Args:
            u: Vertex ID
            flag: If True, mark as input; if False, unmark as input
        """
        if flag:
            if not self.is_input(u):
                self.inputs.append(u)
                self.add_vertex(u)
        else:
            if self.is_input(u):
                self.inputs.remove(u)
                self.add_vertex(u)
    
    def set_output(self, u: int, flag: bool = True) -> None:
        """
        Define a vertex of the graph to be an output.

        If vertex `u` does not exist yet, then a new vertex is created with that id as an output vertex.
        
        Args:
            u: Vertex ID
            flag: If True, mark as output; if False, unmark as output
        """
        if flag:
            if not self.is_output(u):
                self.outputs.append(u)
                self.add_vertex(u)
        else:
            if self.is_output(u):
                self.outputs.remove(u)
                self.add_vertex(u)
    
    def set_num_inputs(self, new_num_inputs: int) -> None:
        """
        Adjust the number of input vertices to match the specified number.
        
        This either adds new input vertices or removes existing ones.
        
        Args:
            new_num_inputs: The desired number of inputs
        """
        while self.num_inputs > new_num_inputs:
            self.set_input(self.inputs[0], flag=False)
        while self.num_inputs < new_num_inputs:
            self.set_input(self.new_vertex(), flag=True)
    
    def set_num_outputs(self, new_num_outputs: int) -> None:
        """
        Adjust the number of output vertices to match the specified number.
        
        This either adds new output vertices or removes existing ones.
        
        Args:
            new_num_outputs: The desired number of outputs
        """
        while self.num_outputs > new_num_outputs:
            self.set_output(self.outputs[0], flag=False)
        while self.num_outputs < new_num_outputs:
            self.set_output(self.new_vertex(), flag=True)
    
    def get_next_unused_input(self) -> Optional[int]:
        """
        Find an input vertex that has no outbound edges.
        
        Returns:
            int or None: The vertex ID of an unused input, or None if all inputs are used
        """
        for u in self.inputs:
            if self.out_degree(u) == 0:
                return u
        return None
    
    def get_next_unused_output(self) -> Optional[int]:
        """
        Find an output vertex that has no inbound edges.
        
        Returns:
            int or None: The vertex ID of an unused output, or None if all outputs are used
        """
        for u in self.outputs:
            if self.in_degree(u) == 0:
                return u
        return None
    
    def is_input(self, u: int) -> bool:
        """
        Check if a vertex is an input.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex is an input, False otherwise
        """
        return u in self.inputs
    
    def is_output(self, u: int) -> bool:
        """
        Check if a vertex is an output.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex is an output, False otherwise
        """
        return u in self.outputs
    
    def get_outbound_limit(self, u: int) -> int:
        """
        Get the maximum number of outbound edges allowed for a vertex.
        
        Args:
            u: Vertex ID
            
        Returns:
            int: The maximum number of outbound edges (2 for normal vertices, 
                 1 for inputs, 0 for outputs)
        """
        if self.is_input(u):
            return 1
        if self.is_output(u):
            return 0
        return 2
    
    def get_inbound_limit(self, u: int) -> int:
        """
        Get the maximum number of inbound edges allowed for a vertex.
        
        Args:
            u: Vertex ID
            
        Returns:
            int: The maximum number of inbound edges (2 for normal vertices,
                 0 for inputs, 1 for outputs)
        """
        if self.is_input(u):
            return 0
        if self.is_output(u):
            return 1
        return 2
    
    def is_functional(self, u: int) -> bool:
        """
        Return whether vertex `u` is functioning as a balancer, splitter, or merger.
        
        Returns False when the vertex has one or fewer inbound edges and outbound edges,
        which would either cause jams in the flow or doesn't propagate flow.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex is functional, False otherwise
        """
        return self.vertex_degree(u) >= 3
    
    def is_disconnected(self, u: int) -> bool:
        """
        Check if a vertex has no connections at all.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex has no edges, False otherwise
        """
        return self.vertex_degree(u) == 0
    
    def is_partially_disconnected(self, u: int) -> bool:
        """
        Check if a vertex has either no inbound edges, no outbound edges, or both.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex is partially disconnected, False otherwise
        """
        return self.in_degree(u) == 0 or self.out_degree(u) == 0
    
    def is_splitter(self, u: int) -> bool:
        """
        Check if a vertex represents a 1-2 splitter.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex has 1 input and 2 outputs, False otherwise
        """
        return self.out_degree(u) == 2 and self.in_degree(u) == 1
    
    def is_merger(self, u: int) -> bool:
        """
        Check if a vertex represents a 2-1 merger.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex has 2 inputs and 1 output, False otherwise
        """
        return self.out_degree(u) == 1 and self.in_degree(u) == 2
    
    def is_balancer(self, u: int) -> bool:
        """
        Check if a vertex represents a 2-2 balancer.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex has 2 inputs and 2 outputs, False otherwise
        """
        return self.vertex_degree(u) == 4
    
    def is_identity(self, u: int) -> bool:
        """
        Check if a vertex represents a 1-1 passthrough, which has no effect on flow.
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex has 1 input and 1 output, False otherwise
        """
        return self.in_degree(u) == 1 and self.out_degree(u) == 1
    
    def is_empty(self) -> bool:
        """
        Check if the graph has no inputs or outputs.
        
        Returns:
            bool: True if there are no inputs and no outputs, False otherwise
        """
        return self.num_inputs == 0 and self.num_outputs == 0
    
    def is_internal_vertex(self, u: int) -> bool:
        """
        Check if a vertex is an internal vertex (not an input or output).
        
        Args:
            u: Vertex ID to check
            
        Returns:
            bool: True if the vertex is internal, False otherwise
        """
        return not self.is_input(u) and not self.is_output(u)
    
    def copy_graph(self) -> 'BeltGraph':
        """
        Create a deep copy of this graph.
        
        Returns:
            BeltGraph: A new graph with identical structure
        """
        return deepcopy(self)
    
    def can_add_edge_or_combine(self, u: int, v: int) -> bool:
        """
        Check if an edge can be added between two vertices or if they can be combined.
        
        Args:
            u: Source vertex ID
            v: Target vertex ID
            
        Returns:
            bool: True if the edge can be added or vertices combined, False otherwise
        """
        if u == v:
            return False
        return True
    
    def add_edge_or_combine(self, u: int, v: int) -> None:
        """
        Add an edge between vertices or combine them if an edge already exists.
        
        If an edge already exists, the vertices are combined to optimize the structure.
        
        Args:
            u: Source vertex ID
            v: Target vertex ID
            
        Raises:
            Exception: If vertices cannot be connected
        """
        if u == v:
            raise Exception('Cannot connect a vertex to itself.')
        if self.has_edge(u, v):
            self.combine_vertices(u, v)
        else:
            self.add_edge(u, v)
    
    def can_add_edge(self, u: int, v: int) -> bool:
        """
        Check if an edge can be added between two vertices.
        
        Args:
            u: Source vertex ID
            v: Target vertex ID
            
        Returns:
            bool: True if the edge can be added, False otherwise
        """
        if u == v:
            return False
        if self.out_degree(u) >= self.get_outbound_limit(u):
            return False
        if self.in_degree(v) >= self.get_inbound_limit(v):
            return False
        if self.has_edge(u, v):
            return False
        return True
    
    def add_edge(self, u: int, v: int) -> None:
        """
        Add an edge between two vertices if possible.
        
        Args:
            u: Source vertex ID
            v: Target vertex ID
            
        Raises:
            Exception: If the edge cannot be added due to constraints
        """
        if u == v:
            raise Exception('Cannot connect a vertex to itself.')
        if self.out_degree(u) >= self.get_outbound_limit(u):
            raise Exception(f'Vertex {self.vertex_to_str(u)} has too many outbound edges.')
        if self.in_degree(v) >= self.get_inbound_limit(v):
            raise Exception(f'Vertex {self.vertex_to_str(v)} has too many inbound edges.')
        if self.has_edge(u, v):
            raise Exception('Cannot create duplicate edges.')
        super().add_edge(u, v)
    
    def can_insert_vertex(self, u: int, v: int) -> bool:
        """
        Check if a vertex can be inserted between two connected vertices.
        
        Args:
            u: Source vertex ID
            v: Target vertex ID
            
        Returns:
            bool: True if a vertex can be inserted, False otherwise
        """
        return self.has_edge(u, v)
    
    def insert_vertex(self, u: int, v: int) -> None:
        """
        Insert a new vertex between two connected vertices.
        
        Args:
            u: Source vertex ID
            v: Target vertex ID
            
        Raises:
            Exception: If there is no edge between the vertices
        """
        if not self.has_edge(u, v):
            raise Exception(f'There is no edge from vertex {self.vertex_to_str(u)} to vertex {self.vertex_to_str(v)}.')
        nv = self.new_vertex()
        self.delete_edge(u, v)
        self.add_edge(u, nv)
        self.add_edge(nv, v)
    
    def insert_graph(self, graph: 'BeltGraph') -> tuple[tuple, tuple]:
        """
        Insert another graph into this graph as a subgraph.
        
        Args:
            graph: The graph to insert
            
        Returns:
            tuple: ((input_vertex_ids), (output_vertex_ids)) of the inserted subgraph
        """
        if graph.num_internal_vertices == 0:
            return tuple(), tuple()
            
        # Create a mapping from the vertices in the source graph to new vertices in this graph
        nv_map = [None] * (max(graph.vertices()) + 1)
        inputs = []
        outputs = []
        
        # Create new vertices for each internal vertex in the source graph
        for u in graph.vertices():
            if graph.is_input(u) or graph.is_output(u):
                continue
            v = self.new_vertex()
            nv_map[u] = v
            self.add_vertex(v)
        
        # Record the corresponding vertices for inputs and outputs
        for u in graph.inputs:
            inputs.append(nv_map[graph.out_vertices(u)[0]])
        for v in graph.outputs:
            outputs.append(nv_map[graph.in_vertices(v)[0]])
        
        # Create edges corresponding to those in the source graph
        for u, v in graph.edges():
            if graph.is_input(u) or graph.is_output(v):
                continue
            self.add_edge(nv_map[u], nv_map[v])
            
        return tuple(inputs), tuple(outputs)
    
    def insert_graph_between(self, graph: 'BeltGraph', from_vertices: list[int], to_vertices: list[int]) -> tuple[tuple, tuple]:
        """
        Insert a graph between two sets of vertices in this graph.
        
        Args:
            graph: The graph to insert
            from_vertices: Source vertices to connect from
            to_vertices: Target vertices to connect to
            
        Returns:
            tuple: ((input_vertex_ids), (output_vertex_ids)) of the inserted subgraph
            
        Raises:
            Exception: If the number of inputs/outputs doesn't match or connections cannot be made
        """
        if graph.num_inputs != len(from_vertices):
            raise Exception(f'Graph has incorrect number of input vertices. Received {graph.num_inputs}, expected {len(from_vertices)}')
        if graph.num_outputs != len(to_vertices):
            raise Exception(f'Graph has incorrect number of output vertices. Received {graph.num_outputs}, expected {len(to_vertices)}')
            
        # Insert the graph
        input_vertices, output_vertices = self.insert_graph(graph)
        
        # Connect from_vertices to the inputs of the inserted graph,
        # and the outputs of the inserted graph to to_vertices
        for u, v in zip(from_vertices + list(output_vertices), list(input_vertices) + to_vertices):
            if self.can_add_edge(u, v):
                self.add_edge(u, v)
            else:
                raise Exception(f'Failed to connect {self.vertex_to_str(u)} to {self.vertex_to_str(v)}. '
                                f'This is likely due to vertex {self.vertex_to_str(u)} having too many outbound edges. '
                                f'Try deleting edges before inserting the graph between these vertices.')
                
        return input_vertices, output_vertices
    
    def out_vertices(self, u: int) -> Union[tuple, tuple[int], tuple[int, int]]:
        """
        Get the outbound neighbor vertices of a vertex.
        
        Args:
            u: Vertex ID
            
        Returns:
            tuple: Empty tuple, single vertex tuple, or pair of vertices tuple
        """
        out_edges = self.out_edges(u)
        if len(out_edges) == 0 or out_edges[0][1] is None:
            return tuple()
        if len(out_edges) == 1:
            return out_edges[0][1],
        if out_edges[0][1] is None:
            if out_edges[1][1] is None:
                return tuple()
            return out_edges[1][1],
        if out_edges[1][1] is None:
            return out_edges[0][1],
        return out_edges[0][1], out_edges[1][1]
    
    def in_vertices(self, u: int) -> Union[tuple, tuple[int], tuple[int, int]]:
        """
        Get the inbound neighbor vertices of a vertex.
        
        Args:
            u: Vertex ID
            
        Returns:
            tuple: Empty tuple, single vertex tuple, or pair of vertices tuple
        """
        in_edges = self.in_edges(u)
        if len(in_edges) == 0 or in_edges[0][0] is None:
            return tuple()
        if len(in_edges) == 1:
            return in_edges[0][0],
        if in_edges[0][0] is None:
            if in_edges[1][0] is None:
                return tuple()
            return in_edges[1][0],
        if in_edges[1][0] is None:
            return in_edges[0][0],
        return in_edges[0][0], in_edges[1][0]
    
    def doubled(self) -> 'BeltGraph':
        """
        Create a doubled version of this balancer with twice the inputs and outputs.
        
        This replaces each 2-2 balancer with a 4-4 balancer, effectively doubling the capacity.
        
        Returns:
            BeltGraph: A new graph with doubled inputs and outputs
        """
        r = self.rearrange_vertices_by_depth()
        r.simplify()
        
        # Load standard building blocks for the doubled structure
        double_balancer = BeltGraph()
        double_balancer.load_blueprint_string('0eNqllu1ugjAQhl/FzDUYWw6lJLsvYszGQ7NpgoW0xWgM775F3KPMKg43hNJ+8zPTv8wZNlWrGquNh/IMelsbB+XyDE6/m3XVj/lTo6CEg7a+DSMRmPW+HxhmxCl0EWizU0coWRc9uZJ3qwiU8dprNQi4PJzeTLvfKBvQX6vVsbHKubgNK+27rcM93qjKB35TuwCoTR88QHMxzyI4QRkzPs9CsKuguvVN66EX+ycKv4ni7dq4prYei5H/ijHCTCjMZJyZTmeKe8xsOlPe+/b8maoVU6smpisv7mWjoGQDYUpK1dglFztt1XaYkI9EYAvKZmPjstmt91xTae/Dy5HMXmELBMYpicUUJpQdcJNZPhaCYjmJyM4I1ZIPqc5Jx2bx03/aIPZjgvAZBZIaiv/kQ1aRhD2DFJQvCKqRTHBGOkMfKyHnT1hcIIIpPzyMmU7RJwZWjrAoPzpMXz5Fn/xfH8VOGcIsCKcXxqQYCGEmFANdmKGN1F7tA+C7sY2gWofFYWx57UXjoQV9+exEV7NZHK7XGTohQA7KuuGkLVgqJBciETIr0q77AA3PvFE=')
        double_balancer = double_balancer.rearrange_vertices_by_depth()
        
        double_splitter = BeltGraph()
        double_splitter.load_blueprint_string('0eNqlVtuOgyAU/JfzjE0BFfFXNpuNtqQhUSSAmxrjvy/apmm2klp5MhiYmTNnuIxQN73QRioH5Qjy1CkL5dcIVl5U1cz/3KAFlCCdaAGBqtp5JK7aCGsTZypldWdcUovGwYRAqrO4QokntBnE6kY6J8zTcjJ9IxDKSSfFTdAyGH5U39Z+ZonfSUGgO+uXd2rm95AJxsUhQzB4dJIdsmlW+A+V7EFl71BpnFaP7405SyNOtyn5Ckf6wtF7J83FdP67hSVfWO596nqn+7mdLzxZnEMvtZAVjnxfLezTWtieWviDhW+ppQinfbUfd+xiPUk8Lkl8HRUf98DmYSPWAopxXHJC2klUE+kxAEt39Y3iAFzs9lx0PiItVSDROItzIyQ/jwpI0GQWBxtSW0SeH1vNjtuNlAbuoONn0eN3OBKAi9t3QZXkM5XsWaW/25eXQPn0+kDwK4y9naMFThknjFHGsyKdpj/P3Nc5')
        double_splitter = double_splitter.rearrange_vertices_by_depth()
        
        double_merger = BeltGraph()
        double_merger.load_blueprint_string('0eNqllt2OgyAQhd9lrrERRBFfpdls+kMaEosEcFNjfPeFtrtpVtm2cmUw8M3xzJngCPu2F9pI5aAZQR46ZaHZjmDlSe3a8M4NWkAD0okzIFC7c1iJizbC2syZnbK6My7bi9bBhECqo7hAgyf0MsTqVjonzMNxMn0gEMpJJ8VN0HUxfKr+vPc7G/xMCgLdWX+8U6G+R2aY8E2JYPD0gmzKKSj8QyVrqPUzahH/4kWVd1qxTKMrNBb5r8ZymVq+pbHI7zS6TKtmtN731ZxM55+vOBlUop/UdL3TfQjXrA57z9n6f9X1OtX8XdV8TcrYsw7iPC0YVQSL02aiunpylEYcblvIUhGSNs6zItVSkZVzyCLG0LQ2xvwuU2eHP6ZQqkgIcZXW1zoin6V1MoZNns0XXeFJrlAcuVfyJFeiWLwq0TQPOH/BXq/j5uEXAMGXMPY2pjWmjBPGCsbLmk7TN4FgrxE=')
        double_merger = double_merger.rearrange_vertices_by_depth()
        
        new_graph = BeltGraph()
        new_graph.set_num_inputs(r.num_inputs * 2)
        new_graph.set_num_outputs(r.num_outputs * 2)

        sub_graphs = {}
        def get_sub_graph(u: int):
            if u not in sub_graphs:
                if r.is_balancer(u):
                    the_graph = double_balancer
                elif r.is_merger(u):
                    the_graph = double_merger
                elif r.is_splitter(u):
                    the_graph = double_splitter
                elif r.is_partially_disconnected(u):
                    return None
                else:
                    raise Exception(f'Cannot get a subgraph for vertex {r.vertex_to_str(u)}\n{r.balancer_type}\n{r.in_degree(u)} {r.out_degree(u)}')
                sub_graphs[u] = new_graph.insert_graph(the_graph)
            return sub_graphs[u]
        
        def connectable_subgraph_outputs(sg: tuple[tuple, tuple]):
            for out_v in sg[1]:
                if new_graph.out_degree(out_v) < new_graph.get_outbound_limit(out_v):
                    yield out_v

        def connectable_subgraph_inputs(sg: tuple[tuple, tuple]):
            for in_v in sg[0]:
                if new_graph.in_degree(in_v) < new_graph.get_inbound_limit(in_v):
                    yield in_v
        
        combined_vertices = []

        def connect_subgraphs(from_sg: list[tuple, tuple], to_sg: list[tuple, tuple]):
            k = False
            for g in range(len(from_sg[1])):
                i = from_sg[1][g]
                if i in combined_vertices:
                    continue
                for j in to_sg[0]:
                    if j in combined_vertices:
                        continue
                    if new_graph.has_edge(i, j) and new_graph.can_combine_vertices(i, j):
                        new_graph.combine_vertices(i, j)
                        from_sg[1] = tuple(j if u == i else u for u in from_sg[1])
                        combined_vertices.append(i)
                        if k:
                            return
                        k = True
                    else:
                        if new_graph.can_add_edge(i, j):
                            new_graph.add_edge(i, j)
                            if k:
                                return
                            k = True

        visited_edges = []

        for u in r.vertices():
            if not r.is_input(u) and not r.is_output(u):
                sg = get_sub_graph(u)

                for v in r.in_vertices(u):
                    if r.is_input(v):
                        vs = list(connectable_subgraph_inputs(sg))
                        for _ in range(2):
                            _u = new_graph.get_next_unused_input()
                            _v = vs.pop()
                            new_graph.add_edge(_u, _v)
                    else:
                        if (v, u) not in visited_edges:
                            visited_edges.append((v, u))
                            from_sg = get_sub_graph(v)
                            connect_subgraphs(from_sg, sg)

                for v in r.out_vertices(u):
                    if r.is_output(v):
                        vs = list(connectable_subgraph_outputs(sg))
                        for _ in range(2):
                            _u = vs.pop()
                            _v = new_graph.get_next_unused_output()
                            new_graph.add_edge(_u, _v)
                    else:
                        if (u, v) not in visited_edges:
                            visited_edges.append((u, v))
                            to_sg = get_sub_graph(v)
                            connect_subgraphs(sg, to_sg)
        
        return new_graph
    
    def transposed(self) -> 'BeltGraph':
        """
        Create a new graph with all edges reversed and inputs/outputs swapped.
        
        This effectively reverses the flow direction of the balancer.
        
        Returns:
            BeltGraph: A new graph with reversed flow direction
        """
        graph = self.copy_graph()
        graph.inputs, graph.outputs = graph.outputs, graph.inputs
        edges = list(graph.edges())
        for u, v in edges:
            graph.delete_edge(u, v)
        # Separate the loops because edges can be added too soon, causing an error where a vertex has either three inputs or three outputs.
        for u, v in edges:
            graph.add_edge(v, u)
        graph = graph.rearrange_vertices_by_depth()
        return graph
    
    def debug_vertex(self, u: int) -> None:
        """
        Print debug information about a vertex and its connections.
        
        Args:
            u: Vertex ID to debug
        """
        print(f'{self.vertex_to_str(u)}: [{", ".join(map(self.vertex_to_str, self.in_vertices(u)))}] -> {self.vertex_to_str(u)} -> [{", ".join(map(self.vertex_to_str, self.out_vertices(u)))}]')
    
    def can_combine_vertices(self, u: int, v: int) -> bool:
        """
        Check if two vertices can be combined to simplify the graph.
        
        Args:
            u: First vertex ID
            v: Second vertex ID
            
        Returns:
            bool: True if the vertices can be combined, False otherwise
        """
        return not self.is_input(u) and not self.is_output(v) and self.out_degree(v) > 0 and ((self.out_degree(u) < 2 and self.in_degree(v) < 2) or self.out_vertices(u) == (v, v))
    
    def combine_vertices(self, u: int, v: int) -> None:
        """
        Combine two vertices to simplify the graph structure.
        
        In the context of belts, this is the act of connecting the two outputs of one splitter
        to the two inputs of another splitter, so that effectively nothing is changed about the flow.
        This allows us to remove one of the splitters while preserving the same flow behavior.
        
        Args:
            u: First vertex ID
            v: Second vertex ID
            
        Raises:
            AssertionError: If the vertices cannot be combined
        """
        assert self.can_combine_vertices(u, v), f'Cannot combine vertices {self.vertex_to_str(u)} and {self.vertex_to_str(v)}.' + ('' if self.out_degree(v) > 0 else f' Vertex {self.vertex_to_str(v)} has no outbound edges.  Are you combining a series of vertices such that vertices are being removed and then referenced later for combination?')
        if self.has_edge(u, v):
            self.delete_edge(u, v)
        in_vertices = self.in_vertices(u)
        for _u in in_vertices:
            self.delete_edge(_u, u)
            if _u == v:
                continue
            self.add_edge(_u, v)
    
    def disconnect_input(self, u: int) -> int:
        """
        Remove an input vertex and its connections from the graph.
        
        Args:
            u: Input vertex ID to disconnect
            
        Returns:
            int: The vertex ID to which the input was connected, or -1 if not connected
        """
        v = -1
        if self.out_degree(u) == 1:
            v = self.out_vertices(u)[0]
            self.delete_edge(u, v)
        self.set_input(u, False)
        self.delete_vertex(u)
        return v
    
    def disconnect_output(self, u: int) -> int:
        """
        Remove an output vertex and its connections from the graph.
        
        Args:
            u: Output vertex ID to disconnect
            
        Returns:
            int: The vertex ID that was connected to the output, or -1 if not connected
        """
        v = -1
        if self.in_degree(u) == 1:
            v = self.in_vertices(u)[0]
            self.delete_edge(v, u)
        self.set_output(u, False)
        self.delete_vertex(u)
        return v
    
    def disconnect_num_inputs(self, amount: int) -> list[int]:
        """
        Disconnect a specified number of input vertices from the graph.
        
        Args:
            amount: Number of inputs to disconnect
            
        Returns:
            list: Vertex IDs that were connected to the inputs
            
        Raises:
            AssertionError: If there aren't enough inputs to disconnect
        """
        assert self.num_inputs >= amount, f'Cannot disconnect {amount} inputs from the current graph with {self.num_inputs} total inputs'
        vs = []
        for _ in range(amount):
            v = self.disconnect_input(self.inputs[-1])
            if v == -1:
                continue
            vs.append(v)
        return vs
    
    def disconnect_num_outputs(self, amount: int) -> list[int]:
        """
        Disconnect a specified number of output vertices from the graph.
        
        Args:
            amount: Number of outputs to disconnect
            
        Returns:
            list: Vertex IDs that were connected to the outputs
            
        Raises:
            AssertionError: If there aren't enough outputs to disconnect
        """
        assert self.num_outputs >= amount, f'Cannot disconnect {amount} outputs from the current graph with {self.num_outputs} total outputs'
        vs = []
        for _ in range(amount):
            v = self.disconnect_output(self.outputs[-1])
            if v == -1:
                continue
            vs.append(v)
        return vs

    def disconnect_vertex(self, u: int) -> Union[int, Tuple[Tuple, Tuple]]:
        """
        Disconnect a vertex from the graph, handling inputs, outputs, and internal vertices differently.
        
        For inputs and outputs, calls disconnect_input() or disconnect_output().
        For internal vertices, removes the vertex and all its edges, returning
        information about its connections.
        
        Args:
            u: Vertex ID to disconnect
            
        Returns:
            For inputs/outputs: The vertex ID that was connected to the vertex
            For internal vertices: A tuple of input vertices and output vertices that were connected
        """
        if self.is_input(u):
            return self.disconnect_input(u)
        if self.is_output(u):
            return self.disconnect_output(u)
            
        # Store connections before removing the vertex
        in_vertices = self.in_vertices(u)
        out_vertices = self.out_vertices(u)
        
        # Remove all connections
        for in_vertex in in_vertices:
            self.delete_edge(in_vertex, u)
        for out_vertex in out_vertices:
            self.delete_edge(u, out_vertex)
            
        # Remove the vertex itself
        self.delete_vertex(u)
        
        # Return the connection information
        return in_vertices, out_vertices
    
    def delete_identity_vertex(self, u: int) -> None:
        """
        Delete a 1-1 passthrough vertex and connect its input directly to its output.
        
        This removes an unnecessary intermediary vertex that doesn't affect flow.
        
        Args:
            u: Vertex ID to delete
            
        Raises:
            AssertionError: If the vertex is not an identity vertex (1 input, 1 output)
        """
        assert self.is_identity(u), (
            f'The vertex {u} is not an identity type of vertex. '
            f'It does not have exactly one inbound edge and one outbound edge.'
        )
        
        # Get the input and output vertices
        input_vertex = self.in_vertices(u)[0]
        output_vertex = self.out_vertices(u)[0]
        
        # Delete the vertex
        self.delete_vertex(u)
        
        # Connect input to output, handling the case where they're already connected
        if self.has_edge(input_vertex, output_vertex):
            self.combine_vertices(input_vertex, output_vertex)
        else:
            self.add_edge(input_vertex, output_vertex)
    
    def delete_balancer_vertex(self, u: int, swap: bool = False) -> None:
        """
        Delete a 2-2 balancer vertex and reconnect its inputs and outputs directly.
        
        This simplifies the graph by removing unnecessary balancers that don't
        affect the overall flow properties.
        
        Args:
            u: Vertex ID to delete
            swap: If True, connect inputs to outputs in the alternative arrangement
            
        Raises:
            AssertionError: If the vertex is not a balancer (2 inputs, 2 outputs)
            Exception: For special cases that can't be handled
        """
        assert self.is_balancer(u), (
            f'The vertex {u} is not a balancer type of vertex. '
            f'It does not have exactly two inbound edges and two outbound edges.'
        )
        
        # Get input and output connections
        (in1, _), (in2, _) = self.in_edges(u)
        (_, out1), (_, out2) = self.out_edges(u)
        
        # Determine how to reconnect
        if swap or in1 == out1 or in2 == out2:
            out1, out2 = out2, out1
            
        # Check for special case
        if in1 == in2:
            raise Exception('Unhandled case: both inputs are the same vertex')
            
        # Delete the vertex and reconnect
        self.delete_vertex(u)
        self.add_edge_or_combine(in1, out1)
        self.add_edge_or_combine(in2, out2)
    
    def simplify(self) -> None:
        """
        Simplify the graph by removing unnecessary vertices.
        
        This repeatedly removes identity vertices (1-1 passthrough) and balancer
        vertices (2-2) that don't affect the overall flow, until no more can be removed.
        """
        while True:
            # Find the next removable vertex
            v = next(self.removable_vertices(), -1)
            if v == -1:
                break
                
            # Remove the vertex appropriately based on its type
            if self.is_identity(v):
                self.delete_identity_vertex(v)
            elif self.is_balancer(v):
                self.delete_balancer_vertex(v)
    
    def new_vertex(self) -> int:
        """
        Get an unused vertex ID for creating a new vertex.
        
        Returns:
            int: A vertex ID that isn't currently in use
        """
        vertices = self.vertices()
        for u in range(len(vertices) + 1):
            if u not in vertices:
                return u
        # Fallback (shouldn't be reached in normal operation)
        return len(vertices)
    
    def rearrange_vertices(self, vertex_map: List[int]) -> 'BeltGraph':
        """
        Create a new graph with vertices rearranged according to the mapping.
        
        Args:
            vertex_map: List where the index is the old vertex ID and the value is the new ID
            
        Returns:
            BeltGraph: A new graph with rearranged vertices
        """
        graph = BeltGraph()
        
        # Set inputs and outputs
        for u in self.inputs:
            graph.set_input(vertex_map[u])
        for v in self.outputs:
            graph.set_output(vertex_map[v])
            
        # Add edges
        for u, v in self.edges():
            graph.add_edge(vertex_map[u], vertex_map[v])
            
        return graph
    
    def rearrange_vertices_by_depth(self, depths: Optional[Dict[int, int]] = None) -> 'BeltGraph':
        """
        Create a new graph with vertices arranged by their depth from inputs.
        
        Vertices are arranged so that those closer to inputs come first,
        which makes the graph structure clearer.
        
        Args:
            depths: Optional mapping of vertex IDs to their depths. 
                   If None, it will be calculated.
                   
        Returns:
            BeltGraph: A new graph with vertices arranged by depth
        """
        if depths is None:
            depths = self.get_vertex_depths()
            
        # Create a mapping from old vertex IDs to new vertex IDs
        # based on the depth of each vertex
        vmap = [None] * (max(self.vertices()) + 1)
        for i, (_, u) in enumerate(sorted((d, vertex) for vertex, d in depths.items())):
            vmap[u] = i
            
        # Handle vertices not in the depths dictionary
        max_mapped = -1
        for v in vmap:
            if v is not None and v > max_mapped:
                max_mapped = v
                
        for u in self.vertices():
            if vmap[u] is None:
                max_mapped += 1
                vmap[u] = max_mapped
                
        return self.rearrange_vertices(vmap)
    
    def rearrange_vertices_by_dfs(self) -> 'BeltGraph':
        """
        Create a new graph with vertices arranged by depth-first search order.
        
        This places connected vertices closer together in the vertex numbering.
        For example, if A -> B and A -> C and B -> D, then the rearranged graph
        would have vertices in the order of A, B, D, C.
        
        Returns:
            BeltGraph: A new graph with vertices arranged by DFS order
        """
        vmap = [None] * (max(self.vertices()) + 1)
        counter = 0
        
        def dfs(u: int) -> None:
            nonlocal counter
            if vmap[u] is not None:
                return
                
            # Assign the current counter value to this vertex
            vmap[u] = counter
            counter += 1
            
            # Recursively process outgoing vertices
            for v in self.out_vertices(u):
                dfs(v)
                
        # Start DFS from each input
        for u in self.inputs:
            dfs(u)
            
        return self.rearrange_vertices(vmap)
    
    def dfs(self, u: int = -1) -> Iterator[int]:
        """
        Perform a depth-first search traversal of the graph.
        
        Traverses from inputs to outputs, following the flow direction.
        
        Args:
            u: Starting vertex ID. If -1, starts from all input vertices.
            
        Yields:
            int: Vertex IDs in DFS traversal order
        """
        visited = set()
        
        def dfs_helper(v: int) -> Iterator[int]:
            if v in visited:
                return
                
            visited.add(v)
            yield v
            
            # Recursively visit outgoing neighbors
            for w in self.out_vertices(v):
                yield from dfs_helper(w)
                
        if u == -1:
            # Start from all inputs
            for input_vertex in self.inputs:
                yield from dfs_helper(input_vertex)
        else:
            # Start from the specified vertex
            yield from dfs_helper(u)
    
    def dfs_bidirectional(self, u: int = -1) -> Iterator[int]:
        """
        Perform a bidirectional depth-first search of the graph.
        
        This traverses both incoming and outgoing edges, useful for finding
        connected components regardless of flow direction.
        
        Args:
            u: Starting vertex ID. If -1, starts from all inputs and outputs.
            
        Yields:
            int: Vertex IDs in bidirectional DFS traversal order
        """
        visited = set()
        
        def dfs_helper(v: int) -> Iterator[int]:
            if v in visited:
                return
                
            visited.add(v)
            yield v
            
            # Visit outgoing neighbors
            for w in self.out_vertices(v):
                yield from dfs_helper(w)
                
            # Visit incoming neighbors
            for w in self.in_vertices(v):
                yield from dfs_helper(w)
                
        if u == -1:
            # Start from all inputs and outputs
            for input_vertex in self.inputs:
                yield from dfs_helper(input_vertex)
            for output_vertex in self.outputs:
                yield from dfs_helper(output_vertex)
        else:
            # Start from the specified vertex
            yield from dfs_helper(u)
    
    def dfs_reverse(self, u: int = -1) -> Iterator[int]:
        """
        Perform a reverse depth-first search of the graph.
        
        Traverses from outputs to inputs, opposite to the flow direction.
        
        Args:
            u: Starting vertex ID. If -1, starts from all output vertices.
            
        Yields:
            int: Vertex IDs in reverse DFS traversal order
        """
        visited = set()
        
        def dfs_helper(v: int) -> Iterator[int]:
            if v in visited:
                return
                
            visited.add(v)
            yield v
            
            # Visit incoming neighbors
            for w in self.in_vertices(v):
                yield from dfs_helper(w)
                
        if u == -1:
            # Start from all outputs
            for output_vertex in self.outputs:
                yield from dfs_helper(output_vertex)
        else:
            # Start from the specified vertex
            yield from dfs_helper(u)
    
    def removable_vertices(self) -> Iterator[int]:
        """
        Find vertices that can be safely removed without changing balancer behavior.
        
        This identifies vertices that are identity nodes (1-to-1 connections) or
        balancer nodes (2-to-2 connections) that can be simplified.
        
        Yields:
            int: Vertex IDs that can be safely removed
        """
        for u in self.vertices():
            # Identity vertices (1-1 connections) can always be removed
            if self.is_identity(u):
                yield u
                
            # Balancer vertices (2-2 connections) can be removed if the
            # graph remains balanced without them
            if self.is_balancer(u):
                # Make a copy to test removal
                graph = self.copy_graph()
                graph.delete_balancer_vertex(u)
                
                # If the graph still balances correctly after removal, the vertex is removable
                if graph.is_solved():
                    yield u
    
    def possible_new_edges(self) -> Iterator[Tuple[int, int]]:
        """
        Find all possible new edges that could be added to the graph.
        
        This includes edges between existing vertices and edges to potential new vertices.
        
        Yields:
            tuple: (u, v) pairs representing possible new edges
        """
        vertices = self.vertices()
        av_vertices = len(vertices) - self.num_inputs - self.num_outputs
        nv = self.new_vertex()
        
        # If we have no internal vertices yet, suggest creating an initial pair
        if av_vertices == 0:
            yield nv, nv+1
            return
            
        # If we have only one internal vertex, suggest connecting to a new vertex
        if av_vertices == 1:
            yield vertices[0], nv
            return
            
        # Otherwise, check all possible vertex pairs
        aug_vertices = vertices + [nv]
        for i, u in enumerate(aug_vertices[:-1]):
            for v in aug_vertices[i+1:]:
                if self.can_add_edge(u, v):
                    yield u, v
                if self.can_add_edge(v, u):
                    yield v, u
    
    def likely_new_edges(self) -> Iterator[Tuple[int, int]]:
        """
        Find edges that would make progress toward functional vertices.
        
        This prioritizes edges that would turn nonfunctional vertices into
        splitters, mergers, or balancers.
        
        Yields:
            tuple: (u, v) pairs representing recommended new edges
        """
        # Get all possible new edges
        ne = list(self.possible_new_edges())
        
        # Find vertices that aren't yet functioning as splitters, mergers, or balancers
        vertices = self.vertices()
        nonfunctional_vertices = [u for u in vertices if not self.is_functional(u)]
        
        # Prioritize edges involving nonfunctional vertices
        prioritized_edges = []
        if nonfunctional_vertices:
            for i in range(len(ne)-1, -1, -1):  # Iterate in reverse to maintain order
                u, v = ne[i]
                if u in nonfunctional_vertices or v in nonfunctional_vertices:
                    prioritized_edges.append((u, v))
        
        # If we have enough prioritized edges, use those
        # Otherwise use all possible edges
        result_edges = prioritized_edges if len(prioritized_edges) > 4 else ne
        yield from result_edges
    
    def possible_actions(self) -> Iterator[Tuple[str, Tuple]]:
        """
        Generate all possible actions that can be performed on the graph.
        
        Actions include adding edges or deleting edges.
        
        Yields:
            tuple: (action_name, (args)) pairs representing possible actions
        """
        # Actions to add new edges
        for edge in self.possible_new_edges():
            yield 'add_edge', edge
            
        # Actions to delete existing edges
        for edge in self.edges():
            yield 'delete_edge', tuple(edge)
    
    def do_action(self, action: Tuple[str, Tuple]) -> None:
        """
        Perform the specified action on the graph.
        
        Args:
            action: A tuple of (action_name, args) where action_name is
                   'add_edge' or 'delete_edge' and args are the edge vertices
        """
        name, args = action
        if name == 'add_edge':
            self.add_edge(*args)
        elif name == 'delete_edge':
            self.delete_edge(*args)
    
    def evaluate(self) -> 'Evaluation':
        """
        Evaluate the graph's performance as a balancer.
        
        This calculates various metrics including flow distribution, accuracy,
        bottlenecks, and balance quality.
        
        Returns:
            Evaluation: An object containing various performance metrics
        """
        # Calculate flow distribution through the graph
        flow_points = self.calculate_flow_points()
        
        # Extract output flow (values at output vertices)
        output_flow = {k: v for k, v in flow_points.items() if self.is_output(k)}
        
        # Find bottlenecks (vertices where flow exceeds capacity)
        bottlenecks = [k for k, v in sorted(flow_points.items()) if sum(v.values()) > 1]
        
        # Calculate accuracy metrics
        accuracy, error, score = measure_accuracy(output_flow, self.inputs, self.outputs)
        
        # Create and populate Evaluation object
        evaluation = Evaluation()
        evaluation.update({
            'flow_points': flow_points,
            'output_flow': output_flow,
            'bottlenecks': bottlenecks,
            'accuracy': accuracy,
            'error': error,
            'score': score,
        })
        
        return evaluation
    
    def calculate_flow_points(self) -> Dict[int, Dict[int, float]]:
        """
        Calculate the flow distribution at each point in the graph.
        
        Returns:
            dict: Mapping of vertex IDs to their flow distribution dictionaries
        """
        return calculate_flow_points(*construct_relations(self))
    
    def is_solved(self, ev: Optional[Evaluation] = None) -> bool:
        """
        Check if the balancer correctly distributes flow with no bottlenecks.
        
        A solved balancer has perfect accuracy (1.0) and has no bottlenecks
        unless there are fewer outputs than inputs.
        
        Args:
            ev: Optional pre-computed Evaluation. If None, it will be calculated.
            
        Returns:
            bool: True if the balancer is solved, False otherwise
        """
        if ev is None:
            ev = self.evaluate()
            
        return (ev['accuracy'] == 1 and 
                (self.num_outputs < self.num_inputs or len(ev['bottlenecks']) == 0))

    def get_vertex_depths(self) -> Dict[int, int]:
        """
        Calculate the depth of each vertex from the inputs.
        
        The depth represents how many steps a vertex is from the input.
        This is useful for organizing the graph in a layered visualization.
        
        Returns:
            dict: Mapping of vertex IDs to their depths
        """
        def dfs(u: int, d: int) -> None:
            # If we've already visited this vertex, update its depth if needed
            if u in dfs.depth:
                old = dfs.depth[u]
                dfs.depth[u] = min(dfs.depth[u], d)
                if d < old:
                    # If we found a shorter path, update successors
                    for v in self.out_vertices(u):
                        dfs(v, d + 1)
            else:
                # First visit to this vertex
                dfs.depth[u] = d
                for v in self.out_vertices(u):
                    dfs(v, d + 1)
        
        # Initialize depth tracking
        dfs.depth = {}
        
        # Start DFS from each input
        for u in self.inputs:
            dfs.visited = [False] * self.num_vertices
            dfs(u, 0)
            
        return dfs.depth
    
    def get_vertices_by_depth(self, depth: Optional[Dict[int, int]] = None) -> List[List[int]]:
        """
        Group vertices by their depth in the graph.
        
        Args:
            depth: Optional pre-computed depth dictionary. If None, it will be calculated.
            
        Returns:
            list: A list where each element is a list of vertices at that depth
        """
        if depth is None:
            depth = self.get_vertex_depths()
            
        if len(depth) == 0:
            return []
            
        # Create lists for each depth level
        max_depth = max(depth.values())
        transpose = [[] for _ in range(max_depth + 1)]
        
        # Group vertices by depth
        for u, d in depth.items():
            transpose[d].append(u)
            
        return transpose
    
    def disjoint_vertices(self) -> List[List[int]]:
        """
        Find sets of vertices that form disjoint subgraphs.
        
        This identifies separate connected components within the graph.
        
        Returns:
            list: A list where each element is a list of vertices in the same component
        """
        # Find all connected components using bidirectional DFS
        components = []
        
        for u in self.vertices():
            # Skip if this vertex is already in a component
            if any(u in component for component in components):
                continue
                
            # Do a bidirectional DFS to find all vertices in this component
            components.append(list(self.dfs_bidirectional(u)))
            
        return components
    
    def separated(self) -> List['BeltGraph']:
        """
        Split the graph into disjoint subgraphs.
        
        For example, if there's a 1-2 graph and a 4-4 graph contained within this graph,
        this function would return a list of two separate graphs.
        
        Returns:
            list: A list of BeltGraph objects, each representing a disjoint subgraph
        """
        graphs = []
        
        def get_connected_graph(u: int) -> 'BeltGraph':
            """Extract a connected subgraph containing vertex u."""
            graph = BeltGraph()
            visited = [False] * (self.num_vertices * 2)  # Extra space to prevent index errors
            
            def dfs(v: int) -> None:
                """Depth-first search to build the connected component."""
                if visited[v]:
                    return
                    
                visited[v] = True
                graph.add_vertex(v)
                
                # Connect to outgoing neighbors
                for w in self.out_vertices(v):
                    if graph.can_add_edge(v, w):
                        graph.add_edge(v, w)
                    dfs(w)
                    
                # Connect to incoming neighbors
                for w in self.in_vertices(v):
                    if graph.can_add_edge(w, v):
                        graph.add_edge(w, v)
                    dfs(w)
            
            # Build the subgraph
            dfs(u)
            
            # Mark vertices with no inputs as inputs and no outputs as outputs
            for v in graph.vertices():
                if graph.in_degree(v) == 0:
                    graph.set_input(v, True)
                if graph.out_degree(v) == 0:
                    graph.set_output(v, True)
                    
            return graph
        
        # Create a separate graph for each root component
        # (starting from vertices with no inputs)
        for u in self.vertices():
            if self.in_degree(u) == 0:
                graphs.append(get_connected_graph(u))
                
        return graphs
    
    @property
    def edges_string(self) -> str:
        """
        Generate a string representation of the graph's structure.
        
        The format is "inputs;outputs;connections" where:
        - inputs: comma-separated list of input vertex IDs
        - outputs: comma-separated list of output vertex IDs
        - connections: space-separated list of "u:v1,v2" where u is a vertex and v1,v2 are its outgoing connections
        
        Returns:
            str: A string representation of the graph structure
        """
        # Format: "inputs;outputs;connections"
        return '{};{};{}'.format(
            # Inputs section
            ",".join(map(str, self.inputs)),
            
            # Outputs section
            ",".join(map(str, self.outputs)),
            
            # Connections section
            ' '.join(
                '{}:{}'.format(
                    u,
                    ','.join(
                        str(v)
                        for _, v in self.out_edges(u)
                    )
                )
                for u in self.vertices()
                if self.out_degree(u) > 0
            )
        )
    
    @property
    def compact_string(self) -> str:
        """
        Generate a compressed string representation of the graph's structure.
        
        This compresses the edges_string using zlib and base64 encoding.
        
        Returns:
            str: A compressed string representation of the graph
        """
        return edges_string_to_compact_string(self.edges_string)
    
    def clear(self) -> None:
        """
        Remove all vertices and edges from the graph.
        
        This resets the graph to an empty state.
        """
        for v in self.vertices():
            self.delete_vertex(v)
        self.inputs.clear()
        self.outputs.clear()
    
    def load_edges_string(self, edges_string: str) -> None:
        """
        Load a graph structure from an edges string representation.
        
        Args:
            edges_string: A string in the format "inputs;outputs;connections"
        """
        self.clear()
        
        # Split the string into sections
        inputs_str, outputs_str, neighbors_str_list = edges_string.split(';')
        
        # Load inputs
        if inputs_str:
            for u_str in inputs_str.split(','):
                self.set_input(int(u_str), True)
                
        # Load outputs
        if outputs_str:
            for v_str in outputs_str.split(','):
                self.set_output(int(v_str), True)
                
        # Load connections
        if neighbors_str_list:
            for neighbors_str in neighbors_str_list.split(' '):
                v_str, u_str = neighbors_str.split(':')
                v = int(v_str)
                
                if ',' in u_str:
                    # Handle case with two outputs
                    u1, u2 = u_str.split(',')
                    self.add_edge(v, int(u1))
                    self.add_edge(v, int(u2))
                else:
                    # Handle case with one output
                    self.add_edge(v, int(u_str))
    
    def load_compact_string(self, compact_string: str) -> None:
        """
        Load a graph structure from a compact string representation.
        
        Args:
            compact_string: A compressed string created by compact_string property
        """
        self.load_edges_string(compact_string_to_edges_string(compact_string))
    
    def load_blueprint_string(self, blueprint_string: str, verbose: bool = True) -> None:
        """
        Load a graph structure from a Factorio blueprint string.
        
        Args:
            blueprint_string: A Factorio blueprint string
            verbose: If True, print diagnostic messages
        """
        try_derive_graph(load_blueprint_string(blueprint_string), self, verbose=verbose)
    
    def load_blueprint_string_from_file(self, fpath: str, verbose: bool = True) -> None:
        """
        Load a graph structure from a file containing a Factorio blueprint string.
        
        Args:
            fpath: Path to the file containing the blueprint string
            verbose: If True, print diagnostic messages
        """
        try_derive_graph(load_blueprint_string_from_file(fpath), self, verbose=verbose)
    
    def load_common_balancer(self, balancer_type: str) -> None:
        """
        Load a predefined common balancer structure by its type.
        
        Args:
            balancer_type: A string in the format "inputs-outputs", e.g., "4-4"
            
        Raises:
            Exception: If the balancer type is not defined in common_balancers
        """
        if balancer_type not in common_balancers:
            raise Exception(f'Common balancers does not contain a definition for balancer type {balancer_type}.')
        self.load_compact_string(common_balancers[balancer_type])
    
    def load_factorio_sat_network(self, path: str) -> None:
        """
        Load a BeltGraph from a file that can be read by the Factorio-SAT solver.
        
        The Factorio-SAT solver by R-O-C-K-E-T uses a specific file format to 
        describe belt networks.
        
        Args:
            path: Path to the SAT network file
        """
        # Read the network file
        with open(path) as f:
            s_arr = [list(map(int, line.strip().split())) for line in f]
            
        # Process each line as a vertex with connections
        for u, (i1, i2, o1, o2) in enumerate(s_arr):
            # Set inputs
            if i1 == 0:
                self.set_input(u, True)
            if i2 == 0:
                self.set_input(u, True)
                
            # Set outputs
            if o1 == 1:
                self.set_output(u, True)
            if o2 == 1:
                self.set_output(u, True)
                
            # Add connections from first input
            if i1 > 1:
                if o1 > 1:
                    self.add_edge(i1, o1)
                if o2 > 1:
                    self.add_edge(i1, o2)
                    
            # Add connections from second input
            if i2 > 1:
                if o1 > 1:
                    self.add_edge(i2, o1)
                if o2 > 1:
                    self.add_edge(i2, o2)
    
    @classmethod
    def from_blueprint_string(cls, blueprint_string: str, verbose: bool = True) -> 'BeltGraph':
        """
        Create a new BeltGraph from a Factorio blueprint string.
        
        Args:
            blueprint_string: A Factorio blueprint string
            verbose: If True, print diagnostic messages
            
        Returns:
            BeltGraph: A new graph loaded from the blueprint
        """
        graph = cls()
        graph.load_blueprint_string(blueprint_string, verbose=verbose)
        return graph
    
    @classmethod
    def from_blueprint_string_file(cls, fpath: str, verbose: bool = True) -> 'BeltGraph':
        """
        Create a new BeltGraph from a file containing a Factorio blueprint string.
        
        Args:
            fpath: Path to the file containing the blueprint string
            verbose: If True, print diagnostic messages
            
        Returns:
            BeltGraph: A new graph loaded from the file
        """
        graph = cls()
        graph.load_blueprint_string_from_file(fpath, verbose=verbose)
        return graph
    
    @classmethod
    def from_edges_string(cls, edges_string: str) -> 'BeltGraph':
        """
        Create a new BeltGraph from an edges string representation.
        
        Args:
            edges_string: A string in the format "inputs;outputs;connections"
            
        Returns:
            BeltGraph: A new graph loaded from the edges string
        """
        graph = cls()
        graph.load_edges_string(edges_string)
        return graph
    
    @classmethod
    def from_compact_string(cls, compact_string: str) -> 'BeltGraph':
        """
        Create a new BeltGraph from a compact string representation.
        
        Args:
            compact_string: A compressed string created by compact_string property
            
        Returns:
            BeltGraph: A new graph loaded from the compact string
        """
        graph = cls()
        graph.load_compact_string(compact_string)
        return graph
    
    @classmethod
    def from_common_balancers(cls, balancer_type: str) -> 'BeltGraph':
        """
        Create a new BeltGraph from a predefined common balancer structure.
        
        Args:
            balancer_type: A string in the format "inputs-outputs", e.g., "4-4"
            
        Returns:
            BeltGraph: A new graph loaded from the common balancer definition
        """
        graph = cls()
        graph.load_common_balancer(balancer_type)
        return graph
    
    def save_as_factorio_sat_network(self, path: Optional[str] = None) -> None:
        """
        Save the BeltGraph to a file that can be read by the Factorio-SAT solver.
        
        This creates a file in the format required by the Factorio-SAT solver by R-O-C-K-E-T.
        See: https://github.com/R-O-C-K-E-T/Factorio-SAT/tree/main
        
        Args:
            path: Directory path to save the file. If None, saves to current directory.
        """
        # Set default path to current directory
        if path is None:
            path = '.'
            
        # Create directory if it doesn't exist
        if not os.path.isdir(path):
            os.makedirs(path)
            
        # Generate filename from balancer dimensions
        fname = f'{self.num_inputs}x{self.num_outputs}'
        
        # Rearrange vertices for better readability
        r = self.rearrange_vertices_by_depth()
        elist = list(r.edges())
        
        # Write to file
        with open(f'{path}/{fname}', 'w') as f:
            # Initialize array for SAT network format
            s_arr = [[-1] * 4 for _ in range(r.num_vertices + 1)]
            
            # Process each edge
            for t, (u, v) in enumerate(elist):
                # Set output connection for source vertex
                k = 1  # Default for output to output vertex
                if not r.is_output(v):
                    k = t + 2  # Index for internal connection
                    
                if s_arr[u][3] == -1:
                    s_arr[u][3] = k  # First output slot
                else:
                    s_arr[u][2] = k  # Second output slot
                
                # Set input connection for target vertex
                k = 0  # Default for input from input vertex
                if not r.is_input(u):
                    k = t + 2  # Index for internal connection
                    
                if s_arr[v][1] == -1:
                    s_arr[v][1] = k  # First input slot
                else:
                    s_arr[v][0] = k  # Second input slot
            
            # Write only relevant rows (those with connections)
            print('\n'.join(
                ' '.join(map(str, a)) 
                for u, a in enumerate(s_arr) 
                if (a[3] != -1 or a[2] != -1) and (a[1] != -1 or a[0] != -1)
            ), file=f)
    
    def vertex_to_str(self, u: int) -> str:
        """
        Convert a vertex ID to a string representation, marking inputs and outputs.
        
        Args:
            u: Vertex ID
            
        Returns:
            str: String representation of the vertex
        """
        if self.is_input(u):
            return f'{u}*'  # Mark inputs with asterisk at end
        if self.is_output(u):
            return f'*{u}'  # Mark outputs with asterisk at start
        return str(u)
    
    def __str__(self) -> str:
        """
        Create a string representation of the entire graph.
        
        Returns:
            str: A formatted string showing the graph structure
        """
        line_width = 32  # Width for separator lines
        
        # Get depth information
        depths = self.get_vertex_depths()
        vertices_by_depth = self.get_vertices_by_depth(depths)
        
        # Build the string representation
        s = '=' * line_width
        s += '\n' + 'BeltGraph ' + self.summary
        
        # Add information for each depth level
        for depth, vertices in enumerate(vertices_by_depth):
            s += '\n' + '-' * line_width
            s += f'\nDepth: {depth}\n'
            
            # Add information for each vertex at this depth
            for k, u in enumerate(sorted(vertices)):
                if k > 0:
                    s += '\n'
                    
                # Add vertex and its connections
                if self.is_input(u):
                    s += '(input) '
                else:
                    in_vertices_str = ", ".join(map(self.vertex_to_str, self.in_vertices(u)))
                    s += f'[{in_vertices_str}] -> '
                    
                s += f'{self.vertex_to_str(u)}'
                
                if self.is_output(u):
                    s += ' (output)'
                else:
                    out_vertices_str = ", ".join(map(self.vertex_to_str, self.out_vertices(u)))
                    s += f' -> [{out_vertices_str}]'
                    
        s += '\n' + '=' * line_width
        return s
    
    def __hash__(self) -> int:
        """
        Generate a hash of the graph based on its edges.
        
        Returns:
            int: Hash value
        """
        # Hash based on the edges of the graph
        return hash(tuple(map(tuple, self.edges())))
