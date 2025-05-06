
"""
Sparse Solver Module for Factorio Belt Balancer Analysis

This module provides functions to analyze flow distribution in belt balancer networks.
It uses a sparse matrix approach to efficiently solve linear equations representing
belt flow through a network of splitters.

The key algorithms are:
- sparse_solve: Efficiently calculates output flows in terms of inputs (O(n log n))
- calculate_flow_points: Calculates flow at all points in the network (O(n²))

These functions help evaluate whether a belt balancer properly distributes input flow
across all output belts.
"""

from fractions import Fraction
from typing import List, Tuple
from .custom_types import Scalar, FlowDict, LinearCombination



def pretty_fraction(fraction: Scalar) -> str:
    """
    Convert a Fraction to a readable string representation.
    
    Args:
        fraction: A Fraction object or numeric value
        
    Returns:
        str: String representation in the form "numerator/denominator" or the original value
    """
    if isinstance(fraction, Fraction):
        return f'{fraction.numerator}/{fraction.denominator}'
    return str(fraction)

def substitute(relations: FlowDict, transitive: int, sub_linear_combination: LinearCombination) -> None:
    """
    Substitute a transitive variable with its linear combination in all relations.
    
    This function replaces occurrences of the transitive variable with its definition
    in all other relations, implementing the core matrix substitution algorithm.
    
    Args:
        relations: Dictionary mapping vertices to their flow relations
        transitive: The vertex ID to substitute
        sub_linear_combination: The linear combination defining the transitive vertex
    """
    # For each vertex and its linear combination
    for vertex, linear_combination in relations.items():
        # If the transitive variable appears in this linear combination
        if transitive in linear_combination:
            # Get the coefficient of the transitive variable
            scalar = linear_combination.pop(transitive)
            
            # Substitute with the transitive's definition, scaled by the coefficient
            for sub_vertex, sub_scalar in sub_linear_combination.items():
                if sub_vertex in linear_combination:
                    linear_combination[sub_vertex] += scalar * sub_scalar
                else:
                    linear_combination[sub_vertex] = scalar * sub_scalar
            
            # Handle self-referential flows
            if vertex in linear_combination:
                flow = linear_combination.pop(vertex)
                if flow == 1:
                    # Special case when coefficient is 1
                    del relations[transitive]
                    return
                
                # Rescale all coefficients due to self-reference
                scale = 1 / (1 - flow)
                for flow_vertex in linear_combination:
                    linear_combination[flow_vertex] *= scale

def sparse_solve(transitives: List[int], relations: FlowDict) -> FlowDict:
    """
    Solve a system of linear equations represented in sparse form.
    
    This function eliminates intermediate vertices (transitives) from the flow relations,
    resulting in output flows expressed directly in terms of input flows.
    
    Args:
        transitives: List of internal vertices (non-inputs, non-outputs) to eliminate
        relations: Dictionary mapping vertices to their flow relationships
    
    Returns:
        FlowDict: Dictionary mapping output vertices to their flow expressions in terms of inputs
    
    Example:
        For a 3-2 balancer:
        ```
        sparse_solve(
            [0, 1, 2, 3],  # Transitive vertices
            {
                0: {2: Fraction(1,2), 1: Fraction(1,2)},
                1: {4: Fraction(1,2), 6: Fraction(1,2)},
                2: {3: Fraction(1,2), 5: Fraction(1,2)},
                3: {1: 1, 2: 1},
                7: {0: 1},
                8: {0: 1},
            }
        )
        ```
        Returns:
        ```
        {
            7: {4: Fraction(1,2), 5: Fraction(1,2), 6: Fraction(1,2)},
            8: {4: Fraction(1,2), 5: Fraction(1,2), 6: Fraction(1,2)},
        }
        ```
        showing that outputs 7 and 8 each receive half of each input.
    """
    # Make a copy to avoid modifying the original
    relations_copy = {k: v.copy() for k, v in relations.items()}
    
    # Process each transitive vertex
    while len(transitives) > 0:
        transitive = transitives.pop()
        sub_linear_combination = relations_copy.pop(transitive) if transitive in relations_copy else {}
        substitute(relations_copy, transitive, sub_linear_combination)
    
    return relations_copy

def calculate_flow_points(transitives: List[int], relations: FlowDict) -> FlowDict:
    """
    Calculate the flow at every point in the graph including internal points.
    
    Similar to sparse_solve but keeps the internal structure and calculates flow
    at all vertices. This is useful for finding bottlenecks where flow exceeds capacity.
    
    Args:
        transitives: List of internal vertices to process
        relations: Dictionary mapping vertices to their flow relationships
    
    Returns:
        FlowDict: Dictionary mapping all vertices to their flow expressions
    
    Note:
        This is O(n²) compared to sparse_solve's O(n log n), but provides
        complete flow information at all points in the network.
    """
    # Make a copy to avoid modifying the original
    relations_copy = {k: v.copy() for k, v in relations.items()}
    
    # Process each transitive vertex
    for transitive in transitives:
        sub_linear_combination = relations_copy.get(transitive, {})
        substitute(relations_copy, transitive, sub_linear_combination)
    
    return relations_copy

def construct_relations(graph: 'BeltGraph') -> Tuple[List[int], FlowDict]:
    """
    Construct the initial flow relation equations from a BeltGraph.
    
    Args:
        graph: A BeltGraph object representing a Factorio belt network
        
    Returns:
        tuple: (transitives, relations) where:
            - transitives is a list of internal vertices to eliminate
            - relations is a dictionary mapping vertices to their initial flow relations
    
    Note:
        Each splitter divides input flow equally among its outputs.
        Input vertices are represented by their IDs (typically 4, 5, 6, etc.).
        Output vertices (7, 8, etc.) have flow values defined by their inputs.
    """
    # Define flow fractions for different numbers of outbound edges per vertex
    fractions = [None, 1, Fraction(1, 2)]
    
    # Identify internal vertices (non-inputs, non-outputs)
    transitives = [
        u
        for u in graph.vertices()
        if not graph.is_input(u) and not graph.is_output(u)
    ]
    
    # Construct flow relations for each vertex
    relations = {
        u: {
            v: 1 if graph.is_output(u) else fractions[graph.out_degree(u)]
            for v, _ in graph.in_edges(u)
        }
        for u in graph.vertices()
        if not graph.is_input(u) and (graph.out_degree(u) > 0 or (graph.is_output(u) and graph.in_degree(u) > 0))
    }
    
    return transitives, relations

def pretty_relations(relations: FlowDict) -> str:
    """
    Format flow relations as a readable string.
    
    Args:
        relations: Dictionary mapping vertices to their flow expressions
        
    Returns:
        str: Formatted string representing the flow relations
    """
    return '\n'.join(
        '{}:\t{}'.format(
            u,
            '\t'.join(
                f'{v}:{pretty_fraction(fraction)}'
                for v, fraction in lc.items()
            )
        )
        for u, lc in relations.items()
    )



# Example usage when run as a script
if __name__ == '__main__':
    from beltgraph import BeltGraph
    
    # Load a 11-11 balancer blueprint
    graph = BeltGraph.from_blueprint_string('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')
    
    # Construct flow relations from the graph
    transitives, relations = construct_relations(graph)
    
    # Calculate and display flow at all points
    print("Flow at all points in the network:")
    print(pretty_relations(calculate_flow_points(transitives.copy(), relations.copy())))
    
    print("\nOutput flows in terms of inputs:")
    print(pretty_relations(sparse_solve(transitives, relations)))
    
    print("\nDIFFERENCE BETWEEN CALCULATE_FLOW_POINTS AND SPARSE_SOLVE:")
    print("calculate_flow_points fully reveals the flow of the graph but is O(n²)")
    print("  - Useful for detecting critical points where flow is too concentrated/dense")
    print("sparse_solve reveals the flow of the graph only at its outputs but is O(n log n)")
    print("  - Useful for quickly testing the effectivity of a balancer graph")


