
"""
Factorio belt balancer utility functions.

This module provides functions for working with belt balancer designs,
including loading from Factorio blueprint strings and managing common balancer patterns.
"""

import os
import json
import zlib
import base64
from typing import Dict, Tuple, Any, Optional, Iterable, Tuple
from fischer.factoriobps import get_entity_map, is_blueprint_data

__all__ = [
    'load_common_balancers',
    'list_vertex_pairs',
    'compact_string_to_edges_string',
    'edges_string_to_compact_string',
    'get_balancer',
    'is_balancer_defined',
    'derive_graph',
    'try_derive_graph',
    'common_balancers',
]



def load_common_balancers() -> Dict[str, str]:
    """
    Load common balancers from a JSON file.
    
    Returns:
        dict: A dictionary of common balancers, where keys are balancer types and values are compact strings.
    """
    with open(os.path.join(os.path.dirname(__file__), 'common_balancers.json'), 'r') as f:
        return json.load(f)

common_balancers = load_common_balancers()



def list_vertex_pairs(iterable: Iterable[Tuple[int, int]]) -> None:
    """
    Print the vertex pairs in an iterable in a readable format.
    
    Args:
        iterable: An iterable containing (u, v) pairs
    """
    for u, v in iterable:
        print(f'{u} --> {v}')

def compact_string_to_edges_string(compact_string: str) -> str:
    """
    Decompress a compact string to an edges string.
    
    Args:
        compact_string: A compressed string created by edges_string_to_compact_string
        
    Returns:
        str: The decompressed edges string
    """
    return zlib.decompress(base64.b64decode(compact_string.encode('utf-8'))).decode('utf-8')

def edges_string_to_compact_string(edges_string: str) -> str:
    """
    Compress an edges string to a compact string.
    
    Args:
        edges_string: A string in the format "inputs;outputs;connections"
        
    Returns:
        str: The compressed compact string
    """
    return base64.b64encode(zlib.compress(edges_string.encode('utf-8'))).decode('utf-8')

def get_balancer(balancer_type: str) -> Optional[str]:
    """
    Get the compact string for a common balancer type.
    
    Args:
        balancer_type: The balancer type (e.g., "4-4")
        
    Returns:
        str or None: The compact string for the balancer, or None if not defined
    """
    return common_balancers.get(balancer_type, None)

def is_balancer_defined(balancer_type: str) -> bool:
    """
    Check if a balancer type is defined in the common_balancers dictionary.
    
    Args:
        balancer_type: The balancer type (e.g., "4-4")
        
    Returns:
        bool: True if the balancer is defined, False otherwise
    """
    return balancer_type in common_balancers

def derive_graph_from_entity_map(entity_map: Dict[Tuple[int, int], Tuple[str, Any]], 
                                graph: 'BeltGraph', 
                                max_underground_length: int) -> None:
    """
    Convert a Factorio entity map, typically obtained from factorio_blueprints.get_entity_map, to a BeltGraph structure.
    
    Args:
        entity_map: A dictionary mapping (x, y) coordinates to entity information
        graph: The BeltGraph to populate
        max_underground_length: Maximum length for underground belts
    
    Raises:
        Exception: If the blueprint contains sideloaded belts, no inputs, or no outputs
    """
    # Direction offsets for each of the 4 cardinal directions (N, E, S, W)
    OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def get_offset(d: int) -> Tuple[int, int]:
        """Get the x, y offset for a direction."""
        return OFFSETS[d]

    def step_pos(x: int, y: int, d: int) -> Tuple[int, int]:
        """Get the position one step in direction d from (x, y)."""
        _x, _y = get_offset(d)
        return x+_x, y+_y

    # Trace map to find connections between entities
    output_positions = []  # List of output belt positions
    trace = {}             # Maps position to next position(s)
    inv_trace = {}         # Maps position to previous position
    
    # Process each entity in the map
    for (x, y), (name, info) in entity_map.items():
        if name == 'belt':
            # For belts, trace to the next tile in belt direction
            d, = info  # Direction
            forward_pos = step_pos(x, y, d)
            trace[(x, y)] = forward_pos
            inv_trace[trace[(x, y)]] = (x, y)
            if forward_pos not in entity_map:
                output_positions.append((x, y))
                
        elif name == 'underground':
            # For underground belts, find the paired underground belt
            d, io_type = info  # Direction, input/output type
            _x, _y = get_offset(d)
            
            # Look for the matching underground belt
            for i in range(1, max_underground_length+2):
                nx = x + i * _x
                ny = y + i * _y
                if (nx, ny) in entity_map:
                    nname, ninfo = entity_map[(nx, ny)]
                    if nname == 'underground' and ninfo[0] == d:
                        break
            else:
                continue
                
            trace[(x, y)] = step_pos(nx, ny, d)
            inv_trace[trace[(x, y)]] = (x, y)
            
        elif name == 'splitter':
            # For splitters, trace both outputs
            d, _, (ox, oy), f = info  # Direction, ID, other half position, filters
            next_positions = []
            
            for i, p in enumerate((step_pos(x, y, d), step_pos(ox, oy, d))):
                # Skip if filtered or special cases where splitters wouldn't connect
                if f[i]:
                    continue
                if p in entity_map:
                    emp = entity_map[p]
                    # Skip if connecting to side of another splitter
                    if emp[0] == 'splitter' and emp[1][0] != d:
                        continue
                    # Skip special underground belt cases
                    if emp[0] == 'underground':
                        nd, io_type = emp[1]
                        if io_type == 'output' and nd == d:
                            continue
                        if io_type == 'input' and nd == (d + 2) % 4:
                            continue
                next_positions.append(p)
                
            trace[(x, y)] = next_positions
            trace[(ox, oy)] = next_positions
            for next_position in next_positions:
                inv_trace[next_position] = (x, y)

    # Depth-first search to build connections
    def dfs(x: int, y: int, px: int, py: int) -> None:
        """
        Depth-first search through the entity map to find connections.
        
        Args:
            x, y: Current position coordinates
            px, py: Previous position coordinates
        """
        if (x, y) not in trace:
            return
            
        name, info = entity_map[(x, y)]
        if name == 'splitter':
            _, _, (ox, oy), _ = info
            dfs.connections.append(((px, py), (x, y)))
            
            if (x, y) in dfs.visited:
                return
                
            dfs.visited.append((x, y))
            dfs.visited.append((ox, oy))
            
            for next_position in trace[(x, y)]:
                dfs(*next_position, x, y)
                
        elif name == 'belt' or name == 'underground':
            if (x, y) in dfs.visited:
                raise Exception('Cannot interpret a graph from a blueprint with sideloaded belts. '
                               'Bypass this by rearranging the splitters to accomplish the same effect without sideloading.')
                
            dfs.visited.append((x, y))
            next_position = trace[(x, y)]
            dfs(*next_position, px, py)
            
            if next_position not in trace:
                dfs.connections.append(((px, py), (x, y)))

    # Find input positions (belts with no incoming connections)
    input_positions = [p for p in trace.keys() 
                      if (p not in trace.values() and not any(p in v for v in trace.values() if isinstance(v, list))) 
                      and entity_map[p][0] == 'belt']
                      
    if len(input_positions) == 0:
        raise Exception('No input belts found')

    if len(output_positions) == 0:
        raise Exception('No output belts found')

    # Build the graph by tracing connections from inputs
    dfs.connections = []
    dfs.visited = []
    for x, y in input_positions:
        dfs(x, y, x, y)

    # Count splitters (each splitter has 2 halves in the entity map)
    num_splitters = sum(1 for name, _ in entity_map.values() if name == 'splitter') // 2
    
    # Clear and set up the graph
    graph.clear()
    for i in range(len(input_positions)):
        graph.set_input(num_splitters + i, True)
    for i in range(len(output_positions)):
        graph.set_output(num_splitters + graph.num_inputs + i, True)

    # Add edges based on the connections found during DFS
    for p1, p2 in dfs.connections:
        name1, info1 = entity_map[p1]
        name2, info2 = entity_map[p2]
        
        if name1 == 'splitter':  # from node
            _, id1, _, _ = info1
            if name2 == 'splitter':  # to node
                _, id2, _, _ = info2
            elif name2 == 'belt':  # to output
                id2 = graph.get_next_unused_output()
        elif name1 == 'belt':  # from input
            id1 = graph.get_next_unused_input()
            if name2 == 'splitter':  # to node
                _, id2, _, _ = info2
            elif name2 == 'belt':  # to output
                id2 = graph.get_next_unused_output()
                
        graph.add_edge(id1, id2)

def derive_graph(bp_data: Dict[str, Any], graph: 'BeltGraph', max_underground_length: int = 8) -> None:
    """
    Convert a Factorio blueprint data, typically obtained from factorio_blueprints.load_blueprint_string, to a BeltGraph structure.
    
    Args:
        bp_data: Blueprint data from Factorio
        graph: The BeltGraph to populate
        max_underground_length: Maximum length for underground belts
    """
    entity_map = get_entity_map(bp_data)
    derive_graph_from_entity_map(entity_map, graph, max_underground_length=max_underground_length)

def try_derive_graph(bp_data: Dict[str, Any], graph: 'BeltGraph', verbose: bool = True) -> bool:
    """
    Attempt to derive a graph from blueprint data, handling any errors gracefully.
    
    Args:
        bp_data: Blueprint data from Factorio
        graph: The BeltGraph to populate
        verbose: If True, print diagnostic messages
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not is_blueprint_data(bp_data):
        if verbose:
            print('The provided blueprint data does not represent a blueprint.')
        return False
        
    try:
        derive_graph(bp_data, graph)
        return True
    except Exception as e:
        if verbose:
            label = bp_data['blueprint'].get('label', 'Blueprint')
            print(f'Could not derive a graph from the provided blueprint (label: "{label}")')
            print(f'Error that occurred: {e}')
        return False


