
"""
Factorio Belt Balancer Blueprint Analyzer

This script analyzes Factorio belt balancer blueprints from the clipboard,
evaluating their properties and checking if they are properly balanced.
It can also load balancers from various other sources like compact strings,
common balancers, Factorio SAT network files, or JSON files.
"""

import json
import argparse
from clipboard import paste
from factorio_balancers import BeltGraph



def analyze_blueprint_string_from_clipboard():
    """
    Load a blueprint from the clipboard and analyze its properties.
    
    This function loads a Factorio blueprint string from the clipboard,
    converts it to a BeltGraph, and evaluates its balance properties.
    """
    print("=== Analyzing Blueprint from Clipboard ===")
    
    # Create a new BeltGraph and load from clipboard
    graph = BeltGraph()
    
    # Attempt to load blueprint string.
    graph.load_blueprint_string(paste())
    
    if graph.balancer_type == '0-0':
        print('Failed to load blueprint from clipboard.')
        return

    print(f"Successfully loaded blueprint: {graph.balancer_type} balancer")
    
    # Display the graph structure
    print("\nGraph Structure:")
    print(graph)
    
    # Evaluate the balancer and display results
    print("\nBalance Evaluation:")
    evaluation = graph.evaluate()
    print(evaluation)
    
    # Check for removable/unnecessary vertices
    print("\nRemovable vertices:")
    removable = list(graph.removable_vertices())
    if removable:
        for u in removable:
            print(f"  Vertex {u}")
    else:
        print("  None - the balancer design is not able to be\n    simplified with the current methods of simplification")
    
    # Show summary information
    print("\nGraph summary:")
    print(graph.advanced_summary)
    
    # Check if the balancer is properly balanced
    is_solved = graph.is_solved()
    print("\nBalancer is properly balanced:", "✓ Yes" if is_solved else "✗ No")
    
    # If balanced, show the compact string for saving
    if is_solved:
        print("\nCompact string representation:")
        print(graph.compact_string)

def analyze_from_compact_string(compact_string):
    """
    Load a balancer from a compact string and analyze it.
    
    Args:
        compact_string: The compact string representation of a balancer
    """
    print("=== Analyzing Balancer from Compact String ===")
    
    graph = BeltGraph()
    graph.load_compact_string(compact_string)
    print(graph)
    print(graph.evaluate())
    print(f"Balancer is properly balanced: {graph.is_solved()}")

def analyze_common_balancer(balancer_type):
    """
    Load a common balancer by type and analyze it.
    
    Args:
        balancer_type: The balancer type (e.g., "4-4")
    """
    print(f"=== Analyzing Common {balancer_type} Balancer ===")
    
    graph = BeltGraph()
    try:
        graph.load_common_balancer(balancer_type)
        print(graph)
        print(graph.evaluate())
        print(f"Balancer is properly balanced: {graph.is_solved()}")
    except Exception as e:
        print(f"Error loading balancer: {e}")

def analyze_from_sat_network(file_path):
    """
    Load a balancer from a Factorio SAT network file and analyze it.
    
    Args:
        file_path: Path to the SAT network file
    """
    print(f"=== Analyzing Balancer from SAT Network: {file_path} ===")
    
    graph = BeltGraph()
    try:
        graph.load_factorio_sat_network(file_path)
        print(graph)
        print(graph.evaluate())
        print(f"Balancer is properly balanced: {graph.is_solved()}")
    except Exception as e:
        print(f"Error loading SAT network: {e}")

def analyze_from_json(file_path, balancer_key):
    """
    Load a balancer from a JSON file and analyze it.
    
    Args:
        file_path: Path to the JSON file containing balancer compact strings
        balancer_key: The key in the JSON dict for the desired balancer
    """
    print(f"=== Analyzing Balancer from JSON: {balancer_key} ===")
    
    try:
        with open(file_path, 'r') as f:
            balancers = json.load(f)
        
        if balancer_key not in balancers:
            print(f"Balancer '{balancer_key}' not found in JSON file")
            print(f"Available balancers: {', '.join(balancers.keys())}")
            return
        
        graph = BeltGraph()
        graph.load_compact_string(balancers[balancer_key])
        print(graph)
        print(graph.evaluate())
        print(f"Balancer is properly balanced: {graph.is_solved()}")
    except Exception as e:
        print(f"Error loading balancer from JSON: {e}")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Analyze Factorio belt balancer blueprints')
    
    # Create a group of mutually exclusive options for the source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--clipboard', action='store_true', 
                            help='Analyze blueprint from clipboard (default)')
    source_group.add_argument('--compact', type=str, metavar='STRING',
                            help='Analyze blueprint from compact string')
    source_group.add_argument('--common', type=str, metavar='TYPE',
                            help='Analyze common balancer (e.g., "4-4")')
    source_group.add_argument('--sat', type=str, metavar='FILE',
                            help='Analyze balancer from Factorio SAT network file')
    source_group.add_argument('--json', nargs=2, metavar=('FILE', 'KEY'),
                            help='Analyze balancer from JSON file with given key')
    
    return parser.parse_args()



def main():
    """Main function to parse arguments and run the appropriate analysis."""
    args = parse_arguments()
    
    # Determine which analysis function to run based on arguments
    if args.compact:
        analyze_from_compact_string(args.compact)
    elif args.common:
        analyze_common_balancer(args.common)
    elif args.sat:
        analyze_from_sat_network(args.sat)
    elif args.json:
        analyze_from_json(args.json[0], args.json[1])
    else:
        # Default to clipboard analysis
        analyze_blueprint_string_from_clipboard()



if __name__ == '__main__':
    main()


