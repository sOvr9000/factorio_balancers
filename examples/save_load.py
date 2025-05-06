
"""
Example of saving and loading balancers.

This example demonstrates how to save a balancer as a compact string
and load it back into a new graph.
"""

from factorio_balancers import BeltGraph



def main():
    """Save a balancer as a compact string and load it back."""
    print("=== Saving and loading balancers ===")
    
    # Create a simple balancer
    graph = BeltGraph()
    graph.load_common_balancer("2-2")
    
    # Save as compact string
    compact_string = graph.compact_string
    print(f"Compact string representation: {compact_string}")
    
    # Load from compact string
    new_graph = BeltGraph.from_compact_string(compact_string)
    
    # Verify they are the same
    print(f"Original graph: {graph.balancer_type} with {graph.num_internal_vertices} internal vertices")
    print(f"Loaded graph: {new_graph.balancer_type} with {new_graph.num_internal_vertices} internal vertices")



if __name__ == "__main__":
    main()


