
"""
Example of converting a balancer to different types.

This example demonstrates how to transform balancers by doubling them
or transposing them (reversing flow direction).
"""

from factorio_balancers import BeltGraph



def main():
    """Convert a balancer to a different type."""
    print("=== Converting a balancer to a different type ===")
    
    # Load a 4-4 balancer
    graph = BeltGraph()
    graph.load_common_balancer("4-4")
    
    print(f"Original balancer: {graph.balancer_type}")
    
    # Create a doubled version (8-8)
    doubled_graph = graph.doubled()
    
    print(f"Doubled balancer: {doubled_graph.balancer_type}")
    print(f"Number of internal vertices: {doubled_graph.num_internal_vertices}")
    
    # Create a transposed version (4-4 but with reversed flow)
    transposed_graph = graph.transposed()
    
    print(f"Transposed balancer: {transposed_graph.balancer_type}")
    print(f"Is the same structure: {graph.num_internal_vertices == transposed_graph.num_internal_vertices}")



if __name__ == "__main__":
    main()


