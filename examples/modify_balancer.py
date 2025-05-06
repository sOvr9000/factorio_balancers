
"""
Example of modifying an existing balancer.

This example demonstrates how to load a common balancer and modify it
by deleting an edge.
"""

from factorio_balancers import BeltGraph



def main():
    """Modify an existing balancer."""
    print("=== Modifying an existing balancer ===")
    
    # Load a common 4-4 balancer
    graph = BeltGraph()
    graph.load_common_balancer("4-4")
    
    print(f"Original balancer: {graph.balancer_type}")
    print(f"Number of internal vertices: {graph.num_internal_vertices}")
    
    # Delete an edge to modify the balancer
    edges = list(graph.edges())
    if len(edges) > 0:
        u, v = edges[0]
        print(f"Deleting edge {u} -> {v}")
        graph.delete_edge(u, v)
    
    # Re-evaluate the modified balancer
    evaluation = graph.evaluate()
    print(f"Modified balancer accuracy: {evaluation['accuracy']}")
    print(f"Is still solved: {graph.is_solved(evaluation)}")



if __name__ == "__main__":
    main()


