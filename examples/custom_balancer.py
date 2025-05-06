
"""
Example of creating a custom 3-3 balancer.

This example demonstrates how to create a more complex balancer design,
add cross-connections to balance flow, and evaluate its effectiveness.
"""

from factorio_balancers import BeltGraph



def main():
    """Create a custom 3-3 balancer and test it."""
    print("=== Creating a custom 3-3 balancer ===")
    
    # Create a new graph
    graph = BeltGraph()
    graph.set_num_inputs(3)
    graph.set_num_outputs(3)
    
    # Add internal vertices - making a simple but imperfect design
    v1 = graph.new_vertex()
    v2 = graph.new_vertex()
    v3 = graph.new_vertex()
    
    # Connect inputs to internal vertices
    graph.add_edge(graph.inputs[0], v1)
    graph.add_edge(graph.inputs[1], v2)
    graph.add_edge(graph.inputs[2], v3)
    
    # Connect internal vertices to outputs
    graph.add_edge(v1, graph.outputs[0])
    graph.add_edge(v2, graph.outputs[1])
    graph.add_edge(v3, graph.outputs[2])
    
    # Add cross-connections to try to balance
    graph.add_edge(v1, v2)
    graph.add_edge(v2, v3)
    graph.add_edge(v3, v1)
    
    # Print the graph
    print(graph)
    
    # Evaluate the design
    evaluation = graph.evaluate()
    print(evaluation)
    
    # Save if it's a good design
    if graph.is_solved(evaluation):
        print("This is a good balancer design!")
        print(f"Compact string: {graph.compact_string}")
    else:
        print("This design needs improvement.")



if __name__ == "__main__":
    main()


