
"""
Example of creating a basic 2-2 balancer manually.

This example demonstrates how to create a simple belt balancer from scratch
by manually setting up the graph structure.
"""

from factorio_balancers import BeltGraph



def main():
    """Create a simple 2-2 balancer manually."""
    print("=== Creating a simple 2-2 balancer manually ===")
    
    # Create a new empty graph
    graph = BeltGraph()
    
    # Set inputs and outputs
    graph.set_num_inputs(2)
    graph.set_num_outputs(2)
    
    # Add internal vertices
    v1 = graph.new_vertex()
    v2 = graph.new_vertex()
    
    # Add edges to connect inputs to internal vertices
    graph.add_edge(graph.inputs[0], v1)
    graph.add_edge(graph.inputs[1], v2)
    
    # Add edges to connect internal vertices to outputs
    graph.add_edge(v1, graph.outputs[0])
    graph.add_edge(v2, graph.outputs[1])
    
    # Print the graph structure
    print(graph)
    
    # Evaluate the balancer
    evaluation = graph.evaluate()
    print(evaluation)
    
    # Check if the balancer is solved (properly balanced)
    print(f"Is this balancer solved? {graph.is_solved()}")



if __name__ == "__main__":
    main()


