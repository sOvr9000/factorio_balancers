
"""
Example of finding bottlenecks in a balancer design.

This example demonstrates how to create a deliberately imperfect balancer
and analyze its bottlenecks and flow distribution.
"""

from factorio_balancers import BeltGraph



def main():
    """Find bottlenecks in a balancer design."""
    print("=== Finding bottlenecks in a balancer design ===")
    
    # Create an imperfect balancer
    graph = BeltGraph()
    graph.set_num_inputs(4)
    graph.set_num_outputs(4)
    
    # Create internal vertices
    v1 = graph.new_vertex()
    v2 = graph.new_vertex()
    
    # Connect inputs to central vertices
    graph.add_edge(graph.inputs[0], v1)
    graph.add_edge(graph.inputs[1], v1)
    graph.add_edge(graph.inputs[2], v2)
    graph.add_edge(graph.inputs[3], v2)
    
    # Connect central vertices to outputs
    graph.add_edge(v1, graph.outputs[0])
    graph.add_edge(v1, graph.outputs[1])
    graph.add_edge(v2, graph.outputs[2])
    graph.add_edge(v2, graph.outputs[3])
    
    # Evaluate the balancer
    evaluation = graph.evaluate()
    
    print(f"Balancer type: {graph.balancer_type}")
    print(f"Accuracy: {evaluation['accuracy']}")
    print(f"Bottlenecks: {evaluation['bottlenecks']}")
    
    # Show output flow
    print("Output flow distribution:")
    for output, flow in evaluation['output_flow'].items():
        print(f"  Output {output}: {flow}")



if __name__ == "__main__":
    main()


