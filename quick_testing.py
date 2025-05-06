from factorio_balancers import BeltGraph

# Create a 2-2 balancer
graph = BeltGraph()
graph.set_num_inputs(2)
graph.set_num_outputs(2)

# Add an internal vertex
v = graph.new_vertex()

# Connect inputs to the internal vertex
graph.add_edge(graph.inputs[0], v)
graph.add_edge(graph.inputs[1], v)

# Connect the internal vertex to outputs
graph.add_edge(v, graph.outputs[0])
graph.add_edge(v, graph.outputs[1])

# Print the graph
print(graph)

# Evaluate the balancer
evaluation = graph.evaluate()
print(evaluation)