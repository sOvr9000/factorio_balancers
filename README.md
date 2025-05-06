# Factorio Belt Balancer Library

A Python library for analyzing, creating, and manipulating belt balancers in the video game Factorio.

## Description

This library provides tools to work with belt balancers from Factorio, allowing you to:

- Convert between Factorio blueprint strings and graph representations
- Create and modify balancer structures
- Evaluate balancer output flows from inputs
- Reference common balancers for which optimal graphs are known, such as 3-2, 4-4, etc.
- Save and load balancer graphs as compact strings

The core of the library is the `BeltGraph` class, which represents a belt balancer as a directed graph with designated inputs and outputs.  Specifically, all vertices of the graph are restricted to have no more than two inbound edges and no more than two outbound edges so that each vertex models a splitter in Factorio, and each edge models a path of belts from one splitter to another.

## Features

- **Blueprint Conversion**: Load Factorio blueprint strings and convert them to graph representations
- **Balancer Analysis**: Evaluate how much of each input is sent to each output, displayed as fractions
- **Bottleneck Detection**: Identify throughput bottlenecks in balancer designs
- **Graph Manipulation**: Create, modify, and transform balancer graphs
- **Serialization**: Save and load balancers in "compact" string format (shorter than blueprint string format)
- **Common Balancers**: Predefined collection of proven-optimal balancer graphs (from 1-1 through 9-9, and more**)
  - The graphs were extracted from [Raynquist's belt balancer book](https://docs.google.com/spreadsheets/d/1_997994858885285/edit#gid=0)

## Installation

```bash
pip install factorio_balancers
```

## Dependencies

- `graph_tools`: For graph data structures and algorithms
- `sympy`: For mathematical operations
- `factoriobps`: For working with Factorio blueprint strings

## Examples

### Creating a Simple Balancer (2 input belts, 2 output belts)

```python
from factorio_balancers import BeltGraph

# Set up a 2-2 balancer
graph = BeltGraph()
graph.set_num_inputs(2)
graph.set_num_outputs(2)

# Get an ID for a new internal vertex (one splitter)
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
```

### Loading from a Blueprint

```python
from factorio_balancers import BeltGraph

# Load from a blueprint string
blueprint_string = '0eNql0+FqwyAQB/B3uc+mRBsb46uUMZLtGEJiRK+jIfjuMxmjpTiyZuAHFe7nXzln6PoLOm8sgZ7BvI02gD7PEMyHbftljyaHoMEQDsDAtsOywqvzGEIRXG+I0ENkYOw7XkHzyP5cTr61wY2eig57ukNEfGGAlgwZ/A60LqZXexm6dJzmWxYDN4ZUPtolRSILqZqDZDClKS+rg4xL0AdW7GCb8saKPHv8/eGy4A93zHPVU5xqNtLJf74lz7OnPay6sWWerZ+7vNpIqfakrB9SpmZdG1zffScGn+jDWiYUr+pG1PKURqVi/AIjDSPW'

graph = BeltGraph.from_blueprint_string(blueprint_string)
print(f"Balancer type: {graph.balancer_type}")

# Check if it's properly balanced
is_balanced = graph.is_solved()
print(f"Is balanced: {is_balanced}")
```

### Working with Common Balancers

```python
from factorio_balancers import BeltGraph, is_balancer_defined, common_balancers

# Check which balancers are available
balancer_types = list(common_balancers.keys())
print(f"Available balancer types: {balancer_types}")

# Load a common balancer
if is_balancer_defined("4-4"):
    graph = BeltGraph()
    graph.load_common_balancer("4-4")
    print(f"Loaded balancer: {graph.balancer_type}")
```

## Advanced Usage

See the `examples` folder for more detailed usage examples, including:

- Converting balancers (doubling, transposing)
- Saving and loading balancer designs
- Finding bottlenecks
- Creating custom balancers
- Analyzing blueprint books

## License

MIT

## Acknowledgements

- The Factorio community for blueprint designs
- Factorio development team (Wube Software) for an amazing game

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.