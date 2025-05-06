
"""
Example of loading a balancer from a Factorio blueprint string.

This example demonstrates how to load a balancer from a Factorio blueprint string
and analyze its properties.
"""

from factorio_balancers import BeltGraph



def main():
    """Load a balancer from a blueprint string."""
    print("=== Loading a balancer from a blueprint string ===")
    
    # Example 4-4 balancer blueprint
    blueprint_string = '0eNql0+FqwyAQB/B3uc+mRBsb46uUMZLtGEJiRK+jIfjuMxmjpTiyZuAHFe7nXzln6PoLOm8sgZ7BvI02gD7PEMyHbftljyaHoMEQDsDAtsOywqvzGEIRXG+I0ENkYOw7XkHzyP5cTr61wY2eig57ukNEfGGAlgwZ/A60LqZXexm6dJzmWxYDN4ZUPtolRSILqZqDZDClKS+rg4xL0AdW7GCb8saKPHv8/eGy4A93zHPVU5xqNtLJf74lz7OnPay6sWWerZ+7vNpIqfakrB9SpmZdG1zffScGn+jDWiYUr+pG1PKURqVi/AIjDSPW'
    
    # Create a new graph and load from the blueprint
    graph = BeltGraph.from_blueprint_string(blueprint_string)
    
    # Print the graph information
    print(f"Balancer type: {graph.balancer_type}")
    print(f"Number of internal vertices: {graph.num_internal_vertices}")
    print(f"Number of internal edges: {graph.num_internal_edges}")
    
    # Evaluate the balancer
    evaluation = graph.evaluate()
    print(f"Accuracy: {evaluation['accuracy']}")
    print(f"Is solved: {graph.is_solved(evaluation)}")



if __name__ == "__main__":
    main()


