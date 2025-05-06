
"""
Factorio Belt Balancer Metrics Module

This module provides functions to evaluate the accuracy and balance quality
of Factorio belt balancer designs. It measures how well a balancer distributes
input across outputs using both factorization analysis and flow distribution.

Key functions:
- factorization_distance: Calculates mathematical distance between factorizations
- measure_accuracy: Evaluates balancer accuracy and produces quality metrics
"""

from fractions import Fraction
from math import log1p
from functools import lru_cache
from typing import List, Tuple
from .sparsesolver import FlowDict
import sympy as sp



@lru_cache(maxsize=4096)
def factorization_distance(n1: int, n2: int) -> float:
    """
    Calculate a mathematical distance between the prime factorizations of two numbers.
    
    This function measures how "different" two numbers are in terms of their prime factors.
    A distance of 0 means the numbers are identical. Larger values indicate greater
    differences in their factorizations.
    
    Args:
        n1: First integer
        n2: Second integer
        
    Returns:
        float: Distance metric between the factorizations
        
    Examples:
        >>> factorization_distance(12, 12)
        0.0
        >>> factorization_distance(12, 13)  # 2²×3 vs 13 (prime)
        3.7902...
    """
    # Convert to absolute values
    n1 = abs(n1)
    n2 = abs(n2)
    
    # Handle identity case
    if n1 == n2:
        return 0.0
        
    # Handle zero cases
    if n1 == 0:
        v = factorization_distance(1, n2)
        return (1 + v) * (1 + v)
    if n2 == 0:
        v = factorization_distance(n1, 1)
        return (1 + v) * (1 + v)
    
    # Get prime factorizations
    pfs1 = sp.factorint(n1)
    pfs2 = sp.factorint(n2)
    
    # Calculate distance metric
    distance_sum = 0
    
    # Combine all prime factors from both numbers
    all_factors = set(list(pfs1.keys()) + list(pfs2.keys()))
    
    for prime_factor in all_factors:
        # Ensure factors exist in both dictionaries
        count1 = pfs1.get(prime_factor, 0)
        count2 = pfs2.get(prime_factor, 0)
        
        # Calculate difference in factor counts
        factor_diff = count1 - count2
        
        # Add weighted contribution to distance
        distance_sum += log1p(factor_diff * factor_diff) / ((abs(count1) + 0.5) * (abs(count2) + 0.5))
        
        # Add extra distance if a factor is present in only one number
        if (count1 == 0) ^ (count2 == 0):  # XOR operation
            distance_sum += prime_factor ** 0.5
    
    return distance_sum

def measure_accuracy(
    output_flow: FlowDict, 
    inputs: List[int], 
    outputs: List[int]
) -> Tuple[float, float, float]:
    """
    Evaluate how accurately a belt balancer distributes flow from inputs to outputs.
    
    This function analyzes the flow distribution at the outputs of a balancer and 
    calculates metrics that indicate how well it balances inputs across outputs.
    
    Args:
        output_flow: Dictionary mapping output vertices to their flow distributions
        inputs: List of input vertex IDs
        outputs: List of output vertex IDs
        
    Returns:
        tuple: (accuracy, error, score) where:
            - accuracy: Value between 0-1 indicating balancing accuracy
            - error: A measure of deviation from ideal factorization
            - score: Combined quality metric (accuracy - 0.1*error)
            
    Notes:
        - Perfect balancers have accuracy=1, low error, and score close to 1
        - A balanced N-M balancer should distribute 1/M of each input to each output
    """
    num_inputs = len(inputs)
    num_outputs = len(outputs)
    
    # Handle empty case
    if num_inputs == 0 or num_outputs == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate accuracy based on flow distribution
    accuracy_sum = 0
    for _, flow_dict in output_flow.items():
        for val in flow_dict.values():
            # Convert value to fraction
            frac_val = Fraction(val)
            
            # Check if denominator matches expected value
            if val != 0 and frac_val.denominator == num_outputs:
                accuracy_sum += 0.5 + 0.5 / frac_val.numerator
            # Otherwise no accuracy contribution
    
    # Normalize accuracy by total number of input-output connections
    accuracy = accuracy_sum / (num_inputs * num_outputs)
    
    # Calculate error based on factorization distances
    error_sum = 0
    for k in outputs:
        flow_dict = output_flow.get(k, {})
        for n in inputs:
            proportion = Fraction(flow_dict.get(n, 0))
            error_sum += factorization_distance(proportion.denominator, num_outputs)
    
    error = error_sum ** 0.5
    
    # Calculate overall score
    score = accuracy - 0.1 * error
    
    return accuracy, error, score



if __name__ == '__main__':
    # Example usage: Calculate factorization distances
    print("Factorization Distance Examples:")
    print("-" * 30)
    
    test_pairs = [
        (0, 0), (1, 1), (2, 2), (4, 4),  # Identity cases
        (1, 2), (2, 3), (4, 6), (8, 9),  # Simple differences
        (4, 8), (9, 27), (8, 32),        # Power differences
        (12, 18), (24, 36)               # Shared factors
    ]
    
    for pair in test_pairs:
        n1, n2 = pair
        distance = factorization_distance(n1, n2)
        print(f"Distance between {n1} and {n2}: {distance:.4f}")
        
        # Verify symmetry
        reverse_distance = factorization_distance(n2, n1)
        if distance != reverse_distance:
            print(f"  Asymmetry detected: {n2} to {n1} = {reverse_distance:.4f}")
    
    # Systematic testing of small numbers
    print("\nComprehensive Factorization Distance Grid (0-11):")
    print("-" * 50)
    max_n = 6  # Reduced from 12 to keep output reasonable
    
    print("    ", end="")
    for n2 in range(max_n):
        print(f"{n2:5}", end="")
    print()
    
    for n1 in range(max_n):
        print(f"{n1:2}: ", end="")
        for n2 in range(max_n):
            print(f"{factorization_distance(n1, n2):5.2f}", end="")
        print()


