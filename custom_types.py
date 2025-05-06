
"""
Define custom types used throughout the files.
"""

from fractions import Fraction
from typing import Dict, Union

__all__ = [
    'Scalar',
    'FlowDict',
    'LinearCombination',
]



# Type aliases for clarity
FlowDict = Dict[int, Dict[int, float]]
Scalar = Union[float, Fraction]
LinearCombination = Dict[int, Scalar]


