"""
Abstract base class for all optimization solutions.

Every problem-specific solution inherits from this class and must implement
fitness(), random_initial_representation(), and with_repr().

The algorithms (GA, SA) depend only on this interface — never on
problem-specific details.  That separation is what makes them reusable.
"""

import numpy as np
from abc import ABC, abstractmethod


class Solution(ABC):

    def __init__(self, repr=None):
        self.repr = repr if repr is not None else self.random_initial_representation()

    @abstractmethod
    def random_initial_representation(self) -> list:
        """Return a random genome for this solution type."""

    @abstractmethod
    def fitness(self) -> float:
        """Return a scalar quality score (cached after first call)."""

    @abstractmethod
    def with_repr(self, new_repr) -> 'Solution':
        """Return a new instance with the given genome; never mutate self."""

    def distance_to(self, other) -> float:
        """Euclidean distance between self.repr and other.repr."""
        return float(np.linalg.norm(
            np.array(self.repr, dtype=np.float64) - np.array(other.repr, dtype=np.float64)
        ))
