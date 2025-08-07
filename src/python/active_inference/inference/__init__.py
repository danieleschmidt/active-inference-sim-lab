"""
Inference engines for belief updating in active inference.
"""

from .variational import VariationalInference
from .belief_updater import BeliefUpdater

__all__ = [
    "VariationalInference",
    "BeliefUpdater",
]