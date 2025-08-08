"""
Inference engines for belief updating in active inference.
"""

from .variational import VariationalInference
from .belief_updater import (
    BeliefUpdater, 
    VariationalBeliefUpdater,
    KalmanBeliefUpdater, 
    ParticleBeliefUpdater
)

__all__ = [
    "VariationalInference",
    "BeliefUpdater",
    "VariationalBeliefUpdater",
    "KalmanBeliefUpdater",
    "ParticleBeliefUpdater",
]