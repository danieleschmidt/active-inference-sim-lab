"""
Belief updater interface and implementations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from .variational import VariationalInference


class BeliefUpdater(ABC):
    """
    Abstract base class for belief updating methods.
    """
    
    @abstractmethod
    def update(self,
               prior_beliefs: BeliefState,
               observations: np.ndarray,
               model: GenerativeModel) -> BeliefState:
        """
        Update beliefs given observations.
        
        Args:
            prior_beliefs: Prior belief state
            observations: New observations
            model: Generative model
            
        Returns:
            Updated belief state
        """
        pass


class VariationalBeliefUpdater(BeliefUpdater):
    """Variational inference belief updater."""
    
    def __init__(self, **kwargs):
        self.inference_engine = VariationalInference(**kwargs)
    
    def update(self,
               prior_beliefs: BeliefState,
               observations: np.ndarray,
               model: GenerativeModel) -> BeliefState:
        """Update beliefs using variational inference."""
        return self.inference_engine.update_beliefs(
            observations, prior_beliefs, model
        )