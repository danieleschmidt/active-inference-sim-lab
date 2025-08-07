"""
Belief representation and manipulation for active inference.

This module implements belief states and their operations, including
uncertainty quantification and belief updating mechanics.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class Belief:
    """
    Represents a belief about a hidden state variable.
    
    Attributes:
        mean: Expected value of the belief
        variance: Uncertainty in the belief  
        support: Valid range/domain for the belief
        precision: Inverse of variance (1/variance)
    """
    mean: np.ndarray
    variance: np.ndarray
    support: Optional[tuple] = None
    
    @property
    def precision(self) -> np.ndarray:
        """Precision (inverse variance) of the belief."""
        return 1.0 / (self.variance + 1e-8)  # Add small epsilon for numerical stability
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief (measure of uncertainty)."""
        return 0.5 * np.log(2 * np.pi * np.e * self.variance).sum()
    
    @property
    def confidence(self) -> float:
        """Confidence level (inverse of entropy)."""
        return 1.0 / (1.0 + self.entropy)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the belief distribution."""
        # Add regularization for numerical stability
        variance_matrix = np.diag(self.variance) + 1e-6 * np.eye(len(self.mean))
        
        try:
            return np.random.multivariate_normal(
                self.mean, 
                variance_matrix,
                size=n_samples
            )
        except np.linalg.LinAlgError:
            # Fallback: sample each dimension independently
            samples = np.random.normal(
                self.mean.reshape(1, -1),
                np.sqrt(self.variance.reshape(1, -1)),
                size=(n_samples, len(self.mean))
            )
            return samples if n_samples > 1 else samples[0]
    
    def log_probability(self, x: np.ndarray) -> float:
        """Compute log probability of observation under this belief."""
        diff = x - self.mean
        return -0.5 * (diff.T @ np.diag(self.precision) @ diff + 
                      np.log(2 * np.pi * self.variance).sum())


class BeliefState:
    """
    Container for multiple beliefs about different aspects of the world state.
    
    This class manages a collection of beliefs about different hidden state
    variables and provides methods for accessing and updating them.
    """
    
    def __init__(self):
        self._beliefs: Dict[str, Belief] = {}
        self._history: list = []
    
    def add_belief(self, name: str, belief: Belief) -> None:
        """Add a named belief to the state."""
        self._beliefs[name] = belief
    
    def get_belief(self, name: str) -> Optional[Belief]:
        """Get a specific belief by name."""
        return self._beliefs.get(name)
    
    def update_belief(self, name: str, new_mean: np.ndarray, 
                     new_variance: np.ndarray) -> None:
        """Update an existing belief with new parameters."""
        if name in self._beliefs:
            self._beliefs[name].mean = new_mean
            self._beliefs[name].variance = new_variance
        else:
            self._beliefs[name] = Belief(new_mean, new_variance)
    
    def get_all_beliefs(self) -> Dict[str, Belief]:
        """Get all beliefs in the state."""
        return self._beliefs.copy()
    
    def total_entropy(self) -> float:
        """Compute total entropy across all beliefs."""
        return sum(belief.entropy for belief in self._beliefs.values())
    
    def average_confidence(self) -> float:
        """Compute average confidence across all beliefs."""
        if not self._beliefs:
            return 0.0
        return np.mean([belief.confidence for belief in self._beliefs.values()])
    
    def save_snapshot(self) -> None:
        """Save current state to history."""
        snapshot = {
            name: {
                'mean': belief.mean.copy(),
                'variance': belief.variance.copy(),
                'entropy': belief.entropy
            }
            for name, belief in self._beliefs.items()
        }
        self._history.append(snapshot)
    
    def get_history(self) -> list:
        """Get belief evolution history."""
        return self._history.copy()
    
    def __len__(self) -> int:
        """Number of beliefs in the state."""
        return len(self._beliefs)
    
    def __contains__(self, name: str) -> bool:
        """Check if a belief exists."""
        return name in self._beliefs
    
    def __getitem__(self, name: str) -> Belief:
        """Get belief by name using bracket notation."""
        return self._beliefs[name]
    
    def __setitem__(self, name: str, belief: Belief) -> None:
        """Set belief by name using bracket notation."""
        self._beliefs[name] = belief