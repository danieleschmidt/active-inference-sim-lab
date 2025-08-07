"""
Generative model implementation for active inference.

This module implements the generative model that defines an agent's beliefs
about how observations are generated from hidden states and actions.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import json

from .beliefs import Belief, BeliefState


class GenerativeModel:
    """
    Generative model defining the agent's beliefs about the world.
    
    A generative model specifies:
    - Prior beliefs p(s) about hidden states
    - Likelihood p(o|s) of observations given states
    - Transition dynamics p(s'|s,a) of how states evolve with actions
    """
    
    def __init__(self, 
                 state_dim: int,
                 obs_dim: int,
                 action_dim: int,
                 model_name: str = "default"):
        """
        Initialize generative model.
        
        Args:
            state_dim: Dimensionality of hidden state space
            obs_dim: Dimensionality of observation space
            action_dim: Dimensionality of action space
            model_name: Identifier for this model
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_name = model_name
        
        # Model components
        self._priors: Dict[str, Belief] = {}
        self._likelihood_fn: Optional[Callable] = None
        self._dynamics_fn: Optional[Callable] = None
        self._observation_noise: float = 0.1
        self._process_noise: float = 0.1
        
        # Initialize with simple default models
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with basic Gaussian priors and linear models."""
        # Default prior: zero-centered Gaussian
        self.add_prior(
            "state",
            mean=np.zeros(self.state_dim),
            variance=np.ones(self.state_dim)
        )
        
        # Default likelihood: linear observation model
        self._observation_matrix = np.random.randn(self.obs_dim, self.state_dim) * 0.1
        self.set_likelihood_function(self._default_likelihood)
        
        # Default dynamics: linear dynamics with action influence
        self._dynamics_matrix = np.eye(self.state_dim)
        self._action_matrix = np.random.randn(self.state_dim, self.action_dim) * 0.1
        self.set_dynamics_function(self._default_dynamics)
    
    def add_prior(self, 
                  name: str,
                  mean: np.ndarray,
                  variance: np.ndarray,
                  support: Optional[Tuple] = None) -> None:
        """
        Add a prior belief about a hidden state variable.
        
        Args:
            name: Name of the state variable
            mean: Prior mean
            variance: Prior variance
            support: Valid range for the variable (optional)
        """
        self._priors[name] = Belief(mean, variance, support)
    
    def get_prior(self, name: str) -> Optional[Belief]:
        """Get a specific prior by name."""
        return self._priors.get(name)
    
    def get_all_priors(self) -> Dict[str, Belief]:
        """Get all priors."""
        return self._priors.copy()
    
    def set_likelihood_function(self, likelihood_fn: Callable) -> None:
        """
        Set the observation likelihood function p(o|s).
        
        Args:
            likelihood_fn: Function that takes (state, observation) and returns likelihood
        """
        self._likelihood_fn = likelihood_fn
    
    def set_dynamics_function(self, dynamics_fn: Callable) -> None:
        """
        Set the state transition dynamics p(s'|s,a).
        
        Args:
            dynamics_fn: Function that takes (state, action) and returns next state
        """
        self._dynamics_fn = dynamics_fn
    
    def _default_likelihood(self, state: np.ndarray, observation: np.ndarray) -> float:
        """Default linear-Gaussian observation model."""
        predicted_obs = self._observation_matrix @ state
        error = observation - predicted_obs
        likelihood = np.exp(-0.5 * np.sum(error**2) / self._observation_noise**2)
        return likelihood
    
    def _default_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Default linear dynamics with additive action effects."""
        next_state = (self._dynamics_matrix @ state + 
                     self._action_matrix @ action +
                     np.random.randn(self.state_dim) * self._process_noise)
        return next_state
    
    def likelihood(self, state: np.ndarray, observation: np.ndarray) -> float:
        """
        Compute observation likelihood p(o|s).
        
        Args:
            state: Hidden state
            observation: Observed data
            
        Returns:
            Likelihood of observation given state
        """
        if self._likelihood_fn is None:
            raise ValueError("Likelihood function not set")
        return self._likelihood_fn(state, observation)
    
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current hidden state
            action: Action taken
            
        Returns:
            Predicted next state
        """
        if self._dynamics_fn is None:
            raise ValueError("Dynamics function not set")
        return self._dynamics_fn(state, action)
    
    def sample_prior(self, name: str = "state", n_samples: int = 1) -> np.ndarray:
        """Sample from prior beliefs."""
        if name not in self._priors:
            raise ValueError(f"No prior defined for '{name}'")
        return self._priors[name].sample(n_samples)
    
    def set_observation_noise(self, noise: float) -> None:
        """Set observation noise level."""
        self._observation_noise = max(1e-6, noise)
    
    def set_process_noise(self, noise: float) -> None:
        """Set process noise level."""  
        self._process_noise = max(1e-6, noise)
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            'state_dim': self.state_dim,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'model_name': self.model_name,
            'observation_noise': self._observation_noise,
            'process_noise': self._process_noise,
            'priors': {
                name: {
                    'mean': prior.mean.tolist(),
                    'variance': prior.variance.tolist(),
                    'support': prior.support
                }
                for name, prior in self._priors.items()
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_model_parameters(), f, indent=2)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GenerativeModel':
        """Load model from file."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        model = cls(
            state_dim=params['state_dim'],
            obs_dim=params['obs_dim'], 
            action_dim=params['action_dim'],
            model_name=params.get('model_name', 'loaded')
        )
        
        model.set_observation_noise(params.get('observation_noise', 0.1))
        model.set_process_noise(params.get('process_noise', 0.1))
        
        # Restore priors
        for name, prior_data in params.get('priors', {}).items():
            model.add_prior(
                name,
                np.array(prior_data['mean']),
                np.array(prior_data['variance']),
                prior_data.get('support')
            )
        
        return model