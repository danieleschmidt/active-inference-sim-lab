"""
Active planning implementation for active inference agents.

This module implements the active planner that selects actions to minimize
expected free energy over a planning horizon.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..core.beliefs import BeliefState
from ..core.generative_model import GenerativeModel
from ..core.free_energy import FreeEnergyObjective
from .expected_free_energy import ExpectedFreeEnergy


class ActivePlanner:
    """
    Active planner for selecting actions that minimize expected free energy.
    
    The planner evaluates candidate actions over a planning horizon and
    selects the action that minimizes expected free energy.
    """
    
    def __init__(self,
                 horizon: int = 5,
                 n_action_samples: int = 10,
                 temperature: float = 1.0,
                 planning_method: str = "sampling"):
        """
        Initialize active planner.
        
        Args:
            horizon: Planning horizon (number of steps to look ahead)
            n_action_samples: Number of action candidates to evaluate
            temperature: Temperature for action selection (0 = greedy)
            planning_method: Method for action generation ("sampling", "grid", "optimization")
        """
        self.horizon = horizon
        self.n_action_samples = n_action_samples
        self.temperature = temperature
        self.planning_method = planning_method
        
        # Initialize expected free energy calculator
        self.efe_calculator = ExpectedFreeEnergy()
        
        # Planning statistics
        self.planning_history = []
        
        # Setup logging
        self.logger = logging.getLogger("ActivePlanner")
    
    def plan(self,
             beliefs: BeliefState,
             generative_model: GenerativeModel,
             free_energy_objective: FreeEnergyObjective,
             horizon: Optional[int] = None) -> np.ndarray:
        """
        Plan optimal action using active inference.
        
        Args:
            beliefs: Current belief state
            generative_model: Agent's generative model
            free_energy_objective: Free energy objective function
            horizon: Planning horizon (uses default if None)
            
        Returns:
            Optimal action
        """
        if horizon is None:
            horizon = self.horizon
        
        # Generate candidate actions
        candidate_actions = self._generate_candidate_actions(
            generative_model.action_dim
        )
        
        # Evaluate actions using expected free energy
        optimal_action, efe_components = self.efe_calculator.select_optimal_action(
            actions=candidate_actions,
            beliefs=beliefs,
            model=generative_model,
            horizon=horizon,
            temperature=self.temperature
        )
        
        # Record planning statistics
        planning_stats = {
            'n_candidates': len(candidate_actions),
            'selected_efe': efe_components.total,
            'epistemic_value': efe_components.epistemic_value,
            'pragmatic_value': efe_components.pragmatic_value,
            'horizon': horizon
        }
        self.planning_history.append(planning_stats)
        
        self.logger.debug(f"Selected action with EFE: {efe_components.total:.4f}")
        
        return optimal_action
    
    def _generate_candidate_actions(self, action_dim: int) -> List[np.ndarray]:
        """
        Generate candidate actions for evaluation.
        
        Args:
            action_dim: Dimensionality of action space
            
        Returns:
            List of candidate actions
        """
        if self.planning_method == "sampling":
            return self._sample_actions(action_dim)
        elif self.planning_method == "grid":
            return self._grid_actions(action_dim)
        elif self.planning_method == "optimization":
            return self._optimize_actions(action_dim)
        else:
            raise ValueError(f"Unknown planning method: {self.planning_method}")
    
    def _sample_actions(self, action_dim: int) -> List[np.ndarray]:
        """Sample random actions from action space."""
        actions = []
        
        for _ in range(self.n_action_samples):
            # Sample from standard normal distribution
            # In practice, this should be adapted to the specific action space
            action = np.random.randn(action_dim)
            # Clip to reasonable range
            action = np.clip(action, -2.0, 2.0)
            actions.append(action)
        
        return actions
    
    def _grid_actions(self, action_dim: int) -> List[np.ndarray]:
        """Generate actions on a grid."""
        actions = []
        
        if action_dim == 1:
            # 1D grid
            for x in np.linspace(-2, 2, self.n_action_samples):
                actions.append(np.array([x]))
        elif action_dim == 2:
            # 2D grid
            n_per_dim = int(np.sqrt(self.n_action_samples))
            for x in np.linspace(-2, 2, n_per_dim):
                for y in np.linspace(-2, 2, n_per_dim):
                    actions.append(np.array([x, y]))
                    if len(actions) >= self.n_action_samples:
                        break
                if len(actions) >= self.n_action_samples:
                    break
        else:
            # High-dimensional: fall back to sampling
            return self._sample_actions(action_dim)
        
        return actions[:self.n_action_samples]
    
    def _optimize_actions(self, action_dim: int) -> List[np.ndarray]:
        """Generate actions using optimization (placeholder)."""
        # This would implement gradient-based action optimization
        # For now, fall back to sampling
        return self._sample_actions(action_dim)
    
    def set_preferences(self, preferences: np.ndarray) -> None:
        """
        Set agent preferences for pragmatic value computation.
        
        Args:
            preferences: Preferred observation distribution
        """
        self.efe_calculator.set_preferences(preferences)
    
    def set_epistemic_weight(self, weight: float) -> None:
        """Set weight for epistemic (information-seeking) behavior."""
        self.efe_calculator.epistemic_weight = weight
    
    def set_pragmatic_weight(self, weight: float) -> None:
        """Set weight for pragmatic (goal-directed) behavior."""
        self.efe_calculator.pragmatic_weight = weight
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get planning performance statistics."""
        if not self.planning_history:
            return {}
        
        recent_history = self.planning_history[-100:]  # Last 100 planning steps
        
        stats = {
            'total_planning_steps': len(self.planning_history),
            'average_efe': np.mean([h['selected_efe'] for h in recent_history]),
            'average_epistemic_value': np.mean([h['epistemic_value'] for h in recent_history]),
            'average_pragmatic_value': np.mean([h['pragmatic_value'] for h in recent_history]),
            'average_candidates_evaluated': np.mean([h['n_candidates'] for h in recent_history]),
            'planning_method': self.planning_method,
            'current_temperature': self.temperature,
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset planning statistics."""
        self.planning_history = []
        self.logger.info("Planning statistics reset")
    
    def update_planning_parameters(self, **kwargs) -> None:
        """
        Update planning parameters dynamically.
        
        Args:
            **kwargs: Parameters to update (horizon, temperature, etc.)
        """
        if 'horizon' in kwargs:
            self.horizon = kwargs['horizon']
        
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        
        if 'n_action_samples' in kwargs:
            self.n_action_samples = kwargs['n_action_samples']
        
        if 'planning_method' in kwargs:
            self.planning_method = kwargs['planning_method']
        
        self.logger.info(f"Updated planning parameters: {kwargs}")
    
    def __repr__(self) -> str:
        """String representation of planner."""
        return (f"ActivePlanner(horizon={self.horizon}, "
                f"n_samples={self.n_action_samples}, "
                f"temperature={self.temperature}, "
                f"method={self.planning_method})")