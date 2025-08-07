"""
Active Inference Agent implementation.

This module implements the main ActiveInferenceAgent class that coordinates
perception, planning, and action based on the Free Energy Principle.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .beliefs import Belief, BeliefState
from .generative_model import GenerativeModel
from .free_energy import FreeEnergyObjective
from ..inference import VariationalInference
from ..planning import ActivePlanner


class ActiveInferenceAgent:
    """
    Active Inference Agent implementing the Free Energy Principle.
    
    The agent maintains beliefs about hidden states, updates these beliefs
    based on observations (perception), and selects actions that minimize
    expected free energy (active inference).
    """
    
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 action_dim: int,
                 inference_method: str = "variational",
                 planning_horizon: int = 5,
                 learning_rate: float = 0.01,
                 temperature: float = 1.0,
                 agent_id: str = "agent_0"):
        """
        Initialize Active Inference Agent.
        
        Args:
            state_dim: Dimensionality of hidden state space
            obs_dim: Dimensionality of observation space
            action_dim: Dimensionality of action space
            inference_method: Method for belief updating ("variational", "particle", "kalman")
            planning_horizon: Number of steps to plan ahead
            learning_rate: Learning rate for model updates
            temperature: Temperature for action selection
            agent_id: Unique identifier for this agent
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.inference_method = inference_method
        self.planning_horizon = planning_horizon
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.agent_id = agent_id
        
        # Core components
        self.generative_model = GenerativeModel(state_dim, obs_dim, action_dim)
        self.free_energy_objective = FreeEnergyObjective(temperature=temperature)
        self.beliefs = BeliefState()
        
        # Initialize inference engine
        self.inference_engine = VariationalInference(
            learning_rate=learning_rate,
            max_iterations=10
        )
        
        # Initialize planning
        self.planner = ActivePlanner(
            horizon=planning_horizon,
            temperature=temperature
        )
        
        # Agent history and statistics
        self.history = {
            'observations': [],
            'actions': [],
            'beliefs': [],
            'free_energy': [],
            'rewards': []
        }
        
        # Performance tracking
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Initialize beliefs with priors
        self._initialize_beliefs()
        
        # Setup logging
        self.logger = logging.getLogger(f"ActiveInferenceAgent.{agent_id}")
        
    def _initialize_beliefs(self):
        """Initialize agent beliefs from model priors."""
        for name, prior in self.generative_model.get_all_priors().items():
            self.beliefs.add_belief(name, Belief(
                mean=prior.mean.copy(),
                variance=prior.variance.copy(),
                support=prior.support
            ))
    
    def infer_states(self, observation: np.ndarray) -> BeliefState:
        """
        Infer hidden states from observations (perception).
        
        Args:
            observation: Current observation
            
        Returns:
            Updated belief state
        """
        # Use inference engine to update beliefs
        updated_beliefs = self.inference_engine.update_beliefs(
            observations=observation,
            prior_beliefs=self.beliefs,
            generative_model=self.generative_model
        )
        
        # Update agent beliefs
        self.beliefs = updated_beliefs
        
        # Save snapshot for history
        self.beliefs.save_snapshot()
        
        return self.beliefs
    
    def plan_action(self, 
                   beliefs: Optional[BeliefState] = None,
                   horizon: Optional[int] = None) -> np.ndarray:
        """
        Plan optimal action using active inference.
        
        Selects actions that minimize expected free energy over the planning horizon.
        
        Args:
            beliefs: Current beliefs (uses agent beliefs if None)
            horizon: Planning horizon (uses default if None)
            
        Returns:
            Optimal action
        """
        if beliefs is None:
            beliefs = self.beliefs
        
        if horizon is None:
            horizon = self.planning_horizon
        
        # Use planner to find optimal action
        optimal_action = self.planner.plan(
            beliefs=beliefs,
            generative_model=self.generative_model,
            free_energy_objective=self.free_energy_objective,
            horizon=horizon
        )
        
        return optimal_action
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Full perception-action cycle.
        
        1. Update beliefs based on observation (perception)
        2. Plan optimal action (active inference)
        3. Return action
        
        Args:
            observation: Current observation
            
        Returns:
            Selected action
        """
        # Perception: Update beliefs
        updated_beliefs = self.infer_states(observation)
        
        # Action: Plan based on updated beliefs
        action = self.plan_action(updated_beliefs)
        
        # Record in history
        self.history['observations'].append(observation.copy())
        self.history['actions'].append(action.copy())
        self.history['beliefs'].append(updated_beliefs.get_all_beliefs())
        
        # Compute and record free energy
        free_energy = self.free_energy_objective.compute_free_energy(
            observations=observation,
            beliefs=updated_beliefs,
            priors=self.generative_model.get_all_priors(),
            likelihood_fn=self.generative_model.likelihood
        )
        self.history['free_energy'].append(free_energy)
        
        self.step_count += 1
        
        return action
    
    def update_model(self, 
                    observation: np.ndarray, 
                    action: np.ndarray,
                    reward: Optional[float] = None) -> None:
        """
        Update generative model based on experience.
        
        Args:
            observation: Observed outcome
            action: Action that was taken
            reward: Optional reward signal
        """
        # Record reward
        if reward is not None:
            self.history['rewards'].append(reward)
            self.total_reward += reward
        
        # Model learning could be implemented here
        # For now, we'll implement a simple parameter update
        # In a full implementation, this would update the generative model
        # parameters based on prediction error
        
        pass  # Placeholder for model learning
    
    def reset(self, observation: Optional[np.ndarray] = None) -> None:
        """
        Reset agent for new episode.
        
        Args:
            observation: Initial observation (optional)
        """
        # Reset beliefs to priors
        self._initialize_beliefs()
        
        # Reset episode-specific counters
        self.step_count = 0
        self.episode_count += 1
        self.total_reward = 0.0
        
        # Process initial observation if provided
        if observation is not None:
            self.infer_states(observation)
        
        self.logger.info(f"Agent reset for episode {self.episode_count}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        if not self.history['free_energy']:
            return {}
        
        recent_fe = [fe.total for fe in self.history['free_energy'][-100:]]
        
        stats = {
            'agent_id': self.agent_id,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, len(self.history['rewards'])),
            'current_free_energy': self.history['free_energy'][-1].total if self.history['free_energy'] else 0,
            'average_free_energy': np.mean(recent_fe) if recent_fe else 0,
            'belief_confidence': self.beliefs.average_confidence(),
            'belief_entropy': self.beliefs.total_entropy(),
        }
        
        return stats
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent state to checkpoint file."""
        checkpoint = {
            'config': {
                'state_dim': self.state_dim,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'inference_method': self.inference_method,
                'planning_horizon': self.planning_horizon,
                'learning_rate': self.learning_rate,
                'temperature': self.temperature,
                'agent_id': self.agent_id,
            },
            'statistics': self.get_statistics(),
            'model': self.generative_model.get_model_parameters(),
            'history_length': len(self.history['observations'])
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"Agent checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'ActiveInferenceAgent':
        """Load agent from checkpoint file."""
        import json
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        config = checkpoint['config']
        agent = cls(**config)
        
        # Restore model if available
        if 'model' in checkpoint:
            # This would restore the generative model parameters
            # Implementation depends on model serialization format
            pass
        
        return agent
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return (f"ActiveInferenceAgent(id={self.agent_id}, "
                f"dims=[{self.state_dim}, {self.obs_dim}, {self.action_dim}], "
                f"episodes={self.episode_count}, steps={self.step_count})")