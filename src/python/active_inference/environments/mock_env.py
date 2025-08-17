"""
Mock environment for testing without gymnasium dependency.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class MockEnvironment:
    """
    Simple mock environment for testing active inference agents
    without requiring gymnasium.
    """
    
    def __init__(self, 
                 obs_dim: int = 4,
                 action_dim: int = 2,
                 episode_length: int = 100,
                 reward_noise: float = 0.0,
                 observation_noise: float = 0.01,
                 temporal_dynamics: bool = False,
                 **kwargs):
        """
        Initialize mock environment.
        
        Args:
            obs_dim: Observation dimensionality
            action_dim: Action dimensionality 
            episode_length: Maximum episode length
            reward_noise: Noise level for reward signal
            observation_noise: Noise level for observations
            temporal_dynamics: Enable temporal dynamics modeling
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.reward_noise = reward_noise
        self.observation_noise = observation_noise
        self.temporal_dynamics = temporal_dynamics
        
        self.state = None
        self.step_count = 0
        self.episode_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.step_count = 0
        self.episode_count += 1
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Simple dynamics: state evolves with action influence + noise
        if len(action) >= self.action_dim:
            action = action[:self.action_dim]
        else:
            # Pad action if too short
            padded_action = np.zeros(self.action_dim)
            padded_action[:len(action)] = action
            action = padded_action
        
        # Update state (pad action to match state dimensions if needed)
        action_effect = np.zeros(self.obs_dim)
        action_effect[:min(len(action), self.obs_dim)] = action[:min(len(action), self.obs_dim)]
        
        self.state += 0.1 * action_effect + 0.05 * np.random.randn(self.obs_dim)
        
        # Compute reward (negative quadratic cost) with optional noise
        reward = -np.sum(self.state**2) - 0.01 * np.sum(action**2)
        if self.reward_noise > 0:
            reward += np.random.normal(0, self.reward_noise)
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.episode_length
        
        # Info
        info = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'state_norm': np.linalg.norm(self.state)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (with configurable noise)."""
        obs = self.state.copy()
        # Add configurable observation noise
        if self.observation_noise > 0:
            obs += np.random.normal(0, self.observation_noise, self.obs_dim)
        return obs
    
    def render(self, mode: str = 'human') -> None:
        """Render environment (no-op for mock env)."""
        if mode == 'human':
            print(f"Step {self.step_count}: State = {self.state}")
    
    def close(self) -> None:
        """Close environment (no-op for mock env)."""
        pass