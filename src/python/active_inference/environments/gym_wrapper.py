"""
Gymnasium environment wrapper for active inference agents.

This module provides a wrapper that adapts Gymnasium environments
for use with active inference agents.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    gym = None


class GymWrapper:
    """
    Wrapper for Gymnasium environments to work with active inference agents.
    
    This wrapper handles observation and action space conversion, and provides
    additional functionality like uncertainty injection and belief-based rewards.
    """
    
    def __init__(self,
                 env,
                 add_observation_noise: bool = False,
                 observation_noise_std: float = 0.1,
                 normalize_observations: bool = True,
                 belief_based_reward: bool = False,
                 uncertainty_penalty: float = 0.1):
        """
        Initialize Gym wrapper.
        
        Args:
            env: Gymnasium environment or environment ID string
            add_observation_noise: Whether to add noise to observations
            observation_noise_std: Standard deviation of observation noise
            normalize_observations: Whether to normalize observations
            belief_based_reward: Whether to include belief-based reward terms
            uncertainty_penalty: Penalty for high belief uncertainty
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium is required for GymWrapper. Install with: pip install gymnasium")
        
        # Create environment if string provided
        if isinstance(env, str):
            self.env = gym.make(env)
            self.env_id = env
        else:
            self.env = env
            self.env_id = getattr(env, 'spec', {}).get('id', 'unknown')
        
        self.add_observation_noise = add_observation_noise
        self.observation_noise_std = observation_noise_std
        self.normalize_observations = normalize_observations
        self.belief_based_reward = belief_based_reward
        self.uncertainty_penalty = uncertainty_penalty
        
        # Extract space information
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Compute dimensions
        self.obs_dim = self._get_space_dim(self.observation_space)
        self.action_dim = self._get_space_dim(self.action_space)
        
        # Observation normalization statistics
        self.obs_mean = None
        self.obs_std = None
        self.obs_history = []
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(f"GymWrapper.{self.env_id}")
        
        self.logger.info(f"Initialized wrapper for {self.env_id}")
        self.logger.info(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
    
    def _get_space_dim(self, space) -> int:
        """Get the dimensionality of a gym space."""
        if isinstance(space, gym.spaces.Discrete):
            return 1  # Discrete actions represented as 1D
        elif isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        elif isinstance(space, gym.spaces.MultiBinary):
            return space.n
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return len(space.nvec)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    
    def _flatten_observation(self, obs: Any) -> np.ndarray:
        """Convert observation to flat numpy array."""
        if isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float32)
        elif isinstance(obs, (int, float)):
            return np.array([obs], dtype=np.float32)
        elif isinstance(obs, (list, tuple)):
            return np.array(obs, dtype=np.float32).flatten()
        else:
            # Try to convert to numpy array
            return np.array(obs, dtype=np.float32).flatten()
    
    def _process_observation(self, obs: Any) -> np.ndarray:
        """Process raw observation into format suitable for active inference."""
        # Flatten observation
        processed_obs = self._flatten_observation(obs)
        
        # Add noise if requested
        if self.add_observation_noise:
            noise = np.random.normal(0, self.observation_noise_std, processed_obs.shape)
            processed_obs += noise
        
        # Normalize if requested
        if self.normalize_observations:
            processed_obs = self._normalize_observation(processed_obs)
        
        # Record for statistics
        self.obs_history.append(processed_obs.copy())
        if len(self.obs_history) > 1000:  # Keep last 1000 observations
            self.obs_history.pop(0)
        
        return processed_obs
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        if self.obs_mean is None or self.obs_std is None:
            # Initialize with current observation
            self.obs_mean = obs.copy()
            self.obs_std = np.ones_like(obs)
            return obs
        
        # Update running statistics (exponential moving average)
        alpha = 0.01  # Learning rate for running statistics
        self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs
        self.obs_std = (1 - alpha) * self.obs_std + alpha * np.abs(obs - self.obs_mean)
        
        # Normalize
        normalized_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return normalized_obs
    
    def _convert_action(self, action: np.ndarray) -> Any:
        """Convert action from agent format to environment format."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Convert continuous action to discrete
            if action.size == 1:
                # Single continuous value -> discrete choice
                action_idx = int(np.clip(action[0] * self.action_space.n, 0, self.action_space.n - 1))
                return action_idx
            else:
                # Multi-dimensional action -> select max index
                return int(np.argmax(action))
        
        elif isinstance(self.action_space, gym.spaces.Box):
            # Clip action to valid range
            action = np.clip(action, self.action_space.low, self.action_space.high)
            
            # Reshape if needed
            if self.action_space.shape:
                try:
                    action = action.reshape(self.action_space.shape)
                except ValueError:
                    # If reshape fails, pad or truncate
                    target_size = np.prod(self.action_space.shape)
                    if action.size < target_size:
                        # Pad with zeros
                        padded_action = np.zeros(target_size)
                        padded_action[:action.size] = action.flatten()
                        action = padded_action.reshape(self.action_space.shape)
                    else:
                        # Truncate
                        action = action.flatten()[:target_size].reshape(self.action_space.shape)
            
            return action.astype(self.action_space.dtype)
        
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            # Convert to binary
            binary_action = (action > 0).astype(int)
            return binary_action[:self.action_space.n]
        
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            # Convert to discrete choices
            discrete_action = []
            for i, n in enumerate(self.action_space.nvec):
                if i < len(action):
                    choice = int(np.clip(action[i] * n, 0, n - 1))
                else:
                    choice = 0
                discrete_action.append(choice)
            return np.array(discrete_action)
        
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return initial observation."""
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_observation(obs)
        
        # Reset episode statistics
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0.0
        
        self.logger.debug(f"Environment reset for episode {self.episode_count}")
        
        return processed_obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action from active inference agent
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to environment format
        env_action = self._convert_action(action)
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        
        # Process observation
        processed_obs = self._process_observation(obs)
        
        # Modify reward if belief-based rewards enabled
        if self.belief_based_reward:
            # This would integrate with agent beliefs to modify reward
            # For now, just return original reward
            pass
        
        # Update statistics
        self.step_count += 1
        self.total_reward += reward
        
        # Add wrapper info
        info['wrapper_stats'] = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'original_reward': reward,
            'processed_obs_shape': processed_obs.shape,
            'original_action': action.copy(),
            'env_action': env_action
        }
        
        return processed_obs, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment wrapper statistics."""
        stats = {
            'env_id': self.env_id,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'observation_space': str(self.observation_space),
            'action_space': str(self.action_space),
            'add_observation_noise': self.add_observation_noise,
            'normalize_observations': self.normalize_observations,
        }
        
        if self.obs_history:
            recent_obs = np.array(self.obs_history[-100:])  # Last 100 observations
            stats.update({
                'obs_mean': recent_obs.mean(axis=0).tolist(),
                'obs_std': recent_obs.std(axis=0).tolist(),
                'obs_min': recent_obs.min(axis=0).tolist(),
                'obs_max': recent_obs.max(axis=0).tolist(),
            })
        
        return stats
    
    @classmethod
    def from_env_id(cls, env_id: str, **kwargs) -> 'GymWrapper':
        """Create wrapper from environment ID."""
        return cls(env_id, **kwargs)
    
    def __repr__(self) -> str:
        """String representation of wrapper."""
        return (f"GymWrapper(env_id={self.env_id}, "
                f"obs_dim={self.obs_dim}, action_dim={self.action_dim})")