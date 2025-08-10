"""
Active Inference Agent implementation.

This module implements the main ActiveInferenceAgent class that coordinates
perception, planning, and action based on the Free Energy Principle.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import traceback
import warnings
from pathlib import Path

from .beliefs import Belief, BeliefState
from .generative_model import GenerativeModel
from .free_energy import FreeEnergyObjective
from ..inference.variational import VariationalInference
from ..planning.active_planner import ActivePlanner
from ..utils.validation import (
    ValidationError, ActiveInferenceError, ModelError, InferenceError, PlanningError,
    validate_array, validate_dimensions, validate_inputs, handle_errors
)


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
                 agent_id: str = "agent_0",
                 enable_logging: bool = True,
                 log_level: int = logging.INFO,
                 max_history_length: int = 10000):
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
            enable_logging: Whether to enable detailed logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            max_history_length: Maximum length of agent history
            
        Raises:
            ValidationError: If input parameters are invalid
            ActiveInferenceError: If initialization fails
        """
        try:
            # Validate input parameters
            validate_dimensions(state_dim, obs_dim, action_dim)
            validate_inputs(
                learning_rate=learning_rate,
                temperature=temperature,
                planning_horizon=planning_horizon,
                max_history_length=max_history_length
            )
            
            if not isinstance(agent_id, str) or not agent_id.strip():
                raise ValidationError("agent_id must be a non-empty string")
            
            if inference_method not in ["variational", "particle", "kalman"]:
                raise ValidationError(f"Unsupported inference method: {inference_method}")
            
            if planning_horizon <= 0 or planning_horizon > 50:
                raise ValidationError(f"planning_horizon must be between 1 and 50, got {planning_horizon}")
            
            if learning_rate <= 0 or learning_rate > 1:
                raise ValidationError(f"learning_rate must be between 0 and 1, got {learning_rate}")
            
            if temperature <= 0 or temperature > 10:
                raise ValidationError(f"temperature must be between 0 and 10, got {temperature}")
            # Store validated parameters
            self.state_dim = state_dim
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.inference_method = inference_method
            self.planning_horizon = planning_horizon
            self.learning_rate = learning_rate
            self.temperature = temperature
            self.agent_id = agent_id.strip()
            self.enable_logging = enable_logging
            self.max_history_length = max_history_length
            
            # Initialize safety and monitoring
            self._is_initialized = False
            self._error_count = 0
            self._last_error = None
            self._health_status = "healthy"
        
            # Initialize core components with error handling
            try:
                self.generative_model = GenerativeModel(state_dim, obs_dim, action_dim)
                self.free_energy_objective = FreeEnergyObjective(temperature=temperature)
                self.beliefs = BeliefState()
            except Exception as e:
                raise ActiveInferenceError(f"Failed to initialize core components: {e}")
            
            # Initialize inference engine with error handling
            try:
                self.inference_engine = VariationalInference(
                    learning_rate=learning_rate,
                    max_iterations=10
                )
            except Exception as e:
                raise InferenceError(f"Failed to initialize inference engine: {e}")
            
            # Initialize planning with error handling
            try:
                self.planner = ActivePlanner(
                    horizon=planning_horizon,
                    temperature=temperature
                )
            except Exception as e:
                raise PlanningError(f"Failed to initialize planner: {e}")
        
            # Agent history and statistics
            self.history = {
                'observations': [],
                'actions': [],
                'beliefs': [],
                'free_energy': [],
                'rewards': [],
                'errors': []
            }
            
            # Performance tracking
            self.step_count = 0
            self.episode_count = 0
            self.total_reward = 0.0
            
            # Setup logging first
            self.logger = logging.getLogger(f"ActiveInferenceAgent.{self.agent_id}")
            if self.enable_logging:
                self.logger.setLevel(log_level)
                if not self.logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
            
            # Initialize beliefs with priors (with error handling)
            try:
                self._initialize_beliefs()
                self._is_initialized = True
                self.logger.info(f"Agent {self.agent_id} initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize beliefs: {e}")
                raise ActiveInferenceError(f"Agent initialization failed: {e}")
                
        except Exception as e:
            # Log initialization error and re-raise
            if hasattr(self, 'logger'):
                self.logger.error(f"Agent initialization failed: {e}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
        
    def _initialize_beliefs(self):
        """Initialize agent beliefs from model priors.
        
        Raises:
            ModelError: If priors are invalid
            ValidationError: If belief initialization fails
        """
        try:
            priors = self.generative_model.get_all_priors()
            if not priors:
                raise ModelError("No priors available from generative model")
            
            for name, prior in priors.items():
                if not hasattr(prior, 'mean') or not hasattr(prior, 'variance'):
                    raise ModelError(f"Prior '{name}' missing required attributes")
                
                validate_array(prior.mean, f"prior '{name}' mean")
                validate_array(prior.variance, f"prior '{name}' variance", min_val=0.0)
                
                belief = Belief(
                    mean=prior.mean.copy(),
                    variance=prior.variance.copy(),
                    support=getattr(prior, 'support', None)
                )
                self.beliefs.add_belief(name, belief)
                
            self.logger.debug(f"Initialized {len(priors)} belief components")
            
        except Exception as e:
            self._record_error("belief_initialization", e)
            raise ModelError(f"Failed to initialize beliefs: {e}")
    
    @handle_errors((InferenceError, ValidationError), log_errors=True)
    def infer_states(self, observation: np.ndarray) -> BeliefState:
        """
        Infer hidden states from observations (perception).
        
        Args:
            observation: Current observation
            
        Returns:
            Updated belief state
            
        Raises:
            ValidationError: If observation is invalid
            InferenceError: If belief update fails
        """
        if not self._is_initialized:
            raise ActiveInferenceError("Agent not properly initialized")
        
        try:
            # Validate observation
            validate_array(observation, "observation", expected_shape=(self.obs_dim,))
            
            # Check for reasonable observation values
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                raise ValidationError("Observation contains NaN or infinite values")
            
            # Check health status
            if self._health_status != "healthy":
                warnings.warn(f"Agent health status: {self._health_status}")
            # Use inference engine to update beliefs
            updated_beliefs = self.inference_engine.update_beliefs(
                observations=observation,
                prior_beliefs=self.beliefs,
                generative_model=self.generative_model
            )
            
            # Validate updated beliefs
            if not updated_beliefs or len(updated_beliefs.get_all_beliefs()) == 0:
                raise InferenceError("Belief update resulted in empty beliefs")
            
            # Update agent beliefs
            self.beliefs = updated_beliefs
            
            # Save snapshot for history
            self.beliefs.save_snapshot()
            
            self.logger.debug(f"Successfully updated beliefs from observation")
            return self.beliefs
            
        except (ValidationError, InferenceError):
            # Re-raise validation and inference errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("inference", e)
            raise InferenceError(f"Unexpected error during state inference: {e}")
    
    @handle_errors((PlanningError, ValidationError), log_errors=True)
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
            
        Raises:
            ValidationError: If inputs are invalid
            PlanningError: If planning fails
        """
        if not self._is_initialized:
            raise ActiveInferenceError("Agent not properly initialized")
        
        try:
            # Use current beliefs if none provided
            if beliefs is None:
                beliefs = self.beliefs
                if not beliefs or len(beliefs.get_all_beliefs()) == 0:
                    raise PlanningError("No beliefs available for planning")
            
            # Validate and set horizon
            if horizon is None:
                horizon = self.planning_horizon
            elif not isinstance(horizon, int) or horizon <= 0 or horizon > 50:
                raise ValidationError(f"Invalid planning horizon: {horizon}")
            
            # Check health status
            if self._health_status != "healthy":
                warnings.warn(f"Planning with degraded agent health: {self._health_status}")
            # Use planner to find optimal action
            optimal_action = self.planner.plan(
                beliefs=beliefs,
                generative_model=self.generative_model,
                free_energy_objective=self.free_energy_objective,
                horizon=horizon
            )
            
            # Validate planned action
            validate_array(optimal_action, "planned_action", expected_shape=(self.action_dim,))
            
            # Check for reasonable action values
            if np.any(np.isnan(optimal_action)) or np.any(np.isinf(optimal_action)):
                raise PlanningError("Planned action contains NaN or infinite values")
            
            self.logger.debug(f"Successfully planned action with horizon {horizon}")
            return optimal_action
            
        except (ValidationError, PlanningError):
            # Re-raise validation and planning errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("planning", e)
            raise PlanningError(f"Unexpected error during action planning: {e}")
    
    def _record_error(self, error_type: str, error: Exception) -> None:
        """Record error for monitoring and debugging.
        
        Args:
            error_type: Type/category of error
            error: The exception that occurred
        """
        self._error_count += 1
        self._last_error = {
            'type': error_type,
            'message': str(error),
            'timestamp': np.datetime64('now'),
            'step_count': self.step_count
        }
        
        # Add to history (limit length)
        self.history['errors'].append(self._last_error)
        if len(self.history['errors']) > 100:  # Keep last 100 errors
            self.history['errors'] = self.history['errors'][-100:]
        
        # Update health status based on error frequency
        recent_errors = [e for e in self.history['errors'] 
                        if self.step_count - e['step_count'] < 100]
        
        if len(recent_errors) > 20:
            self._health_status = "critical"
        elif len(recent_errors) > 10:
            self._health_status = "degraded"
        elif len(recent_errors) > 5:
            self._health_status = "warning"
        else:
            self._health_status = "healthy"
        
        self.logger.warning(f"Error recorded: {error_type} - {error}")
    
    def _trim_history(self) -> None:
        """Trim history to prevent memory bloat."""
        if len(self.history['observations']) > self.max_history_length:
            trim_size = self.max_history_length // 2
            for key in ['observations', 'actions', 'beliefs', 'free_energy', 'rewards']:
                self.history[key] = self.history[key][-trim_size:]
            self.logger.info(f"Trimmed history to {trim_size} entries")
    
    @handle_errors((ActiveInferenceError, ValidationError), log_errors=True)
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
            
        Raises:
            ValidationError: If observation is invalid
            ActiveInferenceError: If perception-action cycle fails
        """
        if not self._is_initialized:
            raise ActiveInferenceError("Agent not properly initialized")
        
        try:
            # Validate input
            validate_array(observation, "observation", expected_shape=(self.obs_dim,))
            # Perception: Update beliefs
            updated_beliefs = self.infer_states(observation)
            
            # Action: Plan based on updated beliefs
            action = self.plan_action(updated_beliefs)
            
            # Record in history (with error handling)
            try:
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
                
            except Exception as e:
                self.logger.warning(f"Failed to record history: {e}")
                # Continue execution even if history recording fails
            
            self.step_count += 1
            
            # Trim history periodically to prevent memory bloat
            if len(self.history['observations']) > self.max_history_length:
                self._trim_history()
            
            self.logger.debug(f"Completed perception-action cycle, step {self.step_count}")
            return action
            
        except (ValidationError, ActiveInferenceError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("perception_action", e)
            raise ActiveInferenceError(f"Perception-action cycle failed: {e}")
    
    @handle_errors((ModelError, ValidationError), log_errors=True)
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
            
        Raises:
            ValidationError: If inputs are invalid
            ModelError: If model update fails
        """
        try:
            # Validate inputs
            validate_array(observation, "observation", expected_shape=(self.obs_dim,))
            validate_array(action, "action", expected_shape=(self.action_dim,))
            
            if reward is not None:
                if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
                    raise ValidationError(f"Invalid reward value: {reward}")
            # Record reward
            if reward is not None:
                self.history['rewards'].append(reward)
                self.total_reward += reward
                self.logger.debug(f"Recorded reward: {reward}, total: {self.total_reward}")
            
            # Basic model learning based on prediction error
            try:
                # Get current beliefs about previous state
                if len(self.history['beliefs']) > 0:
                    previous_beliefs = self.history['beliefs'][-1]
                    
                    # Compute prediction error
                    if hasattr(self.generative_model, 'predict_observation'):
                        predicted_obs = self.generative_model.predict_observation(
                            previous_beliefs, action
                        )
                        prediction_error = np.linalg.norm(observation - predicted_obs)
                        
                        # Simple learning rule: adjust model parameters based on error
                        learning_rate = 0.01
                        if hasattr(self.generative_model, 'update_parameters'):
                            self.generative_model.update_parameters(
                                prediction_error, learning_rate
                            )
                        
                        self.logger.debug(f"Model update: prediction_error={prediction_error:.4f}")
                    else:
                        # Fallback: just update observation statistics
                        if hasattr(self.generative_model, 'update_observation_stats'):
                            self.generative_model.update_observation_stats(observation)
                
                self.logger.debug("Model update completed")
                
            except Exception as e:
                self.logger.warning(f"Model learning failed, using fallback: {e}")
                # Continue without model updates - system remains functional
                
        except (ValidationError, ModelError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("model_update", e)
    
    def update_beliefs(self, observation: np.ndarray, action: Optional[np.ndarray] = None) -> None:
        """
        Update agent beliefs based on new observation.
        
        Args:
            observation: New observation
            action: Optional action that led to this observation
        """
        try:
            validate_array(observation, "observation", expected_shape=(self.obs_dim,))
            
            # Update beliefs through inference
            self.beliefs = self.infer_states(observation)
            self.logger.debug("Beliefs updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update beliefs: {e}")
            raise
    
    @handle_errors((ActiveInferenceError, ValidationError), log_errors=True)
    def reset(self, observation: Optional[np.ndarray] = None) -> None:
        """
        Reset agent for new episode.
        
        Args:
            observation: Initial observation (optional)
            
        Raises:
            ValidationError: If initial observation is invalid
            ActiveInferenceError: If reset fails
        """
        try:
            # Validate initial observation if provided
            if observation is not None:
                validate_array(observation, "initial_observation", expected_shape=(self.obs_dim,))
            # Reset beliefs to priors
            self._initialize_beliefs()
            
            # Reset episode-specific counters
            self.step_count = 0
            self.episode_count += 1
            self.total_reward = 0.0
            
            # Reset health status if not critical
            if self._health_status != "critical":
                self._health_status = "healthy"
            
            # Process initial observation if provided
            if observation is not None:
                self.infer_states(observation)
            
            self.logger.info(f"Agent reset for episode {self.episode_count}")
            
        except (ValidationError, ActiveInferenceError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("reset", e)
            raise ActiveInferenceError(f"Agent reset failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health and error status.
        
        Returns:
            Dictionary with health metrics
        """
        recent_errors = [e for e in self.history['errors'] 
                        if self.step_count - e['step_count'] < 100] if self.history['errors'] else []
        
        return {
            'health_status': self._health_status,
            'is_initialized': self._is_initialized,
            'total_errors': self._error_count,
            'recent_errors': len(recent_errors),
            'last_error': self._last_error,
            'error_rate': len(recent_errors) / max(1, min(100, self.step_count)) if self.step_count > 0 else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.history['free_energy']:
                return self.get_health_status()
        
            recent_fe = [fe.total for fe in self.history['free_energy'][-100:]] if self.history['free_energy'] else []
            
            stats = {
                'agent_id': self.agent_id,
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'total_reward': self.total_reward,
                'average_reward': self.total_reward / max(1, len(self.history['rewards'])) if self.history['rewards'] else 0,
                'current_free_energy': self.history['free_energy'][-1].total if self.history['free_energy'] else 0,
                'average_free_energy': np.mean(recent_fe) if recent_fe else 0,
                'belief_confidence': self.beliefs.average_confidence() if self.beliefs else 0,
                'belief_entropy': self.beliefs.total_entropy() if self.beliefs else 0,
                'history_length': len(self.history['observations']),
            }
            
            # Add health status
            stats.update(self.get_health_status())
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to compute statistics: {e}")
            return self.get_health_status()
    
    @handle_errors((Exception,), log_errors=True)
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent state to checkpoint file.
        
        Args:
            filepath: Path to save checkpoint
            
        Raises:
            ValidationError: If filepath is invalid
            Exception: If save operation fails
        """
        try:
            if not isinstance(filepath, str) or not filepath.strip():
                raise ValidationError("filepath must be a non-empty string")
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
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
                    'max_history_length': self.max_history_length,
                },
                'statistics': self.get_statistics(),
                'model': self.generative_model.get_model_parameters(),
                'history_length': len(self.history['observations']),
                'health_status': self.get_health_status(),
                'version': '2.0'  # Mark as Generation 2 with enhanced error handling
            }
            
            import json
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            checkpoint = convert_numpy(checkpoint)
            
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            
            self.logger.info(f"Agent checkpoint saved to {filepath}")
            
        except (ValidationError, FileNotFoundError, PermissionError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("checkpoint_save", e)
            raise Exception(f"Failed to save checkpoint: {e}")
    
    @classmethod
    @handle_errors((Exception,), log_errors=True)
    def load_checkpoint(cls, filepath: str) -> 'ActiveInferenceAgent':
        """Load agent from checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Restored agent instance
            
        Raises:
            ValidationError: If filepath is invalid or file corrupted
            FileNotFoundError: If checkpoint file doesn't exist
            Exception: If load operation fails
        """
        try:
            if not isinstance(filepath, str) or not filepath.strip():
                raise ValidationError("filepath must be a non-empty string")
            
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            import json
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            
            # Validate checkpoint structure
            required_keys = ['config', 'statistics']
            for key in required_keys:
                if key not in checkpoint:
                    raise ValidationError(f"Invalid checkpoint: missing '{key}'")
            
            config = checkpoint['config']
            
            # Validate config parameters
            required_config_keys = ['state_dim', 'obs_dim', 'action_dim', 'agent_id']
            for key in required_config_keys:
                if key not in config:
                    raise ValidationError(f"Invalid checkpoint config: missing '{key}'")
            
            # Create agent with restored config
            agent = cls(**config)
            
            # Restore statistics if available
            if 'statistics' in checkpoint:
                try:
                    stats = checkpoint['statistics']
                    if 'episode_count' in stats:
                        agent.episode_count = stats['episode_count']
                    if 'total_reward' in stats:
                        agent.total_reward = stats['total_reward']
                except Exception as e:
                    agent.logger.warning(f"Failed to restore statistics: {e}")
            
            # Restore model if available
            if 'model' in checkpoint:
                try:
                    # This would restore the generative model parameters
                    # Implementation depends on model serialization format
                    agent.logger.info("Model parameters available but not restored (not implemented)")
                except Exception as e:
                    agent.logger.warning(f"Failed to restore model: {e}")
            
            agent.logger.info(f"Agent restored from checkpoint: {filepath}")
            return agent
            
        except (ValidationError, FileNotFoundError, json.JSONDecodeError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Handle unexpected errors
            raise Exception(f"Failed to load checkpoint: {e}")
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return (f"ActiveInferenceAgent(id={self.agent_id}, "
                f"dims=[{self.state_dim}, {self.obs_dim}, {self.action_dim}], "
                f"episodes={self.episode_count}, steps={self.step_count}, "
                f"health={self._health_status}, errors={self._error_count})")
    
    def __del__(self):
        """Cleanup when agent is destroyed."""
        try:
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Agent {self.agent_id} destroyed after {self.step_count} steps")
        except:
            pass  # Ignore errors during cleanup