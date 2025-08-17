"""
Enhanced Active Inference Agent with Adaptive Dimensional Handling
Generation 2: MAKE IT ROBUST (Reliable)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from .agent import ActiveInferenceAgent
from ..utils.validation import SecurityValidator, AdvancedInputValidator

logger = logging.getLogger(__name__)


class AdaptiveActiveInferenceAgent(ActiveInferenceAgent):
    """
    Enhanced Active Inference Agent with adaptive dimensional handling
    and robust error recovery mechanisms.
    
    Generation 2 Features:
    - Adaptive observation space handling
    - Robust dimensional compatibility 
    - Enhanced error recovery
    - Security validation
    - Performance monitoring
    """
    
    def __init__(self, 
                 obs_dim: int = 4,
                 action_dim: int = 2,
                 state_dim: Optional[int] = None,
                 belief_dims: Optional[int] = None,
                 adaptive_dimensions: bool = True,
                 security_validation: bool = True,
                 performance_monitoring: bool = True,
                 **kwargs):
        """
        Initialize adaptive active inference agent.
        
        Args:
            obs_dim: Expected observation dimensionality
            action_dim: Action space dimensionality
            state_dim: Internal state dimensionality (auto-inferred if None)
            belief_dims: Belief space dimensionality (auto-inferred if None)
            adaptive_dimensions: Enable adaptive dimensional handling
            security_validation: Enable security validation
            performance_monitoring: Enable performance monitoring
        """
        # Auto-infer state_dim if not provided
        if 'state_dim' not in kwargs:
            kwargs['state_dim'] = state_dim or max(obs_dim, 4)
        
        # Initialize parent with basic configuration
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, **kwargs)
        
        # Enhanced configuration
        self.adaptive_dimensions = adaptive_dimensions
        self.security_validation = security_validation
        self.performance_monitoring = performance_monitoring
        
        # Adaptive dimension tracking
        self.observed_obs_dims = set([obs_dim])
        self.obs_dim_history = []
        self.adaptation_count = 0
        
        # State dimensions (auto-infer if not provided)
        self.state_dim = state_dim or max(obs_dim, 4)
        self.belief_dims = belief_dims or self.state_dim * 2
        
        # Security validator
        if self.security_validation:
            self.security_validator = SecurityValidator()
            self.input_validator = AdvancedInputValidator()
        
        # Performance metrics
        self.performance_metrics = {
            'adaptations': 0,
            'validation_failures': 0,
            'dimension_mismatches_recovered': 0,
            'total_steps': 0,
            'avg_inference_time': 0.0,
            'error_recovery_count': 0
        }
        
        logger.info(f"AdaptiveAgent {self.agent_id} initialized with adaptive_dims={adaptive_dimensions}")
    
    def adapt_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Adapt observation to expected dimensions with robust handling.
        
        Args:
            observation: Input observation
            
        Returns:
            Adapted observation matching expected dimensions
        """
        try:
            if observation is None:
                logger.warning("Received None observation, using zero vector")
                return np.zeros(self.obs_dim)
            
            obs = np.asarray(observation)
            
            # Handle scalar observations
            if obs.ndim == 0:
                obs = np.array([obs])
            
            # Handle multi-dimensional observations
            if obs.ndim > 1:
                obs = obs.flatten()
            
            current_dim = len(obs)
            
            # Track dimensional variations
            if current_dim not in self.observed_obs_dims:
                self.observed_obs_dims.add(current_dim)
                self.adaptation_count += 1
                logger.info(f"Agent {self.agent_id} adapting to new observation dimension: {current_dim}")
            
            self.obs_dim_history.append(current_dim)
            
            # Adaptive dimensional handling
            if self.adaptive_dimensions and current_dim != self.obs_dim:
                if current_dim < self.obs_dim:
                    # Pad with zeros
                    adapted_obs = np.zeros(self.obs_dim)
                    adapted_obs[:current_dim] = obs
                    self.performance_metrics['dimension_mismatches_recovered'] += 1
                elif current_dim > self.obs_dim:
                    # Truncate or compress
                    if current_dim <= self.obs_dim * 2:
                        # Simple truncation
                        adapted_obs = obs[:self.obs_dim]
                    else:
                        # Compression using averaging
                        chunks = np.array_split(obs, self.obs_dim)
                        adapted_obs = np.array([chunk.mean() for chunk in chunks])
                    self.performance_metrics['dimension_mismatches_recovered'] += 1
                else:
                    adapted_obs = obs
            else:
                adapted_obs = obs
            
            # Ensure proper shape
            adapted_obs = np.asarray(adapted_obs).flatten()[:self.obs_dim]
            if len(adapted_obs) < self.obs_dim:
                padded = np.zeros(self.obs_dim)
                padded[:len(adapted_obs)] = adapted_obs
                adapted_obs = padded
                
            return adapted_obs
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} observation adaptation failed: {e}")
            self.performance_metrics['error_recovery_count'] += 1
            return np.zeros(self.obs_dim)
    
    def reset(self, initial_observation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Reset agent with enhanced validation and adaptation."""
        try:
            # Security validation
            if self.security_validation and initial_observation is not None:
                if not self.security_validator.validate_input(initial_observation):
                    logger.warning(f"Agent {self.agent_id} security validation failed on reset")
                    self.performance_metrics['validation_failures'] += 1
                    initial_observation = None
            
            # Adapt initial observation
            if initial_observation is not None:
                initial_observation = self.adapt_observation(initial_observation)
            
            # Call parent reset
            result = super().reset(initial_observation)
            
            # Reset performance counters for new episode
            self.performance_metrics['total_steps'] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} reset failed: {e}")
            self.performance_metrics['error_recovery_count'] += 1
            return {"beliefs": np.random.randn(self.belief_dims), "status": "error_recovery"}
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Enhanced action selection with robust error handling."""
        try:
            import time
            start_time = time.time()
            
            # Security validation
            if self.security_validation:
                if not self.security_validator.validate_input(observation):
                    logger.warning(f"Agent {self.agent_id} security validation failed")
                    self.performance_metrics['validation_failures'] += 1
                    observation = np.zeros(self.obs_dim)
            
            # Adapt observation dimensions
            adapted_obs = self.adapt_observation(observation)
            
            # Call parent action selection
            action = super().act(adapted_obs)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.performance_metrics['total_steps'] += 1
            alpha = 0.9  # Exponential moving average
            self.performance_metrics['avg_inference_time'] = (
                alpha * self.performance_metrics['avg_inference_time'] + 
                (1 - alpha) * inference_time
            )
            
            return action
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} action selection failed: {e}")
            self.performance_metrics['error_recovery_count'] += 1
            return np.zeros(self.action_dim)
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation and performance statistics."""
        return {
            'observed_dimensions': list(self.observed_obs_dims),
            'adaptation_count': self.adaptation_count,
            'dimension_history_length': len(self.obs_dim_history),
            'performance_metrics': self.performance_metrics.copy(),
            'adaptive_capabilities': {
                'dimensional_adaptation': self.adaptive_dimensions,
                'security_validation': self.security_validation,
                'performance_monitoring': self.performance_monitoring
            },
            'current_config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'state_dim': self.state_dim,
                'belief_dims': self.belief_dims
            }
        }
    
    def enable_auto_dimension_optimization(self):
        """Enable automatic dimension optimization based on observation history."""
        if len(self.obs_dim_history) > 10:
            # Find most common dimension
            from collections import Counter
            dim_counts = Counter(self.obs_dim_history[-50:])  # Last 50 observations
            most_common_dim = dim_counts.most_common(1)[0][0]
            
            if most_common_dim != self.obs_dim and most_common_dim in self.observed_obs_dims:
                logger.info(f"Agent {self.agent_id} auto-optimizing to dimension {most_common_dim}")
                self.obs_dim = most_common_dim
                self.performance_metrics['adaptations'] += 1
                return True
        return False


class MultiModalAdaptiveAgent(AdaptiveActiveInferenceAgent):
    """
    Multi-modal adaptive agent with specialized modality handling.
    """
    
    def __init__(self, 
                 modalities: Dict[str, int],
                 fusion_method: str = "attention",
                 **kwargs):
        """
        Initialize multi-modal adaptive agent.
        
        Args:
            modalities: Dictionary mapping modality names to dimensions
            fusion_method: Method for fusing multi-modal information
        """
        total_obs_dim = sum(modalities.values())
        super().__init__(obs_dim=total_obs_dim, **kwargs)
        
        self.modalities = modalities
        self.fusion_method = fusion_method
        self.modality_agents = {}
        
        # Create specialized agents for each modality
        for modality, dim in modalities.items():
            self.modality_agents[modality] = AdaptiveActiveInferenceAgent(
                obs_dim=dim,
                action_dim=self.action_dim,
                agent_id=f"{self.agent_id}_{modality}",
                adaptive_dimensions=True
            )
    
    def process_multimodal_observation(self, 
                                     observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Process multi-modal observations with robust handling.
        
        Args:
            observations: Dictionary of modality observations
            
        Returns:
            Fused observation vector
        """
        try:
            fused_obs = []
            
            for modality, expected_dim in self.modalities.items():
                if modality in observations:
                    # Use modality-specific agent for adaptation
                    adapted_obs = self.modality_agents[modality].adapt_observation(
                        observations[modality]
                    )
                else:
                    # Use zero vector for missing modalities
                    adapted_obs = np.zeros(expected_dim)
                    logger.warning(f"Missing modality {modality}, using zero vector")
                
                fused_obs.extend(adapted_obs)
            
            return np.array(fused_obs)
            
        except Exception as e:
            logger.error(f"Multi-modal observation processing failed: {e}")
            return np.zeros(self.obs_dim)