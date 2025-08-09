"""
Belief updater interface and implementations.

This module provides various belief updating methods including variational inference,
Kalman filtering, and particle filtering with comprehensive logging and error handling.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import time
import warnings

from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from .variational import VariationalInference
from ..utils.validation import (
    ValidationError, InferenceError,
    validate_array, validate_inputs, handle_errors
)
from ..utils.logging_config import (
    get_logger, LogCategory, PerformanceTimer
)


class BeliefUpdater(ABC):
    """
    Abstract base class for belief updating methods.
    
    Provides common logging, validation, and monitoring functionality
    for all belief updating implementations.
    """
    
    def __init__(self, name: str = "BeliefUpdater"):
        """
        Initialize belief updater.
        
        Args:
            name: Name for logging identification
        """
        self.name = name
        self.logger = get_logger("inference")
        self._update_count = 0
        self._error_count = 0
        self._total_update_time = 0.0
        self._last_error = None
        
        self.logger.info(f"Initialized {self.name}", LogCategory.INFERENCE, {
            'updater_type': self.__class__.__name__
        })
    
    def update(self,
               prior_beliefs: BeliefState,
               observations: np.ndarray,
               model: GenerativeModel) -> BeliefState:
        """
        Update beliefs given observations with comprehensive monitoring.
        
        Args:
            prior_beliefs: Prior belief state
            observations: New observations
            model: Generative model
            
        Returns:
            Updated belief state
            
        Raises:
            ValidationError: If inputs are invalid
            InferenceError: If update fails critically
        """
        start_time = time.time()
        success = False
        
        try:
            # Validate inputs
            if not isinstance(prior_beliefs, BeliefState):
                raise ValidationError(f"prior_beliefs must be BeliefState, got {type(prior_beliefs)}")
            
            if len(prior_beliefs) == 0:
                raise ValidationError("prior_beliefs cannot be empty")
            
            validate_array(observations, "observations")
            
            # Perform the actual update (implemented by subclasses)
            updated_beliefs = self._update_implementation(
                prior_beliefs, observations, model
            )
            
            success = True
            return updated_beliefs
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(
                f"Belief update failed in {self.name}",
                LogCategory.INFERENCE,
                error=e
            )
            raise InferenceError(f"Update failed in {self.name}: {e}")
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            self._update_count += 1
            self._total_update_time += duration
    
    @abstractmethod
    def _update_implementation(self,
                              prior_beliefs: BeliefState,
                              observations: np.ndarray,
                              model: GenerativeModel) -> BeliefState:
        """
        Actual belief update implementation (to be implemented by subclasses).
        
        Args:
            prior_beliefs: Prior belief state
            observations: New observations
            model: Generative model
            
        Returns:
            Updated belief state
        """
        pass


class VariationalBeliefUpdater(BeliefUpdater):
    """Variational inference belief updater with enhanced monitoring."""
    
    def __init__(self, **kwargs):
        super().__init__("VariationalBeliefUpdater")
        
        try:
            # Filter kwargs to only include valid VariationalInference parameters
            valid_params = {'learning_rate', 'max_iterations', 'convergence_threshold'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            self.inference_engine = VariationalInference(**filtered_kwargs)
            self.logger.info(
                "Variational inference engine initialized",
                LogCategory.INFERENCE,
                {'config': kwargs}
            )
        except Exception as e:
            self.logger.error(
                "Failed to initialize variational inference engine",
                LogCategory.INFERENCE,
                error=e
            )
            raise InferenceError(f"Variational updater initialization failed: {e}")
    
    def _update_implementation(self,
                              prior_beliefs: BeliefState,
                              observations: np.ndarray,
                              model: GenerativeModel) -> BeliefState:
        """Update beliefs using variational inference."""
        return self.inference_engine.update_beliefs(
            observations, prior_beliefs, model
        )


class KalmanBeliefUpdater(BeliefUpdater):
    """Kalman filter belief updater for linear-Gaussian models with enhanced monitoring."""
    
    def __init__(self, process_noise: float = 0.1, observation_noise: float = 0.1):
        super().__init__("KalmanBeliefUpdater")
        
        try:
            # Validate parameters
            if process_noise <= 0 or process_noise > 1:
                raise ValidationError(f"process_noise must be between 0 and 1, got {process_noise}")
            
            if observation_noise <= 0 or observation_noise > 1:
                raise ValidationError(f"observation_noise must be between 0 and 1, got {observation_noise}")
            
            self.process_noise = process_noise
            self.observation_noise = observation_noise
            
            self.logger.info(
                "Kalman filter initialized",
                LogCategory.INFERENCE,
                {
                    'process_noise': process_noise,
                    'observation_noise': observation_noise
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize Kalman filter",
                LogCategory.INFERENCE,
                error=e
            )
            raise InferenceError(f"Kalman updater initialization failed: {e}")
    
    def _update_implementation(self,
                              prior_beliefs: BeliefState,
                              observations: np.ndarray,
                              model: GenerativeModel) -> BeliefState:
        """Update beliefs using Kalman filter."""
        updated_beliefs = BeliefState()
        
        for name, prior in prior_beliefs.get_all_beliefs().items():
            # Simplified Kalman update
            prior_var = prior.variance + self.process_noise
            
            # Observation prediction
            H = np.eye(len(observations))  # Simplified observation matrix
            R = np.eye(len(observations)) * self.observation_noise
            
            # Kalman gain
            S = H @ np.diag(prior_var) @ H.T + R
            K = np.diag(prior_var) @ H.T @ np.linalg.inv(S)
            
            # Update
            predicted_obs = H @ prior.mean
            innovation = observations - predicted_obs
            
            posterior_mean = prior.mean + K @ innovation
            posterior_var = np.diag((np.eye(len(prior.mean)) - K @ H) @ np.diag(prior_var))
            
            updated_beliefs.add_belief(name, Belief(
                mean=posterior_mean,
                variance=posterior_var,
                support=prior.support
            ))
        
        return updated_beliefs


class ParticleBeliefUpdater(BeliefUpdater):
    """Particle filter belief updater."""
    
    def __init__(self, n_particles: int = 1000, resample_threshold: float = 0.5):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.particles = {}
        self.weights = {}
    
    def update(self,
               prior_beliefs: BeliefState,
               observations: np.ndarray,
               model: GenerativeModel) -> BeliefState:
        """Update beliefs using particle filter."""
        updated_beliefs = BeliefState()
        
        for name, prior in prior_beliefs.get_all_beliefs().items():
            # Initialize particles if needed
            if name not in self.particles:
                self.particles[name] = prior.sample(self.n_particles)
                self.weights[name] = np.ones(self.n_particles) / self.n_particles
            
            # Predict step (motion model)
            noise = np.random.normal(0, 0.1, self.particles[name].shape)
            self.particles[name] += noise
            
            # Update step (measurement model)
            weights = np.zeros(self.n_particles)
            for i, particle in enumerate(self.particles[name]):
                likelihood = model.likelihood(particle, observations)
                weights[i] = self.weights[name][i] * likelihood
            
            # Normalize weights
            weights = weights / (np.sum(weights) + 1e-8)
            self.weights[name] = weights
            
            # Effective sample size
            eff_n = 1.0 / np.sum(weights**2)
            
            # Resample if needed
            if eff_n < self.resample_threshold * self.n_particles:
                indices = np.random.choice(
                    self.n_particles, self.n_particles, p=weights
                )
                self.particles[name] = self.particles[name][indices]
                self.weights[name] = np.ones(self.n_particles) / self.n_particles
            
            # Compute posterior statistics
            posterior_mean = np.average(self.particles[name], weights=self.weights[name], axis=0)
            posterior_var = np.average(
                (self.particles[name] - posterior_mean)**2,
                weights=self.weights[name], axis=0
            )
            
            updated_beliefs.add_belief(name, Belief(
                mean=posterior_mean,
                variance=posterior_var,
                support=prior.support
            ))
        
        return updated_beliefs