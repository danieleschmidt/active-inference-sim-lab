"""
Free Energy Principle implementation.

This module implements the core free energy computations that drive
active inference agents, including accuracy and complexity terms.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .beliefs import Belief, BeliefState


@dataclass
class FreeEnergyComponents:
    """Components of the free energy functional."""
    accuracy: float
    complexity: float
    total: float
    
    def __post_init__(self):
        """Ensure total equals accuracy + complexity."""
        self.total = self.accuracy + self.complexity


class FreeEnergyObjective:
    """
    Implements the Free Energy Principle objective function.
    
    Free Energy = Complexity - Accuracy
    - Complexity: KL divergence between beliefs and priors
    - Accuracy: Expected log-likelihood of observations under beliefs
    """
    
    def __init__(self, 
                 complexity_weight: float = 1.0,
                 accuracy_weight: float = 1.0,
                 temperature: float = 1.0):
        """
        Initialize free energy objective.
        
        Args:
            complexity_weight: Weight for complexity term (KL divergence)
            accuracy_weight: Weight for accuracy term (log-likelihood) 
            temperature: Temperature parameter for softmax computations
        """
        self.complexity_weight = complexity_weight
        self.accuracy_weight = accuracy_weight
        self.temperature = temperature
    
    def compute_accuracy(self, 
                        observations: np.ndarray,
                        beliefs: BeliefState,
                        likelihood_fn: Callable) -> float:
        """
        Compute accuracy term (expected log-likelihood).
        
        Args:
            observations: Observed data
            beliefs: Current belief state
            likelihood_fn: Function mapping beliefs to observation likelihoods
            
        Returns:
            Negative log-likelihood (higher = less accurate)
        """
        if len(beliefs) == 0:
            return float('inf')  # No beliefs = infinite inaccuracy
        
        total_log_likelihood = 0.0
        
        # Compute likelihood for each belief
        for name, belief in beliefs.get_all_beliefs().items():
            # Sample from belief to compute expected likelihood
            n_samples = 100  # Monte Carlo samples
            samples = belief.sample(n_samples)
            
            sample_likelihoods = []
            for sample in samples:
                likelihood = likelihood_fn(sample, observations)
                sample_likelihoods.append(np.log(likelihood + 1e-8))
            
            # Expected log-likelihood
            expected_log_likelihood = np.mean(sample_likelihoods)
            total_log_likelihood += expected_log_likelihood
        
        # Return negative log-likelihood (minimize = maximize likelihood)
        return -total_log_likelihood * self.accuracy_weight
    
    def compute_complexity(self, 
                          beliefs: BeliefState,
                          priors: Dict[str, Belief]) -> float:
        """
        Compute complexity term (KL divergence from priors).
        
        Args:
            beliefs: Current beliefs
            priors: Prior beliefs
            
        Returns:
            KL divergence between beliefs and priors
        """
        total_kl = 0.0
        
        for name, belief in beliefs.get_all_beliefs().items():
            if name not in priors:
                continue
                
            prior = priors[name]
            
            # KL divergence between Gaussians
            kl = self._kl_divergence_gaussian(belief, prior)
            total_kl += kl
        
        return total_kl * self.complexity_weight
    
    def _kl_divergence_gaussian(self, q: Belief, p: Belief) -> float:
        """
        Compute KL divergence between two Gaussian beliefs.
        
        KL(q||p) = 0.5 * (tr(Σp^-1 Σq) + (μp-μq)^T Σp^-1 (μp-μq) - k + log(|Σp|/|Σq|))
        """
        # Ensure beliefs have same dimensionality
        if q.mean.shape != p.mean.shape:
            raise ValueError("Beliefs must have same dimensionality")
        
        k = len(q.mean)  # Dimensionality
        
        # Mean difference
        mu_diff = p.mean - q.mean
        
        # Covariance terms (assuming diagonal covariances)
        sigma_p_inv = p.precision
        sigma_q = q.variance
        sigma_p = p.variance
        
        # KL divergence components
        trace_term = (sigma_p_inv * sigma_q).sum()
        quadratic_term = (mu_diff * sigma_p_inv * mu_diff).sum()
        log_det_term = (np.log(sigma_p) - np.log(sigma_q)).sum()
        
        kl = 0.5 * (trace_term + quadratic_term - k + log_det_term)
        return max(0.0, kl)  # Ensure non-negative
    
    def compute_free_energy(self,
                           observations: np.ndarray,
                           beliefs: BeliefState,
                           priors: Dict[str, Belief],
                           likelihood_fn: Callable) -> FreeEnergyComponents:
        """
        Compute total free energy and its components.
        
        Args:
            observations: Observed data
            beliefs: Current belief state  
            priors: Prior beliefs
            likelihood_fn: Observation likelihood function
            
        Returns:
            Free energy components (accuracy, complexity, total)
        """
        accuracy = self.compute_accuracy(observations, beliefs, likelihood_fn)
        complexity = self.compute_complexity(beliefs, priors)
        
        return FreeEnergyComponents(
            accuracy=accuracy,
            complexity=complexity,
            total=accuracy + complexity
        )
    
    def expected_free_energy(self,
                           future_observations: np.ndarray,
                           predicted_beliefs: BeliefState,
                           priors: Dict[str, Belief],
                           likelihood_fn: Callable) -> float:
        """
        Compute expected free energy for future states.
        
        Used for action selection - agents choose actions that minimize
        expected free energy under their predictive model.
        
        Args:
            future_observations: Predicted future observations
            predicted_beliefs: Predicted future belief state
            priors: Prior beliefs
            likelihood_fn: Observation likelihood function
            
        Returns:
            Expected free energy value
            
        Raises:
            ValidationError: If inputs are invalid
            ModelError: If computation fails
        """
        try:
            # Validate inputs
            validate_array(future_observations, "future_observations")
            
            if not isinstance(predicted_beliefs, BeliefState):
                raise ValidationError(f"predicted_beliefs must be BeliefState, got {type(predicted_beliefs)}")
            
            if not isinstance(priors, dict):
                raise ValidationError(f"priors must be dictionary, got {type(priors)}")
            
            if not callable(likelihood_fn):
                raise ValidationError("likelihood_fn must be callable")
            
            # This is simplified - full implementation would integrate over
            # all possible future observations weighted by their probability
            try:
                future_fe = self.compute_free_energy(
                    future_observations, predicted_beliefs, priors, likelihood_fn
                )
                
                if not future_fe.is_valid():
                    self.logger.warning("Invalid future free energy computed")
                    return float('inf')
                
                return future_fe.total
                
            except Exception as e:
                self.logger.error(f"Error computing future free energy: {e}")
                return float('inf')  # Conservative fallback
            
        except (ValidationError, ModelError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Handle unexpected errors
            self._record_error("expected_free_energy", e)
            raise ModelError(f"Unexpected error computing expected free energy: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get free energy computation statistics.
        
        Returns:
            Dictionary with error counts and settings
        """
        try:
            return {
                'complexity_weight': self.complexity_weight,
                'accuracy_weight': self.accuracy_weight,
                'temperature': self.temperature,
                'numerical_stability': self.numerical_stability,
                'max_samples': self.max_samples,
                'error_count': self._error_count,
                'last_error': self._last_error
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def reset_statistics(self) -> None:
        """Reset error tracking statistics."""
        self._error_count = 0
        self._last_error = None
        self.logger.info("Free energy statistics reset")
    
    def __repr__(self) -> str:
        """String representation of free energy objective."""
        return (f"FreeEnergyObjective(complexity_weight={self.complexity_weight}, "
                f"accuracy_weight={self.accuracy_weight}, temperature={self.temperature}, "
                f"errors={self._error_count})")
    
    def __del__(self):
        """Cleanup when objective is destroyed."""
        try:
            if hasattr(self, 'logger') and self.logger and self._error_count > 0:
                self.logger.info(f"FreeEnergyObjective destroyed with {self._error_count} errors")
        except:
            pass  # Ignore errors during cleanup