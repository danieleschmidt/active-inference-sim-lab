"""
Variational inference implementation for active inference.

This module implements variational inference for updating beliefs
about hidden states based on observations.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
from scipy.optimize import minimize

from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel


class VariationalInference:
    """
    Variational inference engine for belief updating.
    
    Uses variational optimization to approximate the posterior distribution
    over hidden states given observations.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6):
        """
        Initialize variational inference engine.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence criterion for optimization
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def update_beliefs(self,
                      observations: np.ndarray,
                      prior_beliefs: BeliefState,
                      generative_model: GenerativeModel) -> BeliefState:
        """
        Update beliefs using variational inference.
        
        Args:
            observations: Current observations
            prior_beliefs: Prior belief state
            generative_model: Agent's generative model
            
        Returns:
            Updated posterior beliefs
        """
        updated_beliefs = BeliefState()
        
        # Update each belief independently
        for name, prior in prior_beliefs.get_all_beliefs().items():
            posterior = self._update_single_belief(
                observations, prior, generative_model, name
            )
            updated_beliefs.add_belief(name, posterior)
        
        return updated_beliefs
    
    def _update_single_belief(self,
                             observations: np.ndarray,
                             prior: Belief,
                             generative_model: GenerativeModel,
                             belief_name: str) -> Belief:
        """
        Update a single belief using variational optimization.
        
        Args:
            observations: Current observations
            prior: Prior belief
            generative_model: Generative model
            belief_name: Name of the belief being updated
            
        Returns:
            Updated posterior belief
        """
        # Initial parameters (start from prior)
        initial_mean = prior.mean.copy()
        initial_log_var = np.log(prior.variance)
        
        # Flatten parameters for optimization
        initial_params = np.concatenate([initial_mean, initial_log_var])
        
        # Define objective function (negative ELBO)
        def objective(params):
            return -self._compute_elbo(params, observations, prior, generative_model)
        
        # Define gradient function
        def gradient(params):
            return -self._compute_elbo_gradient(params, observations, prior, generative_model)
        
        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=gradient,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_threshold
            }
        )
        
        # Extract optimized parameters
        dim = len(initial_mean)
        optimized_mean = result.x[:dim]
        optimized_log_var = result.x[dim:]
        
        # Clip log variance to prevent overflow
        optimized_log_var = np.clip(optimized_log_var, -10, 10)
        optimized_var = np.exp(optimized_log_var)
        
        # Ensure variance is positive and bounded
        optimized_var = np.clip(optimized_var, 1e-6, 1e2)
        
        return Belief(
            mean=optimized_mean,
            variance=optimized_var,
            support=prior.support
        )
    
    def _compute_elbo(self,
                     params: np.ndarray,
                     observations: np.ndarray,
                     prior: Belief,
                     generative_model: GenerativeModel) -> float:
        """
        Compute Evidence Lower BOund (ELBO).
        
        ELBO = E_q[log p(o|s)] - KL[q(s)||p(s)]
        """
        dim = len(prior.mean)
        mean = params[:dim]
        log_var = params[dim:]
        var = np.exp(log_var)
        
        # Create belief from current parameters
        current_belief = Belief(mean=mean, variance=var)
        
        # Expected log-likelihood term (Monte Carlo estimate)
        n_samples = 50
        samples = current_belief.sample(n_samples)
        
        log_likelihood = 0.0
        for sample in samples:
            likelihood = generative_model.likelihood(sample, observations)
            log_likelihood += np.log(likelihood + 1e-8)
        
        expected_log_likelihood = log_likelihood / n_samples
        
        # KL divergence term
        kl_divergence = self._kl_divergence_gaussian(current_belief, prior)
        
        # ELBO = expected log-likelihood - KL divergence
        elbo = expected_log_likelihood - kl_divergence
        
        return elbo
    
    def _compute_elbo_gradient(self,
                              params: np.ndarray,
                              observations: np.ndarray,
                              prior: Belief,
                              generative_model: GenerativeModel) -> np.ndarray:
        """
        Compute gradient of ELBO with respect to variational parameters.
        
        This is a simplified implementation using numerical gradients.
        A full implementation would use analytical gradients.
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        
        # Numerical gradient
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            
            params_minus = params.copy()
            params_minus[i] -= eps
            
            elbo_plus = self._compute_elbo(params_plus, observations, prior, generative_model)
            elbo_minus = self._compute_elbo(params_minus, observations, prior, generative_model)
            
            grad[i] = (elbo_plus - elbo_minus) / (2 * eps)
        
        return grad
    
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