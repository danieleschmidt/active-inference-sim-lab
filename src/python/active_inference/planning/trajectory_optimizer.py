"""
Trajectory optimization for active inference planning.

This module implements sophisticated trajectory optimization algorithms
for long-horizon planning in active inference agents.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import minimize
import logging

from ..core.beliefs import BeliefState
from ..core.generative_model import GenerativeModel
from .expected_free_energy import ExpectedFreeEnergy, ExpectedFreeEnergyComponents


class TrajectoryOptimizer:
    """
    Trajectory optimizer for long-horizon active inference planning.
    
    Optimizes full action sequences rather than single actions,
    enabling sophisticated multi-step planning strategies.
    """
    
    def __init__(self,
                 horizon: int = 10,
                 action_dim: int = 2,
                 optimization_method: str = "lbfgs",
                 max_iterations: int = 100,
                 convergence_tolerance: float = 1e-6):
        """
        Initialize trajectory optimizer.
        
        Args:
            horizon: Planning horizon length
            action_dim: Dimensionality of action space
            optimization_method: Optimization algorithm ("lbfgs", "adam", "genetic")
            max_iterations: Maximum optimization iterations
            convergence_tolerance: Convergence threshold
        """
        self.horizon = horizon
        self.action_dim = action_dim
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        # Expected free energy calculator
        self.efe_calculator = ExpectedFreeEnergy()
        
        # Optimization history
        self.optimization_history = []
        
        # Setup logging
        self.logger = logging.getLogger("TrajectoryOptimizer")
    
    def optimize_trajectory(self,
                          beliefs: BeliefState,
                          model: GenerativeModel,
                          initial_trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize action trajectory to minimize expected free energy.
        
        Args:
            beliefs: Current belief state
            model: Generative model
            initial_trajectory: Initial guess for trajectory (optional)
            
        Returns:
            Optimized action trajectory
        """
        if initial_trajectory is None:
            # Random initialization
            initial_trajectory = np.random.randn(self.horizon, self.action_dim) * 0.1
        
        # Flatten trajectory for optimization
        initial_params = initial_trajectory.flatten()
        
        # Define objective function
        def objective(params):
            trajectory = params.reshape(self.horizon, self.action_dim)
            return self._evaluate_trajectory(trajectory, beliefs, model)
        
        # Define gradient function
        def gradient(params):
            return self._compute_trajectory_gradient(params, beliefs, model)
        
        # Optimize based on method
        if self.optimization_method == "lbfgs":
            result = self._optimize_lbfgs(objective, gradient, initial_params)
        elif self.optimization_method == "adam":
            result = self._optimize_adam(objective, gradient, initial_params)
        elif self.optimization_method == "genetic":
            result = self._optimize_genetic(objective, initial_params)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
        
        # Reshape to trajectory
        optimal_trajectory = result.x.reshape(self.horizon, self.action_dim)
        
        # Record optimization statistics
        optimization_stats = {
            'method': self.optimization_method,
            'iterations': result.nit if hasattr(result, 'nit') else self.max_iterations,
            'final_cost': result.fun,
            'success': result.success if hasattr(result, 'success') else True,
            'horizon': self.horizon
        }
        self.optimization_history.append(optimization_stats)
        
        self.logger.info(f"Trajectory optimization completed: cost={result.fun:.4f}")
        
        return optimal_trajectory
    
    def _evaluate_trajectory(self,
                           trajectory: np.ndarray,
                           beliefs: BeliefState,
                           model: GenerativeModel) -> float:
        """
        Evaluate total expected free energy for a trajectory.
        
        Args:
            trajectory: Action sequence [horizon x action_dim]
            beliefs: Current beliefs
            model: Generative model
            
        Returns:
            Total expected free energy
        """
        total_efe = 0.0
        current_beliefs = beliefs
        
        for t, action in enumerate(trajectory):
            # Compute expected free energy for this step
            efe_components = self.efe_calculator.compute_expected_free_energy(
                action, current_beliefs, model, horizon=1
            )
            total_efe += efe_components.total
            
            # Predict belief evolution (simplified)
            # In full implementation, this would simulate belief propagation
            # through the generative model
            
            # For now, add some uncertainty growth over time
            uncertainty_growth = 0.01 * t
            for name, belief in current_beliefs.get_all_beliefs().items():
                belief.variance = belief.variance + uncertainty_growth
        
        return total_efe
    
    def _compute_trajectory_gradient(self,
                                   params: np.ndarray,
                                   beliefs: BeliefState,
                                   model: GenerativeModel) -> np.ndarray:
        """
        Compute gradient of trajectory cost using finite differences.
        
        Args:
            params: Flattened trajectory parameters
            beliefs: Current beliefs
            model: Generative model
            
        Returns:
            Gradient vector
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            
            params_minus = params.copy()
            params_minus[i] -= eps
            
            trajectory_plus = params_plus.reshape(self.horizon, self.action_dim)
            trajectory_minus = params_minus.reshape(self.horizon, self.action_dim)
            
            cost_plus = self._evaluate_trajectory(trajectory_plus, beliefs, model)
            cost_minus = self._evaluate_trajectory(trajectory_minus, beliefs, model)
            
            grad[i] = (cost_plus - cost_minus) / (2 * eps)
        
        return grad
    
    def _optimize_lbfgs(self, objective, gradient, initial_params):
        """Optimize using L-BFGS-B algorithm."""
        return minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=gradient,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_tolerance,
                'gtol': self.convergence_tolerance
            }
        )
    
    def _optimize_adam(self, objective, gradient, initial_params):
        """Optimize using Adam algorithm (simplified implementation)."""
        params = initial_params.copy()
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        
        beta1, beta2 = 0.9, 0.999
        alpha = 0.001  # Learning rate
        eps = 1e-8
        
        best_params = params.copy()
        best_cost = objective(params)
        
        for t in range(1, self.max_iterations + 1):
            grad = gradient(params)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2**t)
            
            # Update parameters
            params = params - alpha * m_hat / (np.sqrt(v_hat) + eps)
            
            # Check for improvement
            cost = objective(params)
            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
            
            # Check convergence
            if np.linalg.norm(grad) < self.convergence_tolerance:
                break
        
        # Create result object
        class OptimizationResult:
            def __init__(self):
                self.x = best_params
                self.fun = best_cost
                self.nit = t
                self.success = True
        
        return OptimizationResult()
    
    def _optimize_genetic(self, objective, initial_params):
        """Optimize using genetic algorithm."""
        population_size = 50
        n_params = len(initial_params)
        
        # Initialize population
        population = np.random.randn(population_size, n_params)
        population[0] = initial_params  # Include initial guess
        
        best_params = initial_params.copy()
        best_cost = objective(initial_params)
        
        for generation in range(self.max_iterations):
            # Evaluate population
            costs = np.array([objective(individual) for individual in population])
            
            # Find best individual
            best_idx = np.argmin(costs)
            if costs[best_idx] < best_cost:
                best_cost = costs[best_idx]
                best_params = population[best_idx].copy()
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size)
                tournament_costs = costs[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_costs)]
                new_population.append(population[winner_idx].copy())
            
            population = np.array(new_population)
            
            # Crossover and mutation
            for i in range(0, population_size-1, 2):
                if np.random.random() < 0.8:  # Crossover probability
                    # Single-point crossover
                    crossover_point = np.random.randint(1, n_params)
                    temp = population[i][crossover_point:].copy()
                    population[i][crossover_point:] = population[i+1][crossover_point:]
                    population[i+1][crossover_point:] = temp
                
                # Mutation
                if np.random.random() < 0.1:  # Mutation probability
                    mutation_strength = 0.1
                    population[i] += np.random.randn(n_params) * mutation_strength
                    population[i+1] += np.random.randn(n_params) * mutation_strength
        
        # Create result object
        class OptimizationResult:
            def __init__(self):
                self.x = best_params
                self.fun = best_cost
                self.success = True
        
        return OptimizationResult()
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        if not self.optimization_history:
            return {}
        
        recent_history = self.optimization_history[-10:]  # Last 10 optimizations
        
        stats = {
            'total_optimizations': len(self.optimization_history),
            'average_cost': np.mean([h['final_cost'] for h in recent_history]),
            'average_iterations': np.mean([h['iterations'] for h in recent_history]),
            'success_rate': np.mean([h['success'] for h in recent_history]),
            'optimization_method': self.optimization_method,
            'current_horizon': self.horizon,
        }
        
        return stats


class ModelPredictiveController:
    """
    Model Predictive Control (MPC) for active inference.
    
    Implements receding horizon control where the full trajectory
    is optimized but only the first action is executed.
    """
    
    def __init__(self,
                 horizon: int = 10,
                 action_dim: int = 2,
                 reoptimize_frequency: int = 1):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Planning horizon
            action_dim: Action dimensionality
            reoptimize_frequency: How often to reoptimize (in steps)
        """
        self.horizon = horizon
        self.action_dim = action_dim
        self.reoptimize_frequency = reoptimize_frequency
        
        # Trajectory optimizer
        self.optimizer = TrajectoryOptimizer(horizon, action_dim)
        
        # Current planned trajectory
        self.current_trajectory = None
        self.trajectory_step = 0
        
        # Statistics
        self.control_history = []
    
    def control(self,
                beliefs: BeliefState,
                model: GenerativeModel) -> np.ndarray:
        """
        Compute control action using MPC.
        
        Args:
            beliefs: Current belief state
            model: Generative model
            
        Returns:
            Control action
        """
        # Check if we need to reoptimize
        if (self.current_trajectory is None or 
            self.trajectory_step % self.reoptimize_frequency == 0 or
            self.trajectory_step >= len(self.current_trajectory)):
            
            # Reoptimize trajectory
            self.current_trajectory = self.optimizer.optimize_trajectory(
                beliefs, model
            )
            self.trajectory_step = 0
        
        # Extract current action
        action = self.current_trajectory[self.trajectory_step]
        
        # Record statistics
        control_stats = {
            'reoptimized': self.trajectory_step % self.reoptimize_frequency == 0,
            'trajectory_step': self.trajectory_step,
            'horizon_remaining': len(self.current_trajectory) - self.trajectory_step
        }
        self.control_history.append(control_stats)
        
        # Advance trajectory
        self.trajectory_step += 1
        
        return action
    
    def reset(self):
        """Reset controller state."""
        self.current_trajectory = None
        self.trajectory_step = 0
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """Get control performance statistics."""
        if not self.control_history:
            return {}
        
        recent_history = self.control_history[-100:]  # Last 100 steps
        
        stats = {
            'total_control_steps': len(self.control_history),
            'reoptimization_rate': np.mean([h['reoptimized'] for h in recent_history]),
            'average_horizon_remaining': np.mean([h['horizon_remaining'] for h in recent_history]),
            'current_trajectory_length': len(self.current_trajectory) if self.current_trajectory is not None else 0,
        }
        
        return stats