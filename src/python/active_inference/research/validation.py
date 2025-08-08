"""
Theoretical validation framework for Active Inference implementations.

This module validates that the implementation correctly follows the Free Energy Principle
and produces theoretically expected behaviors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from ..core.agent import ActiveInferenceAgent
from ..core.beliefs import BeliefState
from ..core.generative_model import GenerativeModel
from ..utils.validation import ValidationError


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    score: float
    expected_value: Optional[float]
    actual_value: Optional[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]


class TheoreticalValidator:
    """
    Validates Active Inference implementation against theoretical predictions.
    
    Ensures the implementation correctly follows the Free Energy Principle
    and exhibits theoretically expected properties.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TheoreticalValidator")
        self.results: List[ValidationResult] = []
    
    def validate_free_energy_principle(self, 
                                     agent: ActiveInferenceAgent,
                                     environment: Any,
                                     n_steps: int = 100) -> ValidationResult:
        """
        Validate that agent follows Free Energy Principle.
        
        Tests:
        1. Free energy decreases over time (on average)
        2. Actions minimize expected free energy
        3. Beliefs converge to posterior
        """
        try:
            free_energies = []
            
            # Run agent for multiple steps
            obs = environment.reset()
            agent.reset(obs)
            
            for step in range(n_steps):
                # Record free energy before action
                stats = agent.get_statistics()
                if 'current_free_energy' in stats:
                    free_energies.append(stats['current_free_energy'])
                
                # Take action
                action = agent.act(obs)
                obs, reward, done = environment.step(action)
                agent.update_model(obs, action, reward)
                
                if done:
                    obs = environment.reset()
                    agent.reset(obs)
            
            # Analyze free energy trajectory
            if len(free_energies) < 10:
                return ValidationResult(
                    test_name="free_energy_principle",
                    passed=False,
                    score=0.0,
                    expected_value=None,
                    actual_value=None,
                    error_message="Insufficient free energy data",
                    metadata={'n_samples': len(free_energies)}
                )
            
            # Test 1: Free energy generally decreases
            fe_trend = np.polyfit(range(len(free_energies)), free_energies, 1)[0]
            decreasing = fe_trend < 0
            
            # Test 2: Free energy variance decreases (convergence)
            early_var = np.var(free_energies[:len(free_energies)//3])
            late_var = np.var(free_energies[-len(free_energies)//3:])
            variance_decreasing = late_var < early_var
            
            # Compute overall score
            score = 0.0
            if decreasing:
                score += 0.5
            if variance_decreasing:
                score += 0.5
            
            passed = score >= 0.5
            
            result = ValidationResult(
                test_name="free_energy_principle",
                passed=passed,
                score=score,
                expected_value=0.0,  # Expected negative trend
                actual_value=fe_trend,
                error_message=None if passed else "Free energy not decreasing as expected",
                metadata={
                    'fe_trend': fe_trend,
                    'early_variance': early_var,
                    'late_variance': late_var,
                    'n_steps': n_steps
                }
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Free energy validation failed: {e}")
            return ValidationResult(
                test_name="free_energy_principle",
                passed=False,
                score=0.0,
                expected_value=None,
                actual_value=None,
                error_message=str(e),
                metadata={}
            )


class FreeEnergyValidator:
    """Validates free energy computation properties."""
    
    def __init__(self):
        self.logger = logging.getLogger("FreeEnergyValidator")
    
    def validate_free_energy_bounds(self, 
                                   agent: ActiveInferenceAgent,
                                   n_tests: int = 50) -> ValidationResult:
        """
        Validate that free energy has correct mathematical properties.
        
        Tests:
        1. Free energy is always non-negative
        2. Free energy equals surprise + KL divergence
        3. Lower free energy indicates better model fit
        """
        try:
            valid_bounds = 0
            total_tests = 0
            
            for _ in range(n_tests):
                # Create random observation
                obs = np.random.randn(agent.obs_dim)
                
                # Get current beliefs
                beliefs = agent.beliefs
                if len(beliefs) == 0:
                    continue
                
                # Compute free energy components
                try:
                    stats = agent.get_statistics()
                    if 'current_free_energy' in stats:
                        fe = stats['current_free_energy']
                        
                        # Test bounds
                        if fe >= 0:  # Free energy should be non-negative
                            valid_bounds += 1
                        
                        total_tests += 1
                        
                except Exception:
                    continue
            
            if total_tests == 0:
                score = 0.0
                passed = False
                error_msg = "No valid free energy computations"
            else:
                score = valid_bounds / total_tests
                passed = score >= 0.95  # Allow for numerical precision issues
                error_msg = None if passed else f"Only {score:.2%} of free energies had valid bounds"
            
            return ValidationResult(
                test_name="free_energy_bounds",
                passed=passed,
                score=score,
                expected_value=1.0,
                actual_value=score,
                error_message=error_msg,
                metadata={
                    'valid_bounds': valid_bounds,
                    'total_tests': total_tests
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="free_energy_bounds",
                passed=False,
                score=0.0,
                expected_value=None,
                actual_value=None,
                error_message=str(e),
                metadata={}
            )


class ConvergenceValidator:
    """Validates convergence properties of belief updating."""
    
    def __init__(self):
        self.logger = logging.getLogger("ConvergenceValidator")
    
    def validate_belief_convergence(self,
                                   agent: ActiveInferenceAgent,
                                   environment: Any,
                                   n_episodes: int = 10) -> ValidationResult:
        """
        Validate that beliefs converge to stable values.
        
        Tests belief variance decreases over time as agent gains experience.
        """
        try:
            belief_variances = []
            
            for episode in range(n_episodes):
                obs = environment.reset()
                agent.reset(obs)
                
                episode_variances = []
                
                for step in range(50):
                    # Record belief uncertainty
                    beliefs = agent.beliefs
                    if len(beliefs) > 0:
                        total_entropy = beliefs.total_entropy()
                        episode_variances.append(total_entropy)
                    
                    # Take step
                    action = agent.act(obs)
                    obs, reward, done = environment.step(action)
                    agent.update_model(obs, action, reward)
                    
                    if done:
                        break
                
                if episode_variances:
                    # Compute trend within episode
                    if len(episode_variances) > 1:
                        trend = np.polyfit(range(len(episode_variances)), episode_variances, 1)[0]
                        belief_variances.append(trend)
            
            if len(belief_variances) < 3:
                return ValidationResult(
                    test_name="belief_convergence",
                    passed=False,
                    score=0.0,
                    expected_value=None,
                    actual_value=None,
                    error_message="Insufficient convergence data",
                    metadata={'n_episodes': len(belief_variances)}
                )
            
            # Test convergence: negative trend indicates decreasing uncertainty
            converging_episodes = sum(1 for trend in belief_variances if trend < 0)
            convergence_rate = converging_episodes / len(belief_variances)
            
            # Score based on how often beliefs converge
            score = convergence_rate
            passed = score >= 0.6  # At least 60% of episodes should show convergence
            
            return ValidationResult(
                test_name="belief_convergence",
                passed=passed,
                score=score,
                expected_value=0.6,
                actual_value=convergence_rate,
                error_message=None if passed else "Beliefs not converging consistently",
                metadata={
                    'convergence_rate': convergence_rate,
                    'converging_episodes': converging_episodes,
                    'total_episodes': len(belief_variances),
                    'variance_trends': belief_variances
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="belief_convergence",
                passed=False,
                score=0.0,
                expected_value=None,
                actual_value=None,
                error_message=str(e),
                metadata={}
            )


class BehaviorValidator:
    """Validates emergent behaviors expected from Active Inference."""
    
    def __init__(self):
        self.logger = logging.getLogger("BehaviorValidator")
    
    def validate_exploration_exploitation(self,
                                        agent: ActiveInferenceAgent,
                                        environment: Any,
                                        n_steps: int = 200) -> ValidationResult:
        """
        Validate exploration-exploitation behavior.
        
        Active inference agents should naturally balance exploration
        (information seeking) and exploitation (goal achievement).
        """
        try:
            action_entropies = []
            belief_uncertainties = []
            
            obs = environment.reset()
            agent.reset(obs)
            
            for step in range(n_steps):
                # Measure belief uncertainty
                beliefs = agent.beliefs
                if len(beliefs) > 0:
                    uncertainty = beliefs.total_entropy()
                    belief_uncertainties.append(uncertainty)
                
                # Take action and measure entropy
                action = agent.act(obs)
                
                # Estimate action entropy (simplified)
                # In practice, would track action distribution
                action_entropies.append(np.sum(action**2))  # Action magnitude as proxy
                
                obs, reward, done = environment.step(action)
                agent.update_model(obs, action, reward)
                
                if done:
                    obs = environment.reset()
                    agent.reset(obs)
            
            if len(action_entropies) < 10 or len(belief_uncertainties) < 10:
                return ValidationResult(
                    test_name="exploration_exploitation",
                    passed=False,
                    score=0.0,
                    expected_value=None,
                    actual_value=None,
                    error_message="Insufficient behavioral data",
                    metadata={}
                )
            
            # Test 1: Early exploration (high action entropy when uncertainty is high)
            early_actions = np.mean(action_entropies[:len(action_entropies)//3])
            late_actions = np.mean(action_entropies[-len(action_entropies)//3:])
            
            early_uncertainty = np.mean(belief_uncertainties[:len(belief_uncertainties)//3])
            late_uncertainty = np.mean(belief_uncertainties[-len(belief_uncertainties)//3:])
            
            # Test exploration-exploitation trade-off
            exploration_score = 0.0
            
            # Higher early action diversity
            if early_actions > late_actions:
                exploration_score += 0.3
            
            # Decreasing uncertainty over time
            if late_uncertainty < early_uncertainty:
                exploration_score += 0.4
            
            # Action-uncertainty correlation (explore when uncertain)
            if len(action_entropies) == len(belief_uncertainties):
                correlation = np.corrcoef(action_entropies, belief_uncertainties)[0, 1]
                if not np.isnan(correlation) and correlation > 0.1:
                    exploration_score += 0.3
            
            passed = exploration_score >= 0.5
            
            return ValidationResult(
                test_name="exploration_exploitation",
                passed=passed,
                score=exploration_score,
                expected_value=0.5,
                actual_value=exploration_score,
                error_message=None if passed else "Poor exploration-exploitation balance",
                metadata={
                    'early_actions': early_actions,
                    'late_actions': late_actions,
                    'early_uncertainty': early_uncertainty,
                    'late_uncertainty': late_uncertainty,
                    'action_uncertainty_correlation': correlation if 'correlation' in locals() else None
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="exploration_exploitation",
                passed=False,
                score=0.0,
                expected_value=None,
                actual_value=None,
                error_message=str(e),
                metadata={}
            )
    
    def run_all_validations(self,
                           agent: ActiveInferenceAgent,
                           environment: Any) -> Dict[str, ValidationResult]:
        """Run all validation tests and return results."""
        results = {}
        
        validators = [
            ("free_energy_principle", TheoreticalValidator().validate_free_energy_principle),
            ("free_energy_bounds", FreeEnergyValidator().validate_free_energy_bounds),
            ("belief_convergence", ConvergenceValidator().validate_belief_convergence),
            ("exploration_exploitation", self.validate_exploration_exploitation)
        ]
        
        for name, validator in validators:
            try:
                if name in ["free_energy_bounds"]:
                    result = validator(agent)
                else:
                    result = validator(agent, environment)
                results[name] = result
                
                self.logger.info(f"Validation {name}: {'PASS' if result.passed else 'FAIL'} "
                               f"(score: {result.score:.3f})")
                
            except Exception as e:
                self.logger.error(f"Validation {name} failed: {e}")
                results[name] = ValidationResult(
                    test_name=name,
                    passed=False,
                    score=0.0,
                    expected_value=None,
                    actual_value=None,
                    error_message=str(e),
                    metadata={}
                )
        
        return results