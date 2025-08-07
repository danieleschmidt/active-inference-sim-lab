"""
Expected Free Energy computation for active inference planning.

This module implements the expected free energy functional that agents
minimize when selecting actions.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from ..core.free_energy import FreeEnergyObjective


@dataclass
class ExpectedFreeEnergyComponents:
    """Components of expected free energy."""
    epistemic_value: float  # Information gain (uncertainty reduction)
    pragmatic_value: float  # Goal achievement (preference satisfaction)  
    total: float
    
    def __post_init__(self):
        """Ensure total equals sum of components."""
        self.total = self.epistemic_value + self.pragmatic_value


class ExpectedFreeEnergy:
    """
    Expected Free Energy computation for active inference.
    
    Expected free energy drives action selection by combining:
    - Epistemic value: Information gain about hidden states
    - Pragmatic value: Achievement of preferred outcomes
    """
    
    def __init__(self,
                 epistemic_weight: float = 1.0,
                 pragmatic_weight: float = 1.0,
                 n_samples: int = 100):
        """
        Initialize expected free energy computation.
        
        Args:
            epistemic_weight: Weight for information gain term
            pragmatic_weight: Weight for goal achievement term
            n_samples: Number of Monte Carlo samples for expectations
        """
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        self.n_samples = n_samples
        
        # Default preferences (uniform over observations)
        self.preferences = None
    
    def set_preferences(self, preferences: np.ndarray) -> None:
        """
        Set agent preferences over observations.
        
        Args:
            preferences: Preferred observation distribution (log probabilities)
        """
        self.preferences = preferences
    
    def compute_epistemic_value(self,
                               action: np.ndarray,
                               beliefs: BeliefState,
                               model: GenerativeModel,
                               horizon: int = 1) -> float:
        """
        Compute epistemic value (information gain) of an action.
        
        Epistemic value measures how much an action reduces uncertainty
        about hidden states through observation.
        
        Args:
            action: Candidate action
            beliefs: Current beliefs
            model: Generative model
            horizon: Planning horizon
            
        Returns:
            Expected information gain
        """
        if len(beliefs) == 0:
            return 0.0
        
        # Sample current states from beliefs
        current_entropy = beliefs.total_entropy()
        
        # Predict future beliefs after taking action
        expected_entropy_after_action = 0.0
        
        for _ in range(self.n_samples):
            # Sample current state
            state_samples = []
            for name, belief in beliefs.get_all_beliefs().items():
                sample = belief.sample(1)[0]
                state_samples.append((name, sample))
            
            # Predict next state after action
            if state_samples:
                # Use first state for simplicity (could be extended)
                current_state = state_samples[0][1]
                predicted_next_state = model.predict_next_state(current_state, action)
                
                # Predict observation
                predicted_obs = np.random.randn(model.obs_dim) * 0.1  # Simplified
                
                # Compute expected entropy after observing
                # This is simplified - full implementation would update beliefs
                # and compute the resulting entropy
                entropy_reduction = current_entropy * 0.1  # Placeholder
                expected_entropy_after_action += current_entropy - entropy_reduction
        
        expected_entropy_after_action /= self.n_samples
        
        # Information gain = current entropy - expected future entropy
        information_gain = current_entropy - expected_entropy_after_action
        
        return max(0.0, information_gain) * self.epistemic_weight
    
    def compute_pragmatic_value(self,
                               action: np.ndarray,
                               beliefs: BeliefState,
                               model: GenerativeModel,
                               horizon: int = 1) -> float:
        """
        Compute pragmatic value (goal achievement) of an action.
        
        Pragmatic value measures how well an action leads to preferred outcomes.
        
        Args:
            action: Candidate action
            beliefs: Current beliefs
            model: Generative model
            horizon: Planning horizon
            
        Returns:
            Expected preference satisfaction
        """
        if self.preferences is None:
            # No preferences specified - return neutral value
            return 0.0
        
        expected_preference = 0.0
        
        for _ in range(self.n_samples):
            # Sample trajectory
            trajectory_preference = 0.0
            
            # Sample current state from beliefs
            if len(beliefs) > 0:
                # Use first belief for simplicity
                first_belief = list(beliefs.get_all_beliefs().values())[0]
                current_state = first_belief.sample(1)[0]
                
                # Simulate forward for horizon steps
                for step in range(horizon):
                    # Predict next state
                    next_state = model.predict_next_state(current_state, action)
                    
                    # Predict observation
                    predicted_obs = np.random.randn(model.obs_dim)  # Simplified
                    
                    # Evaluate against preferences
                    if step < len(self.preferences):
                        obs_preference = self.preferences[min(step, len(self.preferences)-1)]
                        # Simplified preference evaluation
                        preference_score = -0.5 * np.sum((predicted_obs - obs_preference)**2)
                        trajectory_preference += preference_score
                    
                    current_state = next_state
            
            expected_preference += trajectory_preference
        
        expected_preference /= self.n_samples
        
        return expected_preference * self.pragmatic_weight
    
    def compute_expected_free_energy(self,
                                   action: np.ndarray,
                                   beliefs: BeliefState,
                                   model: GenerativeModel,
                                   horizon: int = 1) -> ExpectedFreeEnergyComponents:
        """
        Compute expected free energy and its components for an action.
        
        Args:
            action: Candidate action
            beliefs: Current belief state
            model: Generative model
            horizon: Planning horizon
            
        Returns:
            Expected free energy components
        """
        epistemic = self.compute_epistemic_value(action, beliefs, model, horizon)
        pragmatic = self.compute_pragmatic_value(action, beliefs, model, horizon)
        
        # Expected free energy = -epistemic_value - pragmatic_value
        # (negative because we want to maximize both terms)
        return ExpectedFreeEnergyComponents(
            epistemic_value=epistemic,
            pragmatic_value=pragmatic,
            total=-(epistemic + pragmatic)  # Negative for minimization
        )
    
    def evaluate_action_set(self,
                           actions: List[np.ndarray],
                           beliefs: BeliefState,
                           model: GenerativeModel,
                           horizon: int = 1) -> List[ExpectedFreeEnergyComponents]:
        """
        Evaluate expected free energy for a set of actions.
        
        Args:
            actions: List of candidate actions
            beliefs: Current beliefs
            model: Generative model  
            horizon: Planning horizon
            
        Returns:
            List of expected free energy evaluations
        """
        evaluations = []
        
        for action in actions:
            efe = self.compute_expected_free_energy(action, beliefs, model, horizon)
            evaluations.append(efe)
        
        return evaluations
    
    def select_optimal_action(self,
                            actions: List[np.ndarray],
                            beliefs: BeliefState,
                            model: GenerativeModel,
                            horizon: int = 1,
                            temperature: float = 1.0) -> Tuple[np.ndarray, ExpectedFreeEnergyComponents]:
        """
        Select optimal action based on expected free energy.
        
        Args:
            actions: Candidate actions
            beliefs: Current beliefs
            model: Generative model
            horizon: Planning horizon
            temperature: Temperature for action selection
            
        Returns:
            Tuple of (optimal_action, efe_components)
        """
        if not actions:
            raise ValueError("No actions provided")
        
        # Evaluate all actions
        evaluations = self.evaluate_action_set(actions, beliefs, model, horizon)
        
        # Extract expected free energy values
        efe_values = np.array([eval.total for eval in evaluations])
        
        if temperature > 0:
            # Softmax selection (lower EFE = higher probability)
            action_probs = np.exp(-efe_values / temperature)
            action_probs /= action_probs.sum()
            
            # Sample action according to probabilities
            action_idx = np.random.choice(len(actions), p=action_probs)
        else:
            # Greedy selection (lowest EFE)
            action_idx = np.argmin(efe_values)
        
        return actions[action_idx], evaluations[action_idx]