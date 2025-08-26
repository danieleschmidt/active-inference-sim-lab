"""Advanced Cognitive Architectures for Active Inference.

This module implements novel cognitive architectures that push the boundaries of
Active Inference research, including:
- Hybrid Symbolic-Connectionist Active Inference (HSCAI)
- Compositional Active Inference for structured reasoning
- Emergent Communication through Active Inference
- Causal Active Inference with interventional planning
- Continual Active Inference for lifelong learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from ..core.agent import ActiveInferenceAgent
from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from ..core.free_energy import FreeEnergyObjective
from .advanced_algorithms import HierarchicalTemporalActiveInference


@dataclass
class SymbolicRule:
    """Represents a symbolic rule in the hybrid architecture."""
    condition: str
    action: str
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.5


@dataclass
class CausalRelation:
    """Represents a causal relationship between variables."""
    cause: str
    effect: str
    strength: float
    intervention_count: int = 0
    observed_outcomes: List[float] = field(default_factory=list)


class HybridSymbolicConnectionistAI:
    """
    Hybrid Symbolic-Connectionist Active Inference (HSCAI).
    
    Combines symbolic reasoning with neural Active Inference for
    structured decision-making and interpretable learning.
    """
    
    def __init__(self, 
                 base_agent: ActiveInferenceAgent,
                 max_symbolic_rules: int = 100,
                 rule_learning_rate: float = 0.1):
        """
        Initialize Hybrid Symbolic-Connectionist AI.
        
        Args:
            base_agent: Base Active Inference agent
            max_symbolic_rules: Maximum number of symbolic rules
            rule_learning_rate: Learning rate for rule acquisition
        """
        self.base_agent = base_agent
        self.max_symbolic_rules = max_symbolic_rules
        self.rule_learning_rate = rule_learning_rate
        self.logger = logging.getLogger("HSCAI")
        
        # Symbolic components
        self.symbolic_rules: List[SymbolicRule] = []
        self.symbolic_memory: Dict[str, Any] = {}
        self.concept_hierarchy: Dict[str, List[str]] = {}
        
        # Integration components
        self.symbol_activation_threshold = 0.7
        self.neural_symbolic_gate = 0.5  # Balance between neural and symbolic
        
        # Performance tracking
        self.hybrid_performance = {
            'symbolic_activations': [],
            'neural_dominance': [],
            'rule_acquisitions': [],
            'symbolic_accuracy': [],
            'integration_quality': []
        }
        
        # Initialize with basic symbolic knowledge
        self._initialize_symbolic_knowledge()
    
    def _initialize_symbolic_knowledge(self):
        """Initialize basic symbolic knowledge base."""
        # Basic rules for common situations
        basic_rules = [
            SymbolicRule("high_uncertainty", "explore", 0.8),
            SymbolicRule("low_uncertainty", "exploit", 0.7),
            SymbolicRule("high_prediction_error", "update_model", 0.9),
            SymbolicRule("goal_achieved", "maintain_action", 0.6),
            SymbolicRule("repeated_failure", "change_strategy", 0.8)
        ]
        
        self.symbolic_rules.extend(basic_rules)
        
        # Basic concept hierarchy
        self.concept_hierarchy = {
            'exploration': ['random_action', 'information_seeking', 'novelty_pursuit'],
            'exploitation': ['greedy_action', 'goal_pursuit', 'reward_maximization'],
            'learning': ['model_update', 'belief_revision', 'rule_acquisition']
        }
        
        self.logger.info(f"Initialized hybrid architecture with {len(basic_rules)} basic rules")
    
    def hybrid_decision_making(self, 
                             observation: np.ndarray,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make decisions using hybrid symbolic-connectionist approach.
        
        Args:
            observation: Current observation
            context: Additional context information
            
        Returns:
            Decision result with symbolic and neural components
        """
        start_time = time.time()
        context = context or {}
        
        # Neural processing
        neural_result = self._neural_processing(observation)
        
        # Symbolic processing
        symbolic_result = self._symbolic_processing(observation, context, neural_result)
        
        # Integration
        integrated_decision = self._integrate_symbolic_neural(neural_result, symbolic_result)
        
        # Learning: Update rules based on performance
        self._update_symbolic_rules(integrated_decision, context)
        
        # Record performance metrics
        self._record_hybrid_performance(neural_result, symbolic_result, integrated_decision)
        
        processing_time = time.time() - start_time
        
        return {
            'neural_result': neural_result,
            'symbolic_result': symbolic_result,
            'integrated_decision': integrated_decision,
            'processing_time': processing_time,
            'symbolic_rules_activated': len(symbolic_result['activated_rules']),
            'neural_confidence': neural_result.get('confidence', 0.5),
            'symbolic_confidence': symbolic_result.get('confidence', 0.5)
        }
    
    def _neural_processing(self, observation: np.ndarray) -> Dict[str, Any]:
        """Process observation through neural Active Inference."""
        try:
            # Use base agent for neural processing
            action = self.base_agent.act(observation)
            stats = self.base_agent.get_statistics()
            
            # Extract neural confidence from beliefs
            beliefs = self.base_agent.beliefs.get_all_beliefs()
            neural_confidence = self._compute_neural_confidence(beliefs)
            
            return {
                'action': action,
                'beliefs': beliefs,
                'free_energy': stats.get('current_free_energy', 0),
                'confidence': neural_confidence,
                'uncertainty': 1.0 - neural_confidence,
                'prediction_error': stats.get('current_free_energy', 0)  # Simplified
            }
            
        except Exception as e:
            self.logger.error(f"Neural processing failed: {e}")
            return {
                'action': np.zeros(self.base_agent.action_dim),
                'error': str(e),
                'confidence': 0.0,
                'uncertainty': 1.0
            }
    
    def _compute_neural_confidence(self, beliefs: Dict[str, Belief]) -> float:
        """Compute confidence from neural beliefs."""
        if not beliefs:
            return 0.0
        
        # Average inverse of variances as confidence measure
        confidences = []
        for belief in beliefs.values():
            if hasattr(belief, 'variance') and belief.variance is not None:
                variance = np.mean(belief.variance) if isinstance(belief.variance, np.ndarray) else belief.variance
                confidence = 1.0 / (1.0 + variance)
                confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.5
    
    def _symbolic_processing(self, 
                           observation: np.ndarray,
                           context: Dict[str, Any],
                           neural_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process information through symbolic reasoning."""
        # Extract symbolic features from neural state and context
        symbolic_features = self._extract_symbolic_features(observation, context, neural_result)
        
        # Find matching symbolic rules
        activated_rules = self._match_symbolic_rules(symbolic_features)
        
        # Apply symbolic reasoning
        symbolic_recommendation = self._apply_symbolic_reasoning(activated_rules, symbolic_features)
        
        # Compute symbolic confidence
        symbolic_confidence = self._compute_symbolic_confidence(activated_rules)
        
        return {
            'symbolic_features': symbolic_features,
            'activated_rules': activated_rules,
            'recommendation': symbolic_recommendation,
            'confidence': symbolic_confidence,
            'reasoning_chain': self._generate_reasoning_chain(activated_rules)
        }
    
    def _extract_symbolic_features(self, 
                                 observation: np.ndarray,
                                 context: Dict[str, Any],
                                 neural_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract symbolic features from neural state and context."""
        features = {}
        
        # From neural result
        if 'uncertainty' in neural_result:
            if neural_result['uncertainty'] > 0.7:
                features['high_uncertainty'] = True
            elif neural_result['uncertainty'] < 0.3:
                features['low_uncertainty'] = True
        
        if 'prediction_error' in neural_result:
            if neural_result['prediction_error'] > 1.0:
                features['high_prediction_error'] = True
            elif neural_result['prediction_error'] < 0.3:
                features['low_prediction_error'] = True
        
        # From context
        if 'goal_achieved' in context:
            features['goal_achieved'] = context['goal_achieved']
        
        if 'recent_failures' in context and context['recent_failures'] > 3:
            features['repeated_failure'] = True
        
        # From observation patterns
        obs_norm = np.linalg.norm(observation)
        if obs_norm > 2.0:
            features['high_stimulation'] = True
        elif obs_norm < 0.5:
            features['low_stimulation'] = True
        
        # Temporal features
        if hasattr(self, 'previous_observation'):
            change_magnitude = np.linalg.norm(observation - self.previous_observation)
            if change_magnitude > 1.0:
                features['rapid_change'] = True
            elif change_magnitude < 0.1:
                features['stable_environment'] = True
        
        self.previous_observation = observation.copy()
        
        return features
    
    def _match_symbolic_rules(self, features: Dict[str, Any]) -> List[SymbolicRule]:
        """Find symbolic rules that match current features."""
        activated_rules = []
        
        for rule in self.symbolic_rules:
            # Simple string matching for rule conditions
            if rule.condition in features and features[rule.condition]:
                activated_rules.append(rule)
                rule.usage_count += 1
        
        # Sort by confidence
        activated_rules.sort(key=lambda r: r.confidence, reverse=True)
        
        return activated_rules
    
    def _apply_symbolic_reasoning(self, 
                                activated_rules: List[SymbolicRule],
                                features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply symbolic reasoning using activated rules."""
        if not activated_rules:
            return {'action_type': 'neural_default', 'reasoning': 'no_rules_matched'}
        
        # Use highest confidence rule
        best_rule = activated_rules[0]
        
        # Apply rule action
        recommendation = {
            'action_type': best_rule.action,
            'rule_used': best_rule,
            'reasoning': f"Applied rule: {best_rule.condition} -> {best_rule.action}",
            'rule_confidence': best_rule.confidence
        }
        
        # Check for rule conflicts
        conflicting_rules = [r for r in activated_rules[1:] if r.action != best_rule.action]
        if conflicting_rules:
            recommendation['conflicts'] = conflicting_rules
            recommendation['conflict_resolution'] = 'highest_confidence'
        
        return recommendation
    
    def _compute_symbolic_confidence(self, activated_rules: List[SymbolicRule]) -> float:
        """Compute confidence in symbolic reasoning."""
        if not activated_rules:
            return 0.0
        
        # Weighted average of rule confidences
        total_confidence = sum(rule.confidence for rule in activated_rules)
        return total_confidence / len(activated_rules)
    
    def _generate_reasoning_chain(self, activated_rules: List[SymbolicRule]) -> List[str]:
        """Generate interpretable reasoning chain."""
        chain = []
        for rule in activated_rules:
            chain.append(f"IF {rule.condition} THEN {rule.action} (conf: {rule.confidence:.2f})")
        return chain
    
    def _integrate_symbolic_neural(self, 
                                 neural_result: Dict[str, Any],
                                 symbolic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate symbolic and neural recommendations."""
        neural_confidence = neural_result.get('confidence', 0.5)
        symbolic_confidence = symbolic_result.get('confidence', 0.5)
        
        # Compute integration weights
        total_confidence = neural_confidence + symbolic_confidence
        if total_confidence > 0:
            neural_weight = neural_confidence / total_confidence
            symbolic_weight = symbolic_confidence / total_confidence
        else:
            neural_weight = symbolic_weight = 0.5
        
        # Apply gating mechanism
        neural_weight *= (1 - self.neural_symbolic_gate)
        symbolic_weight *= self.neural_symbolic_gate
        
        # Integrated action
        neural_action = neural_result.get('action', np.zeros(2))
        
        # Convert symbolic recommendation to action modification
        symbolic_modifier = self._symbolic_to_action_modifier(symbolic_result['recommendation'])
        
        # Combine actions
        integrated_action = neural_weight * neural_action + symbolic_weight * symbolic_modifier
        
        # Decision explanation
        explanation = {
            'neural_weight': neural_weight,
            'symbolic_weight': symbolic_weight,
            'neural_confidence': neural_confidence,
            'symbolic_confidence': symbolic_confidence,
            'dominant_system': 'symbolic' if symbolic_weight > neural_weight else 'neural',
            'integration_quality': min(neural_confidence, symbolic_confidence)
        }
        
        return {
            'integrated_action': integrated_action,
            'explanation': explanation,
            'neural_component': neural_action * neural_weight,
            'symbolic_component': symbolic_modifier * symbolic_weight
        }
    
    def _symbolic_to_action_modifier(self, symbolic_recommendation: Dict[str, Any]) -> np.ndarray:
        """Convert symbolic recommendation to action space modifier."""
        action_type = symbolic_recommendation.get('action_type', 'neural_default')
        
        # Map symbolic actions to numerical modifiers
        action_mappings = {
            'explore': np.array([0.5, 0.5]),  # Encourage exploration
            'exploit': np.array([0.1, 0.1]),  # Conservative actions
            'update_model': np.array([0.0, 0.0]),  # No action change, focus on learning
            'maintain_action': np.array([1.0, 1.0]),  # Amplify current action
            'change_strategy': np.array([-0.5, -0.5]),  # Opposite direction
            'random_action': np.random.randn(2) * 0.3,
            'neural_default': np.array([0.0, 0.0])
        }
        
        return action_mappings.get(action_type, np.array([0.0, 0.0]))
    
    def _update_symbolic_rules(self, decision_result: Dict[str, Any], context: Dict[str, Any]):
        """Update symbolic rules based on performance feedback."""
        if 'symbolic_result' not in decision_result:
            return
        
        activated_rules = decision_result['symbolic_result'].get('activated_rules', [])
        performance_feedback = context.get('performance_feedback', 0.0)  # -1 to 1
        
        # Update rule confidences based on feedback
        for rule in activated_rules:
            if performance_feedback > 0:
                # Positive feedback: increase confidence
                rule.confidence = min(1.0, rule.confidence + self.rule_learning_rate * performance_feedback)
                rule.success_rate = 0.9 * rule.success_rate + 0.1 * 1.0
            else:
                # Negative feedback: decrease confidence
                rule.confidence = max(0.1, rule.confidence + self.rule_learning_rate * performance_feedback)
                rule.success_rate = 0.9 * rule.success_rate + 0.1 * 0.0
        
        # Rule acquisition: create new rules if current ones are insufficient
        if performance_feedback < -0.5 and len(self.symbolic_rules) < self.max_symbolic_rules:
            self._attempt_rule_acquisition(decision_result, context)
    
    def _attempt_rule_acquisition(self, decision_result: Dict[str, Any], context: Dict[str, Any]):
        """Attempt to acquire new symbolic rules from experience."""
        symbolic_features = decision_result['symbolic_result'].get('symbolic_features', {})
        
        # Create new rule from current situation
        if symbolic_features:
            # Find the most salient feature
            feature_name = next(iter(symbolic_features.keys()))
            
            # Create new rule with low initial confidence
            new_action = self._generate_novel_action_type()
            new_rule = SymbolicRule(
                condition=feature_name,
                action=new_action,
                confidence=0.3  # Start with low confidence
            )
            
            self.symbolic_rules.append(new_rule)
            self.hybrid_performance['rule_acquisitions'].append({
                'rule': new_rule,
                'context': context,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Acquired new rule: {feature_name} -> {new_action}")
    
    def _generate_novel_action_type(self) -> str:
        """Generate a novel action type for rule acquisition."""
        existing_actions = set(rule.action for rule in self.symbolic_rules)
        novel_actions = ['investigate', 'pause', 'amplify', 'reduce', 'alternate', 'combine']
        
        for action in novel_actions:
            if action not in existing_actions:
                return action
        
        return f"novel_action_{len(self.symbolic_rules)}"
    
    def _record_hybrid_performance(self, neural_result: Dict[str, Any], 
                                 symbolic_result: Dict[str, Any],
                                 integrated_decision: Dict[str, Any]):
        """Record performance metrics for hybrid system."""
        # Symbolic activation frequency
        n_symbolic_rules = len(symbolic_result.get('activated_rules', []))
        self.hybrid_performance['symbolic_activations'].append(n_symbolic_rules)
        
        # Neural vs symbolic dominance
        explanation = integrated_decision.get('explanation', {})
        neural_dominant = explanation.get('dominant_system') == 'neural'
        self.hybrid_performance['neural_dominance'].append(neural_dominant)
        
        # Integration quality
        integration_quality = explanation.get('integration_quality', 0.5)
        self.hybrid_performance['integration_quality'].append(integration_quality)
        
        # Symbolic accuracy (would need external feedback in practice)
        symbolic_confidence = symbolic_result.get('confidence', 0.5)
        self.hybrid_performance['symbolic_accuracy'].append(symbolic_confidence)
    
    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for hybrid system."""
        return {
            'n_symbolic_rules': len(self.symbolic_rules),
            'avg_symbolic_activations': np.mean(self.hybrid_performance['symbolic_activations']) if self.hybrid_performance['symbolic_activations'] else 0,
            'neural_dominance_rate': np.mean(self.hybrid_performance['neural_dominance']) if self.hybrid_performance['neural_dominance'] else 0.5,
            'avg_integration_quality': np.mean(self.hybrid_performance['integration_quality']) if self.hybrid_performance['integration_quality'] else 0.5,
            'avg_symbolic_accuracy': np.mean(self.hybrid_performance['symbolic_accuracy']) if self.hybrid_performance['symbolic_accuracy'] else 0.5,
            'rule_acquisition_rate': len(self.hybrid_performance['rule_acquisitions']) / max(1, len(self.hybrid_performance['symbolic_activations'])),
            'top_performing_rules': self._get_top_performing_rules(5),
            'concept_hierarchy_size': len(self.concept_hierarchy),
            'neural_symbolic_balance': self.neural_symbolic_gate
        }
    
    def _get_top_performing_rules(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing symbolic rules."""
        sorted_rules = sorted(self.symbolic_rules, key=lambda r: r.confidence * r.usage_count, reverse=True)
        
        return [{
            'condition': rule.condition,
            'action': rule.action,
            'confidence': rule.confidence,
            'usage_count': rule.usage_count,
            'success_rate': rule.success_rate
        } for rule in sorted_rules[:n]]


class CompositionalActiveInference:
    """
    Compositional Active Inference for structured reasoning.
    
    Enables compositional learning and reasoning by decomposing
    complex tasks into reusable components.
    """
    
    def __init__(self, base_agent: ActiveInferenceAgent):
        self.base_agent = base_agent
        self.logger = logging.getLogger("CompositionalAI")
        
        # Compositional components
        self.primitive_skills: Dict[str, Callable] = {}
        self.composite_skills: Dict[str, List[str]] = {}
        self.skill_relationships: Dict[str, List[str]] = {}
        
        # Composition learning
        self.composition_history: List[Dict[str, Any]] = []
        self.skill_usage_counts: Dict[str, int] = defaultdict(int)
        self.skill_success_rates: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Performance tracking
        self.compositional_performance = {
            'successful_compositions': [],
            'composition_complexity': [],
            'skill_reuse_rate': [],
            'emergent_behaviors': []
        }
        
        # Initialize primitive skills
        self._initialize_primitive_skills()
    
    def _initialize_primitive_skills(self):
        """Initialize basic primitive skills."""
        self.primitive_skills = {
            'move_forward': self._skill_move_forward,
            'move_backward': self._skill_move_backward,
            'turn_left': self._skill_turn_left,
            'turn_right': self._skill_turn_right,
            'observe': self._skill_observe,
            'wait': self._skill_wait,
            'explore': self._skill_explore,
            'approach_target': self._skill_approach_target
        }
        
        # Initialize success rates
        for skill in self.primitive_skills:
            self.skill_success_rates[skill] = 0.7  # Reasonable default
    
    def _skill_move_forward(self, observation: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
        """Primitive skill: move forward."""
        return np.array([magnitude, 0.0])
    
    def _skill_move_backward(self, observation: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
        """Primitive skill: move backward."""
        return np.array([-magnitude, 0.0])
    
    def _skill_turn_left(self, observation: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
        """Primitive skill: turn left."""
        return np.array([0.0, magnitude])
    
    def _skill_turn_right(self, observation: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
        """Primitive skill: turn right."""
        return np.array([0.0, -magnitude])
    
    def _skill_observe(self, observation: np.ndarray) -> np.ndarray:
        """Primitive skill: observe (no action)."""
        return np.array([0.0, 0.0])
    
    def _skill_wait(self, observation: np.ndarray) -> np.ndarray:
        """Primitive skill: wait."""
        return np.array([0.0, 0.0])
    
    def _skill_explore(self, observation: np.ndarray) -> np.ndarray:
        """Primitive skill: explore randomly."""
        return np.random.randn(2) * 0.5
    
    def _skill_approach_target(self, observation: np.ndarray) -> np.ndarray:
        """Primitive skill: approach target based on observation."""
        # Simple heuristic: move toward highest observation values
        if len(observation) >= 2:
            direction = observation[:2] / (np.linalg.norm(observation[:2]) + 1e-6)
            return direction * 0.5
        return np.array([0.1, 0.0])
    
    def compositional_reasoning(self, 
                              observation: np.ndarray,
                              goal_description: str,
                              max_composition_depth: int = 3) -> Dict[str, Any]:
        """
        Perform compositional reasoning to achieve goals.
        
        Args:
            observation: Current observation
            goal_description: Description of desired goal
            max_composition_depth: Maximum depth of skill composition
            
        Returns:
            Compositional reasoning result
        """
        start_time = time.time()
        
        # Analyze goal to determine required skills
        required_skills = self._analyze_goal_requirements(goal_description, observation)
        
        # Compose skills to achieve goal
        skill_composition = self._compose_skills(required_skills, max_composition_depth)
        
        # Execute composed skill sequence
        execution_result = self._execute_skill_composition(skill_composition, observation)
        
        # Learn from composition performance
        self._learn_from_composition(skill_composition, execution_result, goal_description)
        
        # Record performance metrics
        self._record_compositional_performance(skill_composition, execution_result)
        
        reasoning_time = time.time() - start_time
        
        return {
            'goal_description': goal_description,
            'required_skills': required_skills,
            'skill_composition': skill_composition,
            'execution_result': execution_result,
            'reasoning_time': reasoning_time,
            'composition_complexity': len(skill_composition.get('sequence', []))
        }
    
    def _analyze_goal_requirements(self, goal: str, observation: np.ndarray) -> List[str]:
        """Analyze goal to determine required skills."""
        # Simple keyword-based analysis
        goal_lower = goal.lower()
        required_skills = []
        
        # Movement goals
        if any(word in goal_lower for word in ['forward', 'ahead', 'advance']):
            required_skills.append('move_forward')
        if any(word in goal_lower for word in ['backward', 'back', 'retreat']):
            required_skills.append('move_backward')
        if any(word in goal_lower for word in ['left', 'turn left']):
            required_skills.append('turn_left')
        if any(word in goal_lower for word in ['right', 'turn right']):
            required_skills.append('turn_right')
        
        # Behavioral goals
        if any(word in goal_lower for word in ['explore', 'search', 'find']):
            required_skills.append('explore')
        if any(word in goal_lower for word in ['approach', 'reach', 'goto']):
            required_skills.append('approach_target')
        if any(word in goal_lower for word in ['observe', 'look', 'watch']):
            required_skills.append('observe')
        if any(word in goal_lower for word in ['wait', 'pause', 'stop']):
            required_skills.append('wait')
        
        # If no specific skills identified, use exploration
        if not required_skills:
            required_skills = ['explore', 'observe']
        
        return required_skills
    
    def _compose_skills(self, required_skills: List[str], max_depth: int) -> Dict[str, Any]:
        """Compose skills into executable sequence."""
        # Check if this is a known composite skill
        composition_key = tuple(sorted(required_skills))
        if composition_key in self.composite_skills:
            return {
                'type': 'composite',
                'sequence': self.composite_skills[composition_key],
                'is_learned': True
            }
        
        # Create new composition
        skill_sequence = []
        
        # Simple sequencing: order skills by success rate and dependencies
        skills_with_scores = []
        for skill in required_skills:
            success_rate = self.skill_success_rates[skill]
            usage_count = self.skill_usage_counts[skill]
            score = success_rate * (1 + 0.1 * usage_count)  # Bias toward successful, used skills
            skills_with_scores.append((skill, score))
        
        # Sort by score (highest first)
        skills_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build sequence
        for skill, score in skills_with_scores:
            skill_sequence.append({
                'skill_name': skill,
                'expected_success': score,
                'parameters': self._get_skill_parameters(skill)
            })
        
        composition = {
            'type': 'novel',
            'sequence': skill_sequence,
            'is_learned': False,
            'composition_depth': min(len(skill_sequence), max_depth)
        }
        
        # Store as composite skill for future use
        if len(skill_sequence) > 1:
            self.composite_skills[composition_key] = skill_sequence
        
        return composition
    
    def _get_skill_parameters(self, skill_name: str) -> Dict[str, Any]:
        """Get default parameters for a skill."""
        # Simple default parameters
        param_defaults = {
            'move_forward': {'magnitude': 1.0},
            'move_backward': {'magnitude': 0.8},
            'turn_left': {'magnitude': 0.5},
            'turn_right': {'magnitude': 0.5},
            'explore': {'randomness': 0.5},
            'approach_target': {'speed': 0.5}
        }
        return param_defaults.get(skill_name, {})
    
    def _execute_skill_composition(self, composition: Dict[str, Any], observation: np.ndarray) -> Dict[str, Any]:
        """Execute a composed skill sequence."""
        sequence = composition.get('sequence', [])
        execution_results = []
        cumulative_action = np.zeros(2)
        
        for i, skill_info in enumerate(sequence):
            if isinstance(skill_info, dict):
                skill_name = skill_info['skill_name']
                parameters = skill_info.get('parameters', {})
            else:
                skill_name = skill_info
                parameters = {}
            
            try:
                # Execute primitive skill
                if skill_name in self.primitive_skills:
                    skill_action = self.primitive_skills[skill_name](observation, **parameters)
                    cumulative_action += skill_action * (1.0 / len(sequence))  # Weight by sequence length
                    
                    execution_results.append({
                        'skill_name': skill_name,
                        'action': skill_action,
                        'success': True,
                        'step': i
                    })
                    
                    # Update skill usage
                    self.skill_usage_counts[skill_name] += 1
                    
                else:
                    execution_results.append({
                        'skill_name': skill_name,
                        'error': f"Skill {skill_name} not found",
                        'success': False,
                        'step': i
                    })
            
            except Exception as e:
                execution_results.append({
                    'skill_name': skill_name,
                    'error': str(e),
                    'success': False,
                    'step': i
                })
        
        # Compute overall execution success
        successful_steps = sum(1 for result in execution_results if result.get('success', False))
        success_rate = successful_steps / len(execution_results) if execution_results else 0
        
        return {
            'cumulative_action': cumulative_action,
            'execution_results': execution_results,
            'success_rate': success_rate,
            'completed_steps': successful_steps,
            'total_steps': len(execution_results)
        }
    
    def _learn_from_composition(self, composition: Dict[str, Any], 
                              execution_result: Dict[str, Any],
                              goal_description: str):
        """Learn from composition performance."""
        success_rate = execution_result.get('success_rate', 0.0)
        
        # Update skill success rates
        for result in execution_result.get('execution_results', []):
            skill_name = result.get('skill_name')
            if skill_name and result.get('success', False):
                # Positive update
                self.skill_success_rates[skill_name] = (
                    0.9 * self.skill_success_rates[skill_name] + 0.1 * 1.0
                )
            elif skill_name:
                # Negative update
                self.skill_success_rates[skill_name] = (
                    0.9 * self.skill_success_rates[skill_name] + 0.1 * 0.0
                )
        
        # Record composition history
        composition_record = {
            'goal': goal_description,
            'composition': composition,
            'execution_result': execution_result,
            'success_rate': success_rate,
            'timestamp': time.time()
        }
        
        self.composition_history.append(composition_record)
        
        # Detect emergent behaviors
        if success_rate > 0.8 and not composition.get('is_learned', False):
            self._detect_emergent_behavior(composition, execution_result, goal_description)
    
    def _detect_emergent_behavior(self, composition: Dict[str, Any], 
                                execution_result: Dict[str, Any],
                                goal_description: str):
        """Detect emergent behaviors from successful compositions."""
        skill_sequence = [step['skill_name'] for step in composition.get('sequence', []) 
                         if isinstance(step, dict) and 'skill_name' in step]
        
        # Check if this sequence creates an emergent behavior
        if len(skill_sequence) > 2:  # Complex enough to be emergent
            emergent_behavior = {
                'name': f"emergent_behavior_{len(self.compositional_performance['emergent_behaviors'])}",
                'skill_sequence': skill_sequence,
                'goal_achieved': goal_description,
                'success_rate': execution_result.get('success_rate', 0.0),
                'discovery_time': time.time()
            }
            
            self.compositional_performance['emergent_behaviors'].append(emergent_behavior)
            self.logger.info(f"Detected emergent behavior: {emergent_behavior['name']}")
    
    def _record_compositional_performance(self, composition: Dict[str, Any], 
                                        execution_result: Dict[str, Any]):
        """Record performance metrics for compositional system."""
        success_rate = execution_result.get('success_rate', 0.0)
        self.compositional_performance['successful_compositions'].append(success_rate > 0.7)
        
        composition_complexity = len(composition.get('sequence', []))
        self.compositional_performance['composition_complexity'].append(composition_complexity)
        
        # Skill reuse rate
        total_skills = len(self.primitive_skills) + len(self.composite_skills)
        used_skills = len(set(self.skill_usage_counts.keys()))
        reuse_rate = used_skills / total_skills if total_skills > 0 else 0
        self.compositional_performance['skill_reuse_rate'].append(reuse_rate)
    
    def get_compositional_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for compositional system."""
        return {
            'n_primitive_skills': len(self.primitive_skills),
            'n_composite_skills': len(self.composite_skills),
            'n_emergent_behaviors': len(self.compositional_performance['emergent_behaviors']),
            'avg_composition_success': np.mean(self.compositional_performance['successful_compositions']) if self.compositional_performance['successful_compositions'] else 0,
            'avg_composition_complexity': np.mean(self.compositional_performance['composition_complexity']) if self.compositional_performance['composition_complexity'] else 0,
            'skill_reuse_rate': np.mean(self.compositional_performance['skill_reuse_rate']) if self.compositional_performance['skill_reuse_rate'] else 0,
            'total_compositions_attempted': len(self.composition_history),
            'most_used_skills': self._get_most_used_skills(5),
            'best_performing_skills': self._get_best_performing_skills(5),
            'recent_emergent_behaviors': self.compositional_performance['emergent_behaviors'][-3:]
        }
    
    def _get_most_used_skills(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get most frequently used skills."""
        sorted_skills = sorted(self.skill_usage_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'skill': skill, 'usage_count': count, 'success_rate': self.skill_success_rates[skill]} 
                for skill, count in sorted_skills[:n]]
    
    def _get_best_performing_skills(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get best performing skills."""
        sorted_skills = sorted(self.skill_success_rates.items(), key=lambda x: x[1], reverse=True)
        return [{'skill': skill, 'success_rate': rate, 'usage_count': self.skill_usage_counts[skill]} 
                for skill, rate in sorted_skills[:n]]


class CausalActiveInference:
    """
    Causal Active Inference with interventional planning.
    
    Incorporates causal reasoning and interventional planning
    into the Active Inference framework.
    """
    
    def __init__(self, base_agent: ActiveInferenceAgent):
        self.base_agent = base_agent
        self.logger = logging.getLogger("CausalAI")
        
        # Causal model components
        self.causal_graph: Dict[str, List[str]] = {}  # variable -> parents
        self.causal_relations: List[CausalRelation] = []
        self.intervention_history: List[Dict[str, Any]] = []
        
        # Causal learning
        self.observation_history: List[Dict[str, float]] = []
        self.intervention_effects: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.causal_performance = {
            'causal_discovery_accuracy': [],
            'intervention_success_rate': [],
            'counterfactual_predictions': [],
            'causal_model_updates': []
        }
        
        # Initialize basic causal knowledge
        self._initialize_causal_model()
    
    def _initialize_causal_model(self):
        """Initialize basic causal model."""
        # Basic causal relations in Active Inference
        basic_relations = [
            CausalRelation("action", "observation", 0.7),
            CausalRelation("prediction_error", "belief_update", 0.8),
            CausalRelation("uncertainty", "exploration", 0.6),
            CausalRelation("goal", "action_selection", 0.9)
        ]
        
        self.causal_relations.extend(basic_relations)
        
        # Build causal graph
        for relation in self.causal_relations:
            if relation.effect not in self.causal_graph:
                self.causal_graph[relation.effect] = []
            self.causal_graph[relation.effect].append(relation.cause)
    
    def causal_reasoning_and_planning(self, 
                                    observation: np.ndarray,
                                    goal_state: Dict[str, float],
                                    intervention_budget: int = 3) -> Dict[str, Any]:
        """
        Perform causal reasoning and interventional planning.
        
        Args:
            observation: Current observation
            goal_state: Desired goal state
            intervention_budget: Maximum number of interventions to consider
            
        Returns:
            Causal reasoning and planning result
        """
        start_time = time.time()
        
        # Current state estimation
        current_state = self._estimate_current_state(observation)
        
        # Causal model update
        model_update = self._update_causal_model(current_state)
        
        # Plan interventions to achieve goal
        intervention_plan = self._plan_interventions(current_state, goal_state, intervention_budget)
        
        # Execute first intervention
        execution_result = self._execute_intervention(intervention_plan, observation)
        
        # Counterfactual reasoning
        counterfactual_analysis = self._counterfactual_reasoning(current_state, goal_state, intervention_plan)
        
        # Record causal learning
        self._record_causal_experience(current_state, intervention_plan, execution_result)
        
        reasoning_time = time.time() - start_time
        
        return {
            'current_state': current_state,
            'goal_state': goal_state,
            'causal_model_update': model_update,
            'intervention_plan': intervention_plan,
            'execution_result': execution_result,
            'counterfactual_analysis': counterfactual_analysis,
            'reasoning_time': reasoning_time
        }
    
    def _estimate_current_state(self, observation: np.ndarray) -> Dict[str, float]:
        """Estimate current state from observation."""
        # Simple state estimation from observation and agent statistics
        stats = self.base_agent.get_statistics()
        
        current_state = {
            'observation_magnitude': float(np.linalg.norm(observation)),
            'prediction_error': stats.get('current_free_energy', 0.0),
            'uncertainty': 1.0 - stats.get('belief_confidence', 0.5),
            'action_magnitude': float(np.linalg.norm(stats.get('last_action', np.zeros(2)))),
            'exploration_tendency': min(1.0, stats.get('current_free_energy', 0.0)),
            'goal_progress': np.random.random()  # Placeholder - would need actual goal tracking
        }
        
        return current_state
    
    def _update_causal_model(self, current_state: Dict[str, float]) -> Dict[str, Any]:
        """Update causal model based on new observations."""
        self.observation_history.append(current_state.copy())
        
        # Limit history size
        if len(self.observation_history) > 1000:
            self.observation_history = self.observation_history[-1000:]
        
        model_updates = []
        
        # Check for causal relations if we have enough data
        if len(self.observation_history) > 10:
            # Simple causal discovery: look for correlations with temporal precedence
            discovered_relations = self._discover_causal_relations()
            
            for relation in discovered_relations:
                # Add to causal model if not already present
                if not any(r.cause == relation.cause and r.effect == relation.effect 
                          for r in self.causal_relations):
                    self.causal_relations.append(relation)
                    model_updates.append(relation)
        
        # Update existing relation strengths
        updated_relations = self._update_causal_strengths()
        
        return {
            'new_relations': model_updates,
            'updated_relations': updated_relations,
            'total_relations': len(self.causal_relations),
            'model_confidence': self._compute_model_confidence()
        }
    
    def _discover_causal_relations(self) -> List[CausalRelation]:
        """Discover new causal relations from observation history."""
        discovered = []
        
        if len(self.observation_history) < 2:
            return discovered
        
        # Variables to check
        variables = list(self.observation_history[0].keys())
        
        # Check all pairs for causal relationships
        for cause_var in variables:
            for effect_var in variables:
                if cause_var == effect_var:
                    continue
                
                # Check if cause precedes effect and they're correlated
                correlation = self._compute_temporal_correlation(cause_var, effect_var)
                
                if abs(correlation) > 0.5:  # Threshold for causal discovery
                    strength = abs(correlation)
                    new_relation = CausalRelation(cause_var, effect_var, strength)
                    discovered.append(new_relation)
        
        return discovered
    
    def _compute_temporal_correlation(self, cause_var: str, effect_var: str, lag: int = 1) -> float:
        """Compute temporal correlation between variables."""
        if len(self.observation_history) <= lag:
            return 0.0
        
        cause_values = [obs[cause_var] for obs in self.observation_history[:-lag]]
        effect_values = [obs[effect_var] for obs in self.observation_history[lag:]]
        
        if not cause_values or not effect_values:
            return 0.0
        
        # Compute correlation
        cause_array = np.array(cause_values)
        effect_array = np.array(effect_values)
        
        if np.std(cause_array) == 0 or np.std(effect_array) == 0:
            return 0.0
        
        correlation = np.corrcoef(cause_array, effect_array)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _update_causal_strengths(self) -> List[CausalRelation]:
        """Update strengths of existing causal relations."""
        updated = []
        
        for relation in self.causal_relations:
            # Recompute strength based on recent data
            new_correlation = self._compute_temporal_correlation(relation.cause, relation.effect)
            
            # Update with momentum
            momentum = 0.9
            relation.strength = momentum * relation.strength + (1 - momentum) * abs(new_correlation)
            
            updated.append(relation)
        
        return updated
    
    def _compute_model_confidence(self) -> float:
        """Compute confidence in the causal model."""
        if not self.causal_relations:
            return 0.0
        
        # Average strength of causal relations
        avg_strength = np.mean([rel.strength for rel in self.causal_relations])
        
        # Penalize for too few or too many relations (sparse causal models are often better)
        n_relations = len(self.causal_relations)
        complexity_penalty = 1.0 / (1.0 + 0.1 * max(0, n_relations - 5))
        
        confidence = avg_strength * complexity_penalty
        return min(1.0, confidence)
    
    def _plan_interventions(self, 
                          current_state: Dict[str, float],
                          goal_state: Dict[str, float],
                          budget: int) -> Dict[str, Any]:
        """Plan interventions to achieve goal state."""
        # Find causal paths from controllable variables to goal variables
        controllable_vars = ['action_magnitude', 'exploration_tendency']  # Variables we can directly control
        goal_vars = list(goal_state.keys())
        
        causal_paths = []
        for goal_var in goal_vars:
            for control_var in controllable_vars:
                path = self._find_causal_path(control_var, goal_var)
                if path:
                    causal_paths.append({
                        'control_variable': control_var,
                        'goal_variable': goal_var,
                        'path': path,
                        'path_strength': self._compute_path_strength(path)
                    })
        
        # Select best interventions within budget
        interventions = self._select_interventions(causal_paths, current_state, goal_state, budget)
        
        return {
            'causal_paths': causal_paths,
            'selected_interventions': interventions,
            'intervention_budget': budget,
            'expected_effects': self._predict_intervention_effects(interventions, current_state)
        }
    
    def _find_causal_path(self, start_var: str, goal_var: str, max_depth: int = 3) -> List[str]:
        """Find causal path from start variable to goal variable."""
        # Simple breadth-first search in causal graph
        if start_var == goal_var:
            return [start_var]
        
        queue = [(start_var, [start_var])]
        visited = set([start_var])
        
        for _ in range(max_depth):
            if not queue:
                break
            
            current_var, path = queue.pop(0)
            
            # Find variables that are causally influenced by current_var
            for relation in self.causal_relations:
                if relation.cause == current_var and relation.effect not in visited:
                    new_path = path + [relation.effect]
                    
                    if relation.effect == goal_var:
                        return new_path
                    
                    queue.append((relation.effect, new_path))
                    visited.add(relation.effect)
        
        return []  # No path found
    
    def _compute_path_strength(self, path: List[str]) -> float:
        """Compute strength of a causal path."""
        if len(path) < 2:
            return 0.0
        
        total_strength = 1.0
        for i in range(len(path) - 1):
            cause = path[i]
            effect = path[i + 1]
            
            # Find relation strength
            relation_strength = 0.0
            for relation in self.causal_relations:
                if relation.cause == cause and relation.effect == effect:
                    relation_strength = relation.strength
                    break
            
            total_strength *= relation_strength
        
        return total_strength
    
    def _select_interventions(self, 
                            causal_paths: List[Dict[str, Any]],
                            current_state: Dict[str, float],
                            goal_state: Dict[str, float],
                            budget: int) -> List[Dict[str, Any]]:
        """Select best interventions within budget."""
        # Score interventions by expected impact and cost
        intervention_candidates = []
        
        for path_info in causal_paths:
            control_var = path_info['control_variable']
            goal_var = path_info['goal_variable']
            path_strength = path_info['path_strength']
            
            # Compute desired change in goal variable
            current_value = current_state.get(goal_var, 0.0)
            target_value = goal_state.get(goal_var, 0.0)
            desired_change = target_value - current_value
            
            # Compute required intervention strength
            intervention_strength = desired_change / (path_strength + 1e-6)
            
            intervention = {
                'variable': control_var,
                'target_variable': goal_var,
                'strength': abs(intervention_strength),
                'direction': 1.0 if intervention_strength > 0 else -1.0,
                'expected_impact': abs(desired_change) * path_strength,
                'cost': abs(intervention_strength),  # Cost proportional to intervention strength
                'path_info': path_info
            }
            
            intervention_candidates.append(intervention)
        
        # Sort by impact/cost ratio
        intervention_candidates.sort(key=lambda x: x['expected_impact'] / (x['cost'] + 1e-6), reverse=True)
        
        # Select interventions within budget
        selected_interventions = []
        total_cost = 0.0
        
        for intervention in intervention_candidates:
            if len(selected_interventions) < budget and total_cost + intervention['cost'] <= budget:
                selected_interventions.append(intervention)
                total_cost += intervention['cost']
        
        return selected_interventions
    
    def _predict_intervention_effects(self, interventions: List[Dict[str, Any]], 
                                    current_state: Dict[str, float]) -> Dict[str, float]:
        """Predict effects of planned interventions."""
        predicted_state = current_state.copy()
        
        for intervention in interventions:
            variable = intervention['variable']
            strength = intervention['strength'] * intervention['direction']
            
            # Apply direct intervention effect
            if variable in predicted_state:
                predicted_state[variable] += strength
            
            # Propagate effects through causal model
            self._propagate_causal_effects(variable, strength, predicted_state)
        
        return predicted_state
    
    def _propagate_causal_effects(self, 
                                changed_var: str,
                                change_magnitude: float,
                                state: Dict[str, float]):
        """Propagate causal effects through the model."""
        # Find all variables causally influenced by changed_var
        for relation in self.causal_relations:
            if relation.cause == changed_var:
                effect_var = relation.effect
                if effect_var in state:
                    # Propagate effect proportional to causal strength
                    effect_magnitude = change_magnitude * relation.strength
                    state[effect_var] += effect_magnitude
    
    def _execute_intervention(self, intervention_plan: Dict[str, Any], observation: np.ndarray) -> Dict[str, Any]:
        """Execute the first planned intervention."""
        interventions = intervention_plan.get('selected_interventions', [])
        
        if not interventions:
            # No interventions planned, use standard action
            action = self.base_agent.act(observation)
            return {
                'action': action,
                'intervention_applied': False,
                'intervention_type': 'none'
            }
        
        # Apply first intervention
        first_intervention = interventions[0]
        variable = first_intervention['variable']
        strength = first_intervention['strength'] * first_intervention['direction']
        
        # Convert intervention to action
        action = self._intervention_to_action(variable, strength, observation)
        
        # Record intervention
        intervention_record = {
            'variable': variable,
            'strength': strength,
            'target_variable': first_intervention['target_variable'],
            'timestamp': time.time()
        }
        
        self.intervention_history.append(intervention_record)
        
        return {
            'action': action,
            'intervention_applied': True,
            'intervention_type': variable,
            'intervention_strength': strength,
            'intervention_record': intervention_record
        }
    
    def _intervention_to_action(self, variable: str, strength: float, observation: np.ndarray) -> np.ndarray:
        """Convert causal intervention to action space."""
        # Map interventions to actions
        if variable == 'action_magnitude':
            # Directly control action magnitude
            base_action = self.base_agent.act(observation)
            action_magnitude = np.linalg.norm(base_action)
            if action_magnitude > 0:
                direction = base_action / action_magnitude
                new_magnitude = max(0.1, action_magnitude + strength)
                return direction * new_magnitude
            else:
                return np.array([strength, 0.0])
        
        elif variable == 'exploration_tendency':
            # Control exploration vs exploitation
            base_action = self.base_agent.act(observation)
            exploration_noise = np.random.randn(2) * abs(strength) * 0.5
            return base_action + exploration_noise
        
        else:
            # Default: use base agent action
            return self.base_agent.act(observation)
    
    def _counterfactual_reasoning(self, 
                                current_state: Dict[str, float],
                                goal_state: Dict[str, float],
                                intervention_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform counterfactual reasoning."""
        # Predict what would happen without interventions
        no_intervention_prediction = current_state.copy()
        
        # Add some random evolution (simplified)
        for var in no_intervention_prediction:
            no_intervention_prediction[var] += np.random.normal(0, 0.1)
        
        # Predict with interventions
        with_intervention_prediction = intervention_plan.get('expected_effects', current_state)
        
        # Compute counterfactual differences
        counterfactual_differences = {}
        for var in current_state:
            if var in with_intervention_prediction and var in no_intervention_prediction:
                diff = with_intervention_prediction[var] - no_intervention_prediction[var]
                counterfactual_differences[var] = diff
        
        # Compute goal achievement probability
        goal_achievement_no_intervention = self._compute_goal_achievement(no_intervention_prediction, goal_state)
        goal_achievement_with_intervention = self._compute_goal_achievement(with_intervention_prediction, goal_state)
        
        return {
            'no_intervention_prediction': no_intervention_prediction,
            'with_intervention_prediction': with_intervention_prediction,
            'counterfactual_differences': counterfactual_differences,
            'goal_achievement_improvement': goal_achievement_with_intervention - goal_achievement_no_intervention,
            'intervention_necessity': 1.0 - goal_achievement_no_intervention,
            'intervention_sufficiency': goal_achievement_with_intervention
        }
    
    def _compute_goal_achievement(self, predicted_state: Dict[str, float], goal_state: Dict[str, float]) -> float:
        """Compute how well the predicted state achieves the goal."""
        if not goal_state:
            return 1.0
        
        total_error = 0.0
        for var, target_value in goal_state.items():
            predicted_value = predicted_state.get(var, 0.0)
            error = abs(predicted_value - target_value)
            total_error += error
        
        # Convert error to achievement score
        achievement = 1.0 / (1.0 + total_error)
        return achievement
    
    def _record_causal_experience(self, 
                                current_state: Dict[str, float],
                                intervention_plan: Dict[str, Any],
                                execution_result: Dict[str, Any]):
        """Record causal experience for learning."""
        if execution_result.get('intervention_applied', False):
            intervention_record = execution_result.get('intervention_record', {})
            variable = intervention_record.get('variable')
            strength = intervention_record.get('strength', 0.0)
            
            if variable:
                # Record intervention effect (would need actual outcome measurement)
                # For now, use a placeholder
                effect_magnitude = strength * 0.7 + np.random.normal(0, 0.1)  # Simplified
                self.intervention_effects[variable].append(effect_magnitude)
    
    def get_causal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for causal system."""
        return {
            'n_causal_relations': len(self.causal_relations),
            'n_interventions_attempted': len(self.intervention_history),
            'causal_model_confidence': self._compute_model_confidence(),
            'avg_causal_strength': np.mean([rel.strength for rel in self.causal_relations]) if self.causal_relations else 0,
            'top_causal_relations': self._get_top_causal_relations(5),
            'intervention_variables': list(self.intervention_effects.keys()),
            'causal_graph_size': len(self.causal_graph),
            'observation_history_length': len(self.observation_history)
        }
    
    def _get_top_causal_relations(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top causal relations by strength."""
        sorted_relations = sorted(self.causal_relations, key=lambda r: r.strength, reverse=True)
        
        return [{
            'cause': rel.cause,
            'effect': rel.effect,
            'strength': rel.strength,
            'intervention_count': rel.intervention_count
        } for rel in sorted_relations[:n]]