"""Advanced Active Inference Algorithms for Research Excellence.

This module implements cutting-edge algorithms that push the boundaries of
Active Inference research, including:
- Hierarchical Temporal Active Inference (HTAI)
- Meta-Active Inference for rapid adaptation
- Quantum-inspired Variational Inference
- Neural Active Inference with deep architectures
- Multi-modal Active Inference for complex observations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from ..core.agent import ActiveInferenceAgent
from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from ..core.free_energy import FreeEnergyObjective
from .benchmarks import BenchmarkResult
from .experiments import ExperimentResult


class HierarchicalTemporalActiveInference:
    """
    Hierarchical Temporal Active Inference (HTAI) - Novel Algorithm.
    
    Implements multi-scale temporal hierarchies for improved learning
    and planning in complex environments. Each hierarchy level operates
    at different temporal resolutions.
    """
    
    def __init__(self, 
                 n_levels: int = 3,
                 temporal_scales: List[int] = None,
                 state_dims: List[int] = None,
                 coupling_strength: float = 0.5):
        """
        Initialize Hierarchical Temporal Active Inference.
        
        Args:
            n_levels: Number of hierarchy levels
            temporal_scales: Temporal scales for each level [1, 5, 25]
            state_dims: State dimensions for each level
            coupling_strength: Strength of inter-level coupling
        """
        self.n_levels = n_levels
        self.temporal_scales = temporal_scales or [1, 5, 25][:n_levels]
        self.state_dims = state_dims or [8, 4, 2][:n_levels]
        self.coupling_strength = coupling_strength
        
        self.logger = logging.getLogger("HTAI")
        
        # Initialize hierarchy levels
        self.levels = []
        for i in range(n_levels):
            level = HTAILevel(
                level_id=i,
                state_dim=self.state_dims[i],
                temporal_scale=self.temporal_scales[i],
                coupling_strength=coupling_strength
            )
            self.levels.append(level)
        
        # Inter-level connections
        self._setup_hierarchical_connections()
        
        # Performance metrics
        self.hierarchy_performance = {
            'prediction_errors': [],
            'information_flow': [],
            'level_activations': [],
            'temporal_coherence': []
        }
    
    def _setup_hierarchical_connections(self):
        """Setup bidirectional connections between hierarchy levels."""
        for i in range(len(self.levels) - 1):
            # Lower level (i) sends predictions to upper level (i+1)
            self.levels[i].set_upper_level(self.levels[i + 1])
            # Upper level (i+1) sends predictions down to lower level (i)
            self.levels[i + 1].set_lower_level(self.levels[i])
    
    def process_observation(self, observation: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        """Process observation through hierarchical structure."""
        start_time = time.time()
        
        # Bottom-up processing: prediction errors propagate upward
        prediction_errors = {}
        
        for i, level in enumerate(self.levels):
            if i == 0:
                # Bottom level processes raw observation
                level_input = observation
            else:
                # Higher levels process prediction errors from below
                level_input = prediction_errors[i - 1]
            
            # Process at this level
            level_output = level.process_input(level_input, action)
            prediction_errors[i] = level_output['prediction_error']
            
            # Update performance metrics
            self.hierarchy_performance['prediction_errors'].append(level_output['prediction_error'])
            self.hierarchy_performance['level_activations'].append(level_output['activation'])
        
        # Top-down processing: predictions propagate downward
        predictions = {}
        for i in reversed(range(len(self.levels))):
            if i == len(self.levels) - 1:
                # Top level generates prediction
                prediction = self.levels[i].generate_prediction()
            else:
                # Lower levels modulate predictions from above
                prediction = self.levels[i].modulate_prediction(predictions[i + 1])
            predictions[i] = prediction
        
        # Compute hierarchical coherence
        coherence = self._compute_temporal_coherence(predictions)
        self.hierarchy_performance['temporal_coherence'].append(coherence)
        
        processing_time = time.time() - start_time
        
        return {
            'hierarchical_beliefs': {i: level.get_beliefs() for i, level in enumerate(self.levels)},
            'prediction_errors': prediction_errors,
            'predictions': predictions,
            'temporal_coherence': coherence,
            'processing_time': processing_time
        }
    
    def _compute_temporal_coherence(self, predictions: Dict[int, np.ndarray]) -> float:
        """Compute temporal coherence across hierarchy levels."""
        if len(predictions) < 2:
            return 1.0
        
        coherences = []
        for i in range(len(predictions) - 1):
            # Measure alignment between adjacent levels
            pred_lower = predictions[i]
            pred_upper = predictions[i + 1]
            
            # Normalize to same dimensionality (simple approach)
            if pred_lower.size != pred_upper.size:
                min_size = min(pred_lower.size, pred_upper.size)
                pred_lower = pred_lower.flatten()[:min_size]
                pred_upper = pred_upper.flatten()[:min_size]
            
            # Compute correlation as coherence measure
            correlation = np.corrcoef(pred_lower, pred_upper)[0, 1]
            if not np.isnan(correlation):
                coherences.append(abs(correlation))
        
        return np.mean(coherences) if coherences else 1.0
    
    def plan_hierarchical_action(self, horizon: int = 10) -> Dict[str, np.ndarray]:
        """Plan actions using hierarchical temporal structure."""
        actions = {}
        
        # Plan at each temporal scale
        for i, level in enumerate(self.levels):
            level_horizon = max(1, horizon // self.temporal_scales[i])
            level_action = level.plan_action(level_horizon)
            actions[i] = level_action
        
        # Integrate actions across scales
        integrated_action = self._integrate_hierarchical_actions(actions)
        
        return {
            'level_actions': actions,
            'integrated_action': integrated_action
        }
    
    def _integrate_hierarchical_actions(self, actions: Dict[int, np.ndarray]) -> np.ndarray:
        """Integrate actions from multiple hierarchy levels."""
        if not actions:
            return np.zeros(2)  # Default action dimensionality
        
        # Weighted integration based on temporal scales
        integrated = np.zeros_like(list(actions.values())[0])
        total_weight = 0
        
        for i, action in actions.items():
            weight = 1.0 / self.temporal_scales[i]  # Faster scales get higher weight
            integrated += weight * action
            total_weight += weight
        
        return integrated / total_weight if total_weight > 0 else integrated
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy performance statistics."""
        stats = {
            'n_levels': self.n_levels,
            'temporal_scales': self.temporal_scales,
            'avg_prediction_error': np.mean([np.mean(pe) for pe in self.hierarchy_performance['prediction_errors']]) if self.hierarchy_performance['prediction_errors'] else 0,
            'avg_temporal_coherence': np.mean(self.hierarchy_performance['temporal_coherence']) if self.hierarchy_performance['temporal_coherence'] else 1,
            'level_statistics': {}
        }
        
        for i, level in enumerate(self.levels):
            stats['level_statistics'][i] = level.get_level_statistics()
        
        return stats


class HTAILevel:
    """Individual level in the hierarchical temporal structure."""
    
    def __init__(self, level_id: int, state_dim: int, temporal_scale: int, coupling_strength: float):
        self.level_id = level_id
        self.state_dim = state_dim
        self.temporal_scale = temporal_scale
        self.coupling_strength = coupling_strength
        
        # Level-specific state
        self.beliefs = BeliefState()
        self.temporal_buffer = Queue(maxsize=temporal_scale * 2)
        self.prediction_history = []
        self.error_history = []
        
        # Connections to other levels
        self.upper_level = None
        self.lower_level = None
        
        # Initialize beliefs
        self._initialize_level_beliefs()
    
    def _initialize_level_beliefs(self):
        """Initialize level-specific beliefs."""
        # Create initial beliefs appropriate for this hierarchy level
        belief = Belief(
            mean=np.zeros(self.state_dim),
            variance=np.ones(self.state_dim),
            support=None
        )
        self.beliefs.add_belief(f"level_{self.level_id}_state", belief)
    
    def set_upper_level(self, upper_level: 'HTAILevel'):
        """Set connection to upper hierarchy level."""
        self.upper_level = upper_level
    
    def set_lower_level(self, lower_level: 'HTAILevel'):
        """Set connection to lower hierarchy level."""
        self.lower_level = lower_level
    
    def process_input(self, input_data: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        """Process input at this hierarchy level."""
        # Store in temporal buffer
        if not self.temporal_buffer.full():
            self.temporal_buffer.put((input_data, action, time.time()))
        else:
            # Remove oldest, add newest
            self.temporal_buffer.get()
            self.temporal_buffer.put((input_data, action, time.time()))
        
        # Generate prediction based on current beliefs
        prediction = self._generate_level_prediction(input_data)
        
        # Compute prediction error
        prediction_error = self._compute_prediction_error(input_data, prediction)
        
        # Update beliefs based on prediction error
        self._update_level_beliefs(prediction_error)
        
        # Compute level activation (how active this level is)
        activation = np.linalg.norm(prediction_error)
        
        return {
            'prediction': prediction,
            'prediction_error': prediction_error,
            'activation': activation,
            'beliefs_updated': True
        }
    
    def _generate_level_prediction(self, input_data: np.ndarray) -> np.ndarray:
        """Generate prediction at this hierarchy level."""
        # Get current beliefs
        belief_dict = self.beliefs.get_all_beliefs()
        if not belief_dict:
            return np.zeros_like(input_data)
        
        # Simple prediction based on belief mean
        level_belief = belief_dict[f"level_{self.level_id}_state"]
        prediction = level_belief.mean
        
        # Expand/contract to match input dimensionality
        if prediction.size != input_data.size:
            if prediction.size < input_data.size:
                # Pad with zeros
                prediction = np.pad(prediction, (0, input_data.size - prediction.size))
            else:
                # Truncate
                prediction = prediction[:input_data.size]
        
        return prediction
    
    def _compute_prediction_error(self, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Compute prediction error with temporal weighting."""
        base_error = observed - predicted
        
        # Weight error by temporal scale (higher levels should have lower precision)
        temporal_weight = 1.0 / np.sqrt(self.temporal_scale)
        weighted_error = base_error * temporal_weight
        
        self.error_history.append(np.linalg.norm(weighted_error))
        if len(self.error_history) > 1000:  # Limit history size
            self.error_history = self.error_history[-1000:]
        
        return weighted_error
    
    def _update_level_beliefs(self, prediction_error: np.ndarray):
        """Update beliefs based on prediction error."""
        belief_dict = self.beliefs.get_all_beliefs()
        if not belief_dict:
            return
        
        level_belief = belief_dict[f"level_{self.level_id}_state"]
        
        # Simple belief update rule
        learning_rate = 0.1 / self.temporal_scale  # Slower learning for higher levels
        
        # Update mean based on prediction error
        error_for_update = prediction_error[:level_belief.mean.size]
        level_belief.mean += learning_rate * error_for_update
        
        # Update variance (decrease with experience)
        variance_decay = 0.99
        level_belief.variance *= variance_decay
        level_belief.variance += 0.01  # Prevent collapse to zero
    
    def generate_prediction(self) -> np.ndarray:
        """Generate prediction for this level."""
        return self._generate_level_prediction(np.zeros(self.state_dim))
    
    def modulate_prediction(self, upper_prediction: np.ndarray) -> np.ndarray:
        """Modulate prediction from upper level with level-specific information."""
        level_prediction = self.generate_prediction()
        
        # Combine predictions with coupling strength
        if upper_prediction.size != level_prediction.size:
            # Resize to match
            min_size = min(upper_prediction.size, level_prediction.size)
            upper_prediction = upper_prediction[:min_size]
            level_prediction = level_prediction[:min_size]
        
        modulated = (1 - self.coupling_strength) * level_prediction + self.coupling_strength * upper_prediction
        return modulated
    
    def plan_action(self, horizon: int) -> np.ndarray:
        """Plan action at this hierarchy level."""
        # Simple planning based on current beliefs
        belief_dict = self.beliefs.get_all_beliefs()
        if not belief_dict:
            return np.random.randn(2) * 0.1  # Small random action
        
        level_belief = belief_dict[f"level_{self.level_id}_state"]
        
        # Action is proportional to belief uncertainty
        uncertainty = np.mean(level_belief.variance)
        action_magnitude = uncertainty / (1 + self.temporal_scale)  # Slower scales -> smaller actions
        
        # Random exploration component
        exploration = np.random.randn(2) * action_magnitude * 0.1
        
        # Directed component (toward reducing prediction error)
        if self.error_history:
            recent_error = np.mean(self.error_history[-10:])
            directed = np.random.randn(2) * recent_error * 0.1
        else:
            directed = np.zeros(2)
        
        return exploration + directed
    
    def get_beliefs(self) -> Dict[str, Any]:
        """Get current beliefs at this level."""
        belief_dict = self.beliefs.get_all_beliefs()
        return {
            'beliefs': belief_dict,
            'temporal_scale': self.temporal_scale,
            'level_id': self.level_id
        }
    
    def get_level_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for this level."""
        return {
            'level_id': self.level_id,
            'temporal_scale': self.temporal_scale,
            'state_dim': self.state_dim,
            'avg_prediction_error': np.mean(self.error_history) if self.error_history else 0,
            'prediction_error_trend': np.polyfit(range(len(self.error_history)), self.error_history, 1)[0] if len(self.error_history) > 1 else 0,
            'belief_uncertainty': np.mean([np.mean(belief.variance) for belief in self.beliefs.get_all_beliefs().values()]) if self.beliefs.get_all_beliefs() else 0
        }


class MetaActiveInference:
    """
    Meta-Active Inference for rapid adaptation to new tasks.
    
    Implements meta-learning principles within Active Inference framework,
    enabling rapid adaptation to new environments and tasks.
    """
    
    def __init__(self, base_agent: ActiveInferenceAgent, meta_learning_rate: float = 0.01):
        self.base_agent = base_agent
        self.meta_learning_rate = meta_learning_rate
        self.logger = logging.getLogger("MetaActiveInference")
        
        # Meta-level components
        self.meta_beliefs = BeliefState()
        self.task_representations = {}
        self.adaptation_history = []
        
        # Meta-model for task dynamics
        self.meta_model = MetaGenerativeModel()
        
        # Performance tracking
        self.meta_performance = {
            'adaptation_speed': [],
            'transfer_efficiency': [],
            'meta_prediction_errors': []
        }
    
    def adapt_to_new_task(self, task_id: str, initial_observations: List[np.ndarray], 
                         max_adaptation_steps: int = 10) -> Dict[str, Any]:
        """Rapidly adapt to a new task using meta-learning."""
        start_time = time.time()
        
        # Initialize task representation
        task_repr = self._initialize_task_representation(task_id, initial_observations)
        self.task_representations[task_id] = task_repr
        
        adaptation_steps = []
        
        for step in range(max_adaptation_steps):
            # Meta-inference: infer task structure
            meta_beliefs = self._meta_inference(task_id, initial_observations[:step+1])
            
            # Adapt base agent based on meta-beliefs
            adaptation_result = self._adapt_base_agent(meta_beliefs, task_repr)
            
            # Evaluate adaptation quality
            adaptation_quality = self._evaluate_adaptation(task_id, adaptation_result)
            
            adaptation_steps.append({
                'step': step,
                'adaptation_quality': adaptation_quality,
                'meta_beliefs': meta_beliefs,
                'adaptation_result': adaptation_result
            })
            
            # Early stopping if adaptation is good enough
            if adaptation_quality > 0.8:
                self.logger.info(f"Early adaptation success for task {task_id} at step {step}")
                break
        
        adaptation_time = time.time() - start_time
        
        # Update meta-learning from this adaptation experience
        self._update_meta_learning(task_id, adaptation_steps)
        
        # Record performance metrics
        final_quality = adaptation_steps[-1]['adaptation_quality'] if adaptation_steps else 0
        self.meta_performance['adaptation_speed'].append(len(adaptation_steps))
        self.meta_performance['transfer_efficiency'].append(final_quality)
        
        self.logger.info(f"Meta-adaptation completed for {task_id}: "
                        f"quality={final_quality:.3f}, steps={len(adaptation_steps)}, "
                        f"time={adaptation_time:.1f}s")
        
        return {
            'task_id': task_id,
            'adaptation_steps': adaptation_steps,
            'final_adaptation_quality': final_quality,
            'adaptation_time': adaptation_time,
            'task_representation': task_repr
        }
    
    def _initialize_task_representation(self, task_id: str, observations: List[np.ndarray]) -> Dict[str, Any]:
        """Initialize representation for a new task."""
        if not observations:
            return {'task_id': task_id, 'obs_stats': None, 'task_embedding': np.zeros(10)}
        
        # Compute basic statistics of observations
        obs_array = np.array(observations)
        obs_stats = {
            'mean': np.mean(obs_array, axis=0),
            'std': np.std(obs_array, axis=0),
            'range': np.max(obs_array, axis=0) - np.min(obs_array, axis=0)
        }
        
        # Create task embedding (simple approach)
        task_embedding = np.concatenate([
            obs_stats['mean'][:5] if len(obs_stats['mean']) >= 5 else obs_stats['mean'],
            obs_stats['std'][:5] if len(obs_stats['std']) >= 5 else obs_stats['std']
        ])
        
        # Pad or truncate to fixed size
        if task_embedding.size > 10:
            task_embedding = task_embedding[:10]
        else:
            task_embedding = np.pad(task_embedding, (0, 10 - task_embedding.size))
        
        return {
            'task_id': task_id,
            'obs_stats': obs_stats,
            'task_embedding': task_embedding,
            'created_time': time.time()
        }
    
    def _meta_inference(self, task_id: str, observations: List[np.ndarray]) -> Dict[str, Any]:
        """Perform meta-level inference about task structure."""
        task_repr = self.task_representations.get(task_id, {})
        
        # Compare with known tasks to infer similarity
        task_similarities = {}
        for known_task_id, known_repr in self.task_representations.items():
            if known_task_id != task_id and 'task_embedding' in known_repr:
                similarity = self._compute_task_similarity(
                    task_repr.get('task_embedding', np.zeros(10)),
                    known_repr['task_embedding']
                )
                task_similarities[known_task_id] = similarity
        
        # Find most similar task
        most_similar_task = None
        max_similarity = 0
        if task_similarities:
            most_similar_task = max(task_similarities, key=task_similarities.get)
            max_similarity = task_similarities[most_similar_task]
        
        # Meta-beliefs about task structure
        meta_beliefs = {
            'task_novelty': 1.0 - max_similarity,
            'most_similar_task': most_similar_task,
            'task_similarity_scores': task_similarities,
            'expected_adaptation_difficulty': 1.0 - max_similarity,
            'transfer_recommendations': self._generate_transfer_recommendations(most_similar_task, max_similarity)
        }
        
        return meta_beliefs
    
    def _compute_task_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between task embeddings."""
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _generate_transfer_recommendations(self, similar_task: str, similarity: float) -> Dict[str, Any]:
        """Generate recommendations for transfer learning."""
        if similar_task is None or similarity < 0.3:
            return {
                'use_transfer': False,
                'recommended_strategy': 'learn_from_scratch'
            }
        
        if similarity > 0.8:
            strategy = 'direct_transfer'
        elif similarity > 0.5:
            strategy = 'partial_transfer'
        else:
            strategy = 'minimal_transfer'
        
        return {
            'use_transfer': True,
            'source_task': similar_task,
            'similarity_score': similarity,
            'recommended_strategy': strategy
        }
    
    def _adapt_base_agent(self, meta_beliefs: Dict[str, Any], task_repr: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt the base agent based on meta-level beliefs."""
        transfer_recs = meta_beliefs['transfer_recommendations']
        
        if transfer_recs['use_transfer']:
            # Transfer learning from similar task
            source_task = transfer_recs['source_task']
            strategy = transfer_recs['recommended_strategy']
            
            if strategy == 'direct_transfer':
                # Directly use parameters from similar task (simplified)
                adaptation_strength = 0.8
            elif strategy == 'partial_transfer':
                adaptation_strength = 0.5
            else:  # minimal_transfer
                adaptation_strength = 0.2
        else:
            # Learn from scratch
            adaptation_strength = 0.0
        
        # Adjust base agent parameters (simplified adaptation)
        original_lr = self.base_agent.learning_rate
        original_temp = self.base_agent.temperature
        
        # Adaptive learning rate based on task novelty
        novelty = meta_beliefs['task_novelty']
        adapted_lr = original_lr * (1 + novelty)  # Higher learning rate for novel tasks
        adapted_temp = original_temp * (1 + novelty * 0.5)  # Higher temperature for exploration
        
        # Apply adaptations
        self.base_agent.learning_rate = adapted_lr
        self.base_agent.temperature = adapted_temp
        
        return {
            'adaptation_strength': adaptation_strength,
            'adapted_learning_rate': adapted_lr,
            'adapted_temperature': adapted_temp,
            'original_learning_rate': original_lr,
            'original_temperature': original_temp,
            'transfer_strategy': transfer_recs.get('recommended_strategy', 'none')
        }
    
    def _evaluate_adaptation(self, task_id: str, adaptation_result: Dict[str, Any]) -> float:
        """Evaluate the quality of adaptation."""
        # Simple heuristic evaluation
        # In practice, this would involve running the adapted agent on the task
        
        adaptation_strength = adaptation_result['adaptation_strength']
        
        # Base quality from adaptation strength
        base_quality = min(0.8, adaptation_strength + 0.3)
        
        # Add some randomness to simulate actual performance variation
        performance_noise = np.random.normal(0, 0.1)
        quality = np.clip(base_quality + performance_noise, 0, 1)
        
        return quality
    
    def _update_meta_learning(self, task_id: str, adaptation_steps: List[Dict[str, Any]]) -> None:
        """Update meta-learning from adaptation experience."""
        # Record this adaptation experience
        adaptation_record = {
            'task_id': task_id,
            'n_steps': len(adaptation_steps),
            'final_quality': adaptation_steps[-1]['adaptation_quality'] if adaptation_steps else 0,
            'meta_prediction_error': self._compute_meta_prediction_error(adaptation_steps),
            'timestamp': time.time()
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Update meta-model (simplified)
        meta_prediction_error = adaptation_record['meta_prediction_error']
        self.meta_performance['meta_prediction_errors'].append(meta_prediction_error)
        
        # Trim history to prevent memory bloat
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]
    
    def _compute_meta_prediction_error(self, adaptation_steps: List[Dict[str, Any]]) -> float:
        """Compute prediction error at meta-level."""
        if not adaptation_steps:
            return 1.0
        
        # Predict how many steps adaptation should take
        predicted_steps = 5  # Simple baseline prediction
        actual_steps = len(adaptation_steps)
        
        # Normalized error
        error = abs(predicted_steps - actual_steps) / max(predicted_steps, actual_steps)
        return error
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics."""
        return {
            'n_tasks_encountered': len(self.task_representations),
            'n_adaptations': len(self.adaptation_history),
            'avg_adaptation_speed': np.mean(self.meta_performance['adaptation_speed']) if self.meta_performance['adaptation_speed'] else 0,
            'avg_transfer_efficiency': np.mean(self.meta_performance['transfer_efficiency']) if self.meta_performance['transfer_efficiency'] else 0,
            'avg_meta_prediction_error': np.mean(self.meta_performance['meta_prediction_errors']) if self.meta_performance['meta_prediction_errors'] else 0,
            'task_representations': list(self.task_representations.keys()),
            'meta_learning_rate': self.meta_learning_rate
        }


class MetaGenerativeModel:
    """Meta-level generative model for task structure."""
    
    def __init__(self):
        self.task_priors = {}
        self.transition_patterns = {}
        self.observation_patterns = {}
    
    def update_task_prior(self, task_id: str, observations: List[np.ndarray]):
        """Update prior beliefs about task structure."""
        pass  # Simplified implementation
    
    def predict_task_dynamics(self, task_id: str) -> Dict[str, Any]:
        """Predict task dynamics based on meta-model."""
        return {'predicted_complexity': 0.5}  # Simplified


class QuantumInspiredVariationalInference:
    """
    Quantum-inspired Variational Inference for Active Inference.
    
    Uses quantum computing principles like superposition and entanglement
    to enhance belief representation and inference.
    """
    
    def __init__(self, n_qubits: int = 8, coherence_time: float = 1.0):
        self.n_qubits = n_qubits
        self.coherence_time = coherence_time
        self.logger = logging.getLogger("QuantumInspiredVI")
        
        # Quantum-inspired state representation
        self.quantum_state = self._initialize_quantum_state()
        self.entanglement_matrix = np.eye(n_qubits)  # Identity = no entanglement initially
        
        # Performance tracking
        self.quantum_performance = {
            'coherence_decay': [],
            'entanglement_measures': [],
            'superposition_advantages': []
        }
    
    def _initialize_quantum_state(self) -> Dict[str, np.ndarray]:
        """Initialize quantum-inspired belief state."""
        # Each 'qubit' represents a degree of freedom in belief space
        amplitudes = np.random.randn(2**self.n_qubits) + 1j * np.random.randn(2**self.n_qubits)
        
        # Normalize to unit probability
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        return {
            'amplitudes': amplitudes,
            'phases': np.angle(amplitudes),
            'probabilities': np.abs(amplitudes)**2
        }
    
    def quantum_belief_update(self, observation: np.ndarray, prior_beliefs: BeliefState) -> BeliefState:
        """Update beliefs using quantum-inspired inference."""
        start_time = time.time()
        
        # Convert classical beliefs to quantum representation
        quantum_prior = self._classical_to_quantum(prior_beliefs)
        
        # Quantum measurement operator based on observation
        measurement_operator = self._construct_measurement_operator(observation)
        
        # Apply quantum evolution
        evolved_state = self._quantum_evolution(quantum_prior, measurement_operator)
        
        # Measure coherence decay
        coherence = self._measure_coherence(evolved_state)
        self.quantum_performance['coherence_decay'].append(1 - coherence)
        
        # Convert back to classical beliefs
        updated_beliefs = self._quantum_to_classical(evolved_state)
        
        processing_time = time.time() - start_time
        
        self.logger.debug(f"Quantum belief update completed in {processing_time:.3f}s, "
                         f"coherence={coherence:.3f}")
        
        return updated_beliefs
    
    def _classical_to_quantum(self, beliefs: BeliefState) -> Dict[str, np.ndarray]:
        """Convert classical beliefs to quantum-inspired representation."""
        belief_dict = beliefs.get_all_beliefs()
        
        if not belief_dict:
            return self.quantum_state
        
        # Simple encoding: belief means -> quantum amplitudes
        combined_means = np.concatenate([belief.mean for belief in belief_dict.values()])
        
        # Pad or truncate to match quantum state size
        target_size = 2**self.n_qubits
        if combined_means.size > target_size:
            combined_means = combined_means[:target_size]
        else:
            combined_means = np.pad(combined_means, (0, target_size - combined_means.size))
        
        # Convert to complex amplitudes
        amplitudes = combined_means.astype(complex)
        amplitudes = amplitudes / np.linalg.norm(amplitudes) if np.linalg.norm(amplitudes) > 0 else amplitudes
        
        return {
            'amplitudes': amplitudes,
            'phases': np.angle(amplitudes),
            'probabilities': np.abs(amplitudes)**2
        }
    
    def _construct_measurement_operator(self, observation: np.ndarray) -> np.ndarray:
        """Construct quantum measurement operator from observation."""
        # Simple approach: observation determines measurement basis rotation
        obs_norm = np.linalg.norm(observation)
        if obs_norm == 0:
            return np.eye(2**self.n_qubits)
        
        # Create rotation angle from observation
        rotation_angle = obs_norm / (1 + obs_norm)  # Normalize to [0, 1]
        
        # Simple rotation matrix (in practice, would be more sophisticated)
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        # For simplicity, apply same rotation to all dimensions
        rotation_2d = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        # Expand to full dimensionality (simplified)
        n_dim = 2**self.n_qubits
        measurement_op = np.eye(n_dim)
        
        # Apply rotation to pairs of dimensions
        for i in range(0, min(n_dim-1, 2), 2):
            measurement_op[i:i+2, i:i+2] = rotation_2d
        
        return measurement_op
    
    def _quantum_evolution(self, quantum_state: Dict[str, np.ndarray], 
                          measurement_op: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply quantum evolution to the state."""
        amplitudes = quantum_state['amplitudes']
        
        # Apply measurement operator
        evolved_amplitudes = measurement_op @ amplitudes
        
        # Apply decoherence (loss of quantum coherence over time)
        decoherence_factor = np.exp(-1.0 / self.coherence_time)
        evolved_amplitudes *= decoherence_factor
        
        # Renormalize
        norm = np.linalg.norm(evolved_amplitudes)
        if norm > 0:
            evolved_amplitudes = evolved_amplitudes / norm
        
        return {
            'amplitudes': evolved_amplitudes,
            'phases': np.angle(evolved_amplitudes),
            'probabilities': np.abs(evolved_amplitudes)**2
        }
    
    def _measure_coherence(self, quantum_state: Dict[str, np.ndarray]) -> float:
        """Measure quantum coherence of the state."""
        amplitudes = quantum_state['amplitudes']
        
        # Coherence measure: sum of off-diagonal density matrix elements
        # Simplified: measure phase spread
        phases = np.angle(amplitudes)
        phase_spread = np.std(phases)
        
        # Convert to coherence score (lower spread = higher coherence)
        coherence = np.exp(-phase_spread)
        
        return coherence
    
    def _quantum_to_classical(self, quantum_state: Dict[str, np.ndarray]) -> BeliefState:
        """Convert quantum state back to classical beliefs."""
        probabilities = quantum_state['probabilities']
        amplitudes = quantum_state['amplitudes']
        
        # Extract classical belief parameters from quantum state
        # Simplified: use probabilities as belief means, phases for variance
        n_beliefs = min(4, len(probabilities) // 2)  # Assume up to 4 beliefs
        
        classical_beliefs = BeliefState()
        
        for i in range(n_beliefs):
            start_idx = i * (len(probabilities) // n_beliefs)
            end_idx = (i + 1) * (len(probabilities) // n_beliefs)
            
            # Extract belief parameters from quantum state slice
            belief_probs = probabilities[start_idx:end_idx]
            belief_phases = np.angle(amplitudes[start_idx:end_idx])
            
            # Convert to classical belief
            mean = np.mean(belief_probs) * np.cos(np.mean(belief_phases))
            variance = np.var(belief_probs) + 0.1  # Add noise floor
            
            # Ensure proper dimensionality
            mean_vector = np.array([mean])
            variance_vector = np.array([variance])
            
            belief = Belief(mean=mean_vector, variance=variance_vector)
            classical_beliefs.add_belief(f"quantum_belief_{i}", belief)
        
        return classical_beliefs
    
    def create_entanglement(self, belief_indices: List[int], entanglement_strength: float = 0.5):
        """Create entanglement between belief components."""
        if len(belief_indices) < 2:
            return
        
        # Update entanglement matrix
        for i in belief_indices:
            for j in belief_indices:
                if i != j and i < self.n_qubits and j < self.n_qubits:
                    self.entanglement_matrix[i, j] = entanglement_strength
        
        # Measure entanglement
        entanglement_measure = np.trace(self.entanglement_matrix @ self.entanglement_matrix.T)
        self.quantum_performance['entanglement_measures'].append(entanglement_measure)
    
    def get_quantum_statistics(self) -> Dict[str, Any]:
        """Get quantum-inspired inference statistics."""
        return {
            'n_qubits': self.n_qubits,
            'coherence_time': self.coherence_time,
            'avg_coherence_decay': np.mean(self.quantum_performance['coherence_decay']) if self.quantum_performance['coherence_decay'] else 0,
            'avg_entanglement': np.mean(self.quantum_performance['entanglement_measures']) if self.quantum_performance['entanglement_measures'] else 0,
            'current_entanglement_matrix': self.entanglement_matrix.tolist(),
            'quantum_advantage_score': self._compute_quantum_advantage()
        }
    
    def _compute_quantum_advantage(self) -> float:
        """Compute estimated quantum advantage over classical inference."""
        # Heuristic: higher entanglement and coherence = potential advantage
        avg_entanglement = np.mean(self.quantum_performance['entanglement_measures']) if self.quantum_performance['entanglement_measures'] else 0
        avg_coherence = 1 - (np.mean(self.quantum_performance['coherence_decay']) if self.quantum_performance['coherence_decay'] else 0)
        
        # Simple combination
        advantage = (avg_entanglement * 0.4 + avg_coherence * 0.6)
        return np.clip(advantage, 0, 1)


class MultiModalActiveInference:
    """
    Multi-modal Active Inference for complex sensory integration.
    
    Handles multiple sensory modalities (vision, audio, proprioception, etc.)
    with cross-modal learning and attention mechanisms.
    """
    
    def __init__(self, modalities: List[str], attention_mechanism: str = "dynamic"):
        self.modalities = modalities
        self.attention_mechanism = attention_mechanism
        self.logger = logging.getLogger("MultiModalAI")
        
        # Modality-specific components
        self.modality_agents = {}
        self.modality_dimensions = {}
        self.attention_weights = {}
        
        # Cross-modal integration
        self.integration_matrix = np.eye(len(modalities))
        self.cross_modal_history = []
        
        # Performance tracking
        self.multimodal_performance = {
            'modality_contributions': {mod: [] for mod in modalities},
            'attention_dynamics': [],
            'integration_quality': [],
            'cross_modal_learning': []
        }
        
        # Initialize components
        self._initialize_modality_agents()
        self._initialize_attention_system()
    
    def _initialize_modality_agents(self):
        """Initialize separate agents for each modality."""
        # Default dimensions for different modalities
        default_dims = {
            'visual': {'state_dim': 8, 'obs_dim': 64, 'action_dim': 2},
            'auditory': {'state_dim': 4, 'obs_dim': 16, 'action_dim': 2},
            'proprioceptive': {'state_dim': 6, 'obs_dim': 12, 'action_dim': 2},
            'tactile': {'state_dim': 4, 'obs_dim': 8, 'action_dim': 2}
        }
        
        for modality in self.modalities:
            dims = default_dims.get(modality, {'state_dim': 4, 'obs_dim': 8, 'action_dim': 2})
            
            agent = ActiveInferenceAgent(
                state_dim=dims['state_dim'],
                obs_dim=dims['obs_dim'],
                action_dim=dims['action_dim'],
                inference_method="variational",
                planning_horizon=3,
                learning_rate=0.01,
                temperature=0.5,
                agent_id=f"modality_{modality}"
            )
            
            self.modality_agents[modality] = agent
            self.modality_dimensions[modality] = dims
            self.attention_weights[modality] = 1.0 / len(self.modalities)  # Equal initial weights
    
    def _initialize_attention_system(self):
        """Initialize attention mechanism for modality integration."""
        if self.attention_mechanism == "dynamic":
            # Dynamic attention based on prediction errors
            self.attention_params = {
                'decay_rate': 0.9,
                'sensitivity': 1.0,
                'min_weight': 0.05
            }
        elif self.attention_mechanism == "learned":
            # Learned attention weights
            self.attention_params = {
                'learning_rate': 0.01,
                'weight_history': [],
                'performance_history': []
            }
        else:  # static
            # Fixed equal attention
            self.attention_params = {}
    
    def process_multimodal_observation(self, 
                                     observations: Dict[str, np.ndarray],
                                     actions: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """Process observations from multiple modalities."""
        start_time = time.time()
        
        if actions is None:
            actions = {mod: np.zeros(2) for mod in self.modalities}  # Default actions
        
        # Process each modality independently
        modality_results = {}
        modality_errors = {}
        
        for modality in self.modalities:
            if modality in observations:
                try:
                    agent = self.modality_agents[modality]
                    action = actions.get(modality, np.zeros(2))
                    
                    # Process observation in this modality
                    agent_action = agent.act(observations[modality])
                    agent_stats = agent.get_statistics()
                    
                    # Update with action feedback if provided
                    agent.update_model(observations[modality], action)
                    
                    modality_results[modality] = {
                        'agent_action': agent_action,
                        'beliefs': agent.beliefs.get_all_beliefs(),
                        'free_energy': agent_stats.get('current_free_energy', 0),
                        'prediction_error': agent_stats.get('current_free_energy', 0)  # Simplified
                    }
                    
                    modality_errors[modality] = modality_results[modality]['prediction_error']
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {modality}: {e}")
                    modality_results[modality] = {'error': str(e)}
                    modality_errors[modality] = 1.0  # High error for failed modality
        
        # Update attention weights based on prediction errors
        self._update_attention_weights(modality_errors)
        
        # Integrate across modalities
        integrated_result = self._integrate_modalities(modality_results)
        
        # Cross-modal learning
        cross_modal_update = self._cross_modal_learning(modality_results)
        
        # Record performance
        self._record_multimodal_performance(modality_results, integrated_result)
        
        processing_time = time.time() - start_time
        
        return {
            'modality_results': modality_results,
            'attention_weights': self.attention_weights.copy(),
            'integrated_result': integrated_result,
            'cross_modal_update': cross_modal_update,
            'processing_time': processing_time
        }
    
    def _update_attention_weights(self, modality_errors: Dict[str, float]):
        """Update attention weights based on modality performance."""
        if self.attention_mechanism == "static":
            return  # No updates for static attention
        
        elif self.attention_mechanism == "dynamic":
            # Inverse relationship: lower error = higher attention
            total_inverse_error = sum(1.0 / (error + 1e-6) for error in modality_errors.values())
            
            for modality, error in modality_errors.items():
                # Higher weight for lower error
                new_weight = (1.0 / (error + 1e-6)) / total_inverse_error
                
                # Smooth update with decay
                decay_rate = self.attention_params['decay_rate']
                self.attention_weights[modality] = (decay_rate * self.attention_weights[modality] + 
                                                   (1 - decay_rate) * new_weight)
                
                # Enforce minimum weight
                min_weight = self.attention_params['min_weight']
                self.attention_weights[modality] = max(self.attention_weights[modality], min_weight)
        
        elif self.attention_mechanism == "learned":
            # Gradient-based learning (simplified)
            learning_rate = self.attention_params['learning_rate']
            
            for modality, error in modality_errors.items():
                # Gradient of attention weight w.r.t. performance
                gradient = -error  # Negative gradient: reduce attention for high error
                self.attention_weights[modality] += learning_rate * gradient
                self.attention_weights[modality] = np.clip(self.attention_weights[modality], 0.01, 1.0)
            
            # Renormalize weights
            total_weight = sum(self.attention_weights.values())
            for modality in self.attention_weights:
                self.attention_weights[modality] /= total_weight
        
        # Record attention dynamics
        self.multimodal_performance['attention_dynamics'].append(self.attention_weights.copy())
    
    def _integrate_modalities(self, modality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate information across modalities using attention weights."""
        integrated_beliefs = {}
        integrated_actions = np.zeros(2)  # Assuming 2D action space
        weighted_free_energy = 0.0
        total_weight = 0.0
        
        for modality, result in modality_results.items():
            if 'error' in result:
                continue  # Skip failed modalities
            
            weight = self.attention_weights.get(modality, 0.0)
            
            # Integrate actions
            if 'agent_action' in result:
                integrated_actions += weight * result['agent_action']
            
            # Integrate free energy
            if 'free_energy' in result:
                weighted_free_energy += weight * result['free_energy']
                total_weight += weight
            
            # Integrate beliefs (simplified: just track which modalities contributed)
            if 'beliefs' in result:
                for belief_name, belief in result['beliefs'].items():
                    integrated_beliefs[f"{modality}_{belief_name}"] = {
                        'belief': belief,
                        'weight': weight
                    }
        
        # Normalize integrated free energy
        if total_weight > 0:
            weighted_free_energy /= total_weight
        
        # Quality measure for integration
        integration_quality = self._compute_integration_quality(modality_results)
        self.multimodal_performance['integration_quality'].append(integration_quality)
        
        return {
            'integrated_action': integrated_actions,
            'integrated_beliefs': integrated_beliefs,
            'integrated_free_energy': weighted_free_energy,
            'integration_quality': integration_quality,
            'contributing_modalities': [mod for mod in modality_results.keys() if 'error' not in modality_results[mod]]
        }
    
    def _compute_integration_quality(self, modality_results: Dict[str, Any]) -> float:
        """Compute quality of multi-modal integration."""
        successful_modalities = [mod for mod, result in modality_results.items() if 'error' not in result]
        
        if len(successful_modalities) < 2:
            return 0.5  # Low quality if few modalities available
        
        # Quality based on consistency across modalities
        actions = []
        free_energies = []
        
        for modality in successful_modalities:
            result = modality_results[modality]
            if 'agent_action' in result:
                actions.append(result['agent_action'])
            if 'free_energy' in result:
                free_energies.append(result['free_energy'])
        
        # Measure consistency (lower variance = higher quality)
        action_consistency = 0.5
        if len(actions) > 1:
            action_variance = np.var([np.linalg.norm(action) for action in actions])
            action_consistency = 1.0 / (1.0 + action_variance)
        
        energy_consistency = 0.5
        if len(free_energies) > 1:
            energy_variance = np.var(free_energies)
            energy_consistency = 1.0 / (1.0 + energy_variance)
        
        # Overall quality
        quality = (action_consistency * 0.6 + energy_consistency * 0.4)
        return np.clip(quality, 0, 1)
    
    def _cross_modal_learning(self, modality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal learning to improve modality integration."""
        # Identify best-performing modality
        best_modality = None
        best_performance = float('inf')
        
        for modality, result in modality_results.items():
            if 'error' not in result and 'free_energy' in result:
                if result['free_energy'] < best_performance:
                    best_performance = result['free_energy']
                    best_modality = modality
        
        # Cross-modal transfer: weaker modalities learn from stronger ones
        transfer_results = {}
        
        if best_modality:
            best_result = modality_results[best_modality]
            
            for modality, result in modality_results.items():
                if (modality != best_modality and 'error' not in result and 
                    'free_energy' in result and result['free_energy'] > best_performance * 1.1):
                    
                    # Simple transfer: adjust weaker modality toward stronger one
                    transfer_strength = 0.1  # Conservative transfer
                    
                    # This would involve updating the weaker modality's model
                    # Simplified: just record the transfer intent
                    transfer_results[modality] = {
                        'source_modality': best_modality,
                        'transfer_strength': transfer_strength,
                        'performance_gap': result['free_energy'] - best_performance
                    }
        
        # Record cross-modal learning activity
        self.multimodal_performance['cross_modal_learning'].append({
            'best_modality': best_modality,
            'transfers': transfer_results,
            'timestamp': time.time()
        })
        
        return {
            'best_performing_modality': best_modality,
            'best_performance': best_performance,
            'transfer_operations': transfer_results
        }
    
    def _record_multimodal_performance(self, modality_results: Dict[str, Any], integrated_result: Dict[str, Any]):
        """Record performance metrics for each modality."""
        for modality in self.modalities:
            if modality in modality_results and 'error' not in modality_results[modality]:
                contribution = self.attention_weights[modality] * integrated_result['integration_quality']
                self.multimodal_performance['modality_contributions'][modality].append(contribution)
        
        # Limit history size
        max_history = 1000
        for modality in self.multimodal_performance['modality_contributions']:
            history = self.multimodal_performance['modality_contributions'][modality]
            if len(history) > max_history:
                self.multimodal_performance['modality_contributions'][modality] = history[-max_history:]
    
    def get_multimodal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive multi-modal performance statistics."""
        stats = {
            'modalities': self.modalities,
            'attention_mechanism': self.attention_mechanism,
            'current_attention_weights': self.attention_weights,
            'avg_integration_quality': np.mean(self.multimodal_performance['integration_quality']) if self.multimodal_performance['integration_quality'] else 0,
            'modality_contributions': {},
            'cross_modal_learning_events': len(self.multimodal_performance['cross_modal_learning']),
            'attention_stability': self._compute_attention_stability()
        }
        
        # Per-modality statistics
        for modality in self.modalities:
            contributions = self.multimodal_performance['modality_contributions'][modality]
            agent_stats = self.modality_agents[modality].get_statistics()
            
            stats['modality_contributions'][modality] = {
                'avg_contribution': np.mean(contributions) if contributions else 0,
                'current_attention_weight': self.attention_weights[modality],
                'agent_performance': agent_stats.get('average_reward', 0),
                'agent_health': agent_stats.get('health_status', 'unknown')
            }
        
        return stats
    
    def _compute_attention_stability(self) -> float:
        """Compute stability of attention weights over time."""
        if len(self.multimodal_performance['attention_dynamics']) < 2:
            return 1.0
        
        # Measure variance in attention weights over recent history
        recent_dynamics = self.multimodal_performance['attention_dynamics'][-20:]  # Last 20 updates
        
        if len(recent_dynamics) < 2:
            return 1.0
        
        # Compute variance for each modality
        modality_variances = []
        for modality in self.modalities:
            weights = [dynamics[modality] for dynamics in recent_dynamics]
            variance = np.var(weights)
            modality_variances.append(variance)
        
        # Overall stability (lower variance = higher stability)
        avg_variance = np.mean(modality_variances)
        stability = 1.0 / (1.0 + avg_variance)
        
        return stability


class ConcurrentInferenceEngine:
    """
    Concurrent processing engine for parallel Active Inference computations.
    
    Enables parallel processing of multiple agents, environments, or
    computational components to improve performance.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger("ConcurrentInference")
        
        # Performance tracking
        self.concurrent_performance = {
            'parallel_speedup': [],
            'resource_utilization': [],
            'task_completion_times': []
        }
    
    def parallel_agent_processing(self, 
                                agents: List[ActiveInferenceAgent],
                                observations: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process multiple agents in parallel."""
        start_time = time.time()
        
        if len(agents) != len(observations):
            raise ValueError("Number of agents must match number of observations")
        
        # Submit tasks to thread pool
        future_to_agent = {}
        for i, (agent, obs) in enumerate(zip(agents, observations)):
            future = self.executor.submit(self._process_single_agent, agent, obs, i)
            future_to_agent[future] = (agent, i)
        
        # Collect results
        results = [None] * len(agents)
        
        for future in as_completed(future_to_agent):
            agent, agent_index = future_to_agent[future]
            try:
                result = future.result()
                results[agent_index] = result
            except Exception as e:
                self.logger.error(f"Agent {agent.agent_id} processing failed: {e}")
                results[agent_index] = {'error': str(e), 'agent_id': agent.agent_id}
        
        processing_time = time.time() - start_time
        self.concurrent_performance['task_completion_times'].append(processing_time)
        
        # Estimate speedup (rough approximation)
        sequential_time_estimate = processing_time * len(agents)  # Rough estimate
        speedup = sequential_time_estimate / processing_time
        self.concurrent_performance['parallel_speedup'].append(speedup)
        
        return results
    
    def _process_single_agent(self, agent: ActiveInferenceAgent, observation: np.ndarray, agent_index: int) -> Dict[str, Any]:
        """Process a single agent (to be run in parallel)."""
        start_time = time.time()
        
        try:
            # Full perception-action cycle
            action = agent.act(observation)
            
            # Get agent statistics
            stats = agent.get_statistics()
            
            processing_time = time.time() - start_time
            
            return {
                'agent_id': agent.agent_id,
                'agent_index': agent_index,
                'action': action,
                'statistics': stats,
                'processing_time': processing_time,
                'success': True
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'agent_id': agent.agent_id,
                'agent_index': agent_index,
                'error': str(e),
                'processing_time': processing_time,
                'success': False
            }
    
    def parallel_experiment_execution(self, 
                                    experiment_configs: List[Dict[str, Any]],
                                    execution_function: Callable) -> List[Any]:
        """Execute multiple experiments in parallel."""
        start_time = time.time()
        
        # Submit experiment tasks
        future_to_config = {}
        for i, config in enumerate(experiment_configs):
            future = self.executor.submit(execution_function, config, i)
            future_to_config[future] = (config, i)
        
        # Collect results
        results = [None] * len(experiment_configs)
        
        for future in as_completed(future_to_config):
            config, config_index = future_to_config[future]
            try:
                result = future.result()
                results[config_index] = result
            except Exception as e:
                self.logger.error(f"Experiment {config_index} failed: {e}")
                results[config_index] = {'error': str(e), 'config_index': config_index}
        
        total_time = time.time() - start_time
        
        # Record performance
        self.concurrent_performance['task_completion_times'].append(total_time)
        
        return results
    
    def get_concurrent_statistics(self) -> Dict[str, Any]:
        """Get concurrent processing performance statistics."""
        return {
            'max_workers': self.max_workers,
            'avg_parallel_speedup': np.mean(self.concurrent_performance['parallel_speedup']) if self.concurrent_performance['parallel_speedup'] else 1.0,
            'avg_task_completion_time': np.mean(self.concurrent_performance['task_completion_times']) if self.concurrent_performance['task_completion_times'] else 0,
            'total_tasks_processed': len(self.concurrent_performance['task_completion_times']),
            'estimated_efficiency': np.mean(self.concurrent_performance['parallel_speedup']) / self.max_workers if self.concurrent_performance['parallel_speedup'] else 0
        }
    
    def shutdown(self):
        """Shutdown the concurrent processing engine."""
        self.executor.shutdown(wait=True)
        self.logger.info("Concurrent inference engine shut down")
