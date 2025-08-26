"""Continual Active Inference for Lifelong Learning.

This module implements advanced continual learning mechanisms for Active Inference
agents that can learn continuously without catastrophic forgetting, including:
- Elastic Weight Consolidation for Active Inference (EWC-AI)
- Progressive Neural Networks for hierarchical skill acquisition
- Memory-Augmented Active Inference with episodic replay
- Adaptive capacity expansion for growing knowledge
- Interference-resistant belief updating
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import json
from collections import deque, defaultdict
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.agent import ActiveInferenceAgent
from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel


@dataclass
class EpisodicMemory:
    """Represents an episodic memory for replay-based learning."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    beliefs: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: float
    importance_score: float = 1.0
    replay_count: int = 0


@dataclass
class TaskMemory:
    """Memory structure for a specific task."""
    task_id: str
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    importance_weights: Dict[str, float] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)


class ElasticWeightConsolidationAI:
    """
    Elastic Weight Consolidation for Active Inference (EWC-AI).
    
    Prevents catastrophic forgetting by constraining important parameters
    to stay close to their values on previous tasks.
    """
    
    def __init__(self, 
                 base_agent: ActiveInferenceAgent,
                 ewc_lambda: float = 1000.0,
                 fisher_estimation_samples: int = 100):
        """
        Initialize EWC-AI.
        
        Args:
            base_agent: Base Active Inference agent
            ewc_lambda: EWC regularization strength
            fisher_estimation_samples: Samples for Fisher information estimation
        """
        self.base_agent = base_agent
        self.ewc_lambda = ewc_lambda
        self.fisher_estimation_samples = fisher_estimation_samples
        self.logger = logging.getLogger("EWC-AI")
        
        # EWC components
        self.task_memories: Dict[str, TaskMemory] = {}
        self.fisher_information: Dict[str, Dict[str, float]] = {}
        self.optimal_parameters: Dict[str, Dict[str, Any]] = {}
        
        # Current task tracking
        self.current_task: Optional[str] = None
        self.task_switch_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.ewc_performance = {
            'catastrophic_forgetting_scores': [],
            'backward_transfer': [],
            'forward_transfer': [],
            'task_performance_retention': [],
            'fisher_information_quality': []
        }
    
    def begin_new_task(self, task_id: str, task_description: str = "") -> Dict[str, Any]:
        """
        Begin learning a new task with EWC protection.
        
        Args:
            task_id: Unique identifier for the new task
            task_description: Optional description of the task
            
        Returns:
            Task initialization result
        """
        start_time = time.time()
        
        # Consolidate previous task if exists
        consolidation_result = None
        if self.current_task is not None:
            consolidation_result = self._consolidate_previous_task()
        
        # Initialize new task memory
        new_task_memory = TaskMemory(
            task_id=task_id,
            model_parameters=self._extract_model_parameters(),
            importance_weights={}
        )
        self.task_memories[task_id] = new_task_memory
        
        # Update current task
        previous_task = self.current_task
        self.current_task = task_id
        
        # Record task switch
        task_switch_record = {
            'previous_task': previous_task,
            'new_task': task_id,
            'task_description': task_description,
            'switch_time': start_time,
            'consolidation_result': consolidation_result
        }
        self.task_switch_history.append(task_switch_record)
        
        initialization_time = time.time() - start_time
        
        self.logger.info(f"Started new task: {task_id} (previous: {previous_task})")
        
        return {
            'task_id': task_id,
            'previous_task': previous_task,
            'consolidation_result': consolidation_result,
            'initialization_time': initialization_time,
            'n_previous_tasks': len(self.task_memories) - 1
        }
    
    def _consolidate_previous_task(self) -> Dict[str, Any]:
        """Consolidate learning from the previous task."""
        if self.current_task is None:
            return {'status': 'no_previous_task'}
        
        start_time = time.time()
        task_memory = self.task_memories[self.current_task]
        
        # Estimate Fisher Information Matrix for important parameters
        fisher_info = self._estimate_fisher_information(task_memory)
        self.fisher_information[self.current_task] = fisher_info
        
        # Store optimal parameters for this task
        optimal_params = self._extract_model_parameters()
        self.optimal_parameters[self.current_task] = optimal_params
        
        # Compute importance weights
        importance_weights = self._compute_importance_weights(task_memory, fisher_info)
        task_memory.importance_weights = importance_weights
        
        consolidation_time = time.time() - start_time
        
        return {
            'task_id': self.current_task,
            'fisher_information_computed': len(fisher_info) > 0,
            'n_important_parameters': len(importance_weights),
            'consolidation_time': consolidation_time,
            'avg_importance_weight': np.mean(list(importance_weights.values())) if importance_weights else 0
        }
    
    def _extract_model_parameters(self) -> Dict[str, Any]:
        """Extract model parameters that need EWC protection."""
        # Extract key parameters from the agent's generative model
        params = {}
        
        try:
            # Get model parameters (simplified - would be more comprehensive)
            model_params = self.base_agent.generative_model.get_model_parameters()
            
            # Store learning-related parameters
            params['learning_rate'] = self.base_agent.learning_rate
            params['temperature'] = self.base_agent.temperature
            
            # Store model-specific parameters
            if model_params:
                params.update(model_params)
            
            # Store belief parameters
            beliefs = self.base_agent.beliefs.get_all_beliefs()
            belief_params = {}
            for belief_name, belief in beliefs.items():
                if hasattr(belief, 'mean') and hasattr(belief, 'variance'):
                    belief_params[f"{belief_name}_mean"] = belief.mean.copy() if isinstance(belief.mean, np.ndarray) else belief.mean
                    belief_params[f"{belief_name}_variance"] = belief.variance.copy() if isinstance(belief.variance, np.ndarray) else belief.variance
            
            params['beliefs'] = belief_params
            
        except Exception as e:
            self.logger.warning(f"Failed to extract some model parameters: {e}")
        
        return params
    
    def _estimate_fisher_information(self, task_memory: TaskMemory) -> Dict[str, float]:
        """Estimate Fisher Information Matrix for task parameters."""
        if not task_memory.observations:
            return {}
        
        fisher_info = {}
        
        try:
            # Use a subset of task data for Fisher information estimation
            n_samples = min(self.fisher_estimation_samples, len(task_memory.observations))
            sample_indices = np.random.choice(len(task_memory.observations), n_samples, replace=False)
            
            # Simplified Fisher information estimation
            # In practice, this would involve computing second derivatives of the log-likelihood
            parameter_variations = defaultdict(list)
            
            for idx in sample_indices:
                obs = task_memory.observations[idx]
                action = task_memory.actions[idx]
                
                # Perturb parameters and measure sensitivity
                current_params = self._extract_model_parameters()
                
                for param_name, param_value in current_params.items():
                    if param_name == 'beliefs':
                        continue  # Skip belief parameters for now
                    
                    if isinstance(param_value, (int, float)):
                        # Small perturbation
                        perturbation = 0.01 * abs(param_value) if param_value != 0 else 0.01
                        
                        # Measure sensitivity (simplified)
                        sensitivity = self._compute_parameter_sensitivity(param_name, param_value, perturbation, obs)
                        parameter_variations[param_name].append(sensitivity)
            
            # Estimate Fisher information as variance of sensitivity
            for param_name, variations in parameter_variations.items():
                if variations:
                    fisher_info[param_name] = np.var(variations)
            
        except Exception as e:
            self.logger.warning(f"Fisher information estimation failed: {e}")
        
        return fisher_info
    
    def _compute_parameter_sensitivity(self, param_name: str, param_value: float, 
                                     perturbation: float, observation: np.ndarray) -> float:
        """Compute sensitivity of model output to parameter changes."""
        try:
            # Store original value
            original_value = getattr(self.base_agent, param_name, param_value)
            
            # Compute baseline output
            baseline_action = self.base_agent.act(observation)
            baseline_output = np.linalg.norm(baseline_action)
            
            # Perturb parameter
            if hasattr(self.base_agent, param_name):
                setattr(self.base_agent, param_name, param_value + perturbation)
            
            # Compute perturbed output
            perturbed_action = self.base_agent.act(observation)
            perturbed_output = np.linalg.norm(perturbed_action)
            
            # Restore original value
            if hasattr(self.base_agent, param_name):
                setattr(self.base_agent, param_name, original_value)
            
            # Compute sensitivity
            sensitivity = abs(perturbed_output - baseline_output) / perturbation
            return sensitivity
            
        except Exception as e:
            self.logger.debug(f"Sensitivity computation failed for {param_name}: {e}")
            return 0.0
    
    def _compute_importance_weights(self, task_memory: TaskMemory, 
                                  fisher_info: Dict[str, float]) -> Dict[str, float]:
        """Compute importance weights for parameters."""
        importance_weights = {}
        
        if not fisher_info:
            return importance_weights
        
        # Normalize Fisher information to get importance weights
        max_fisher = max(fisher_info.values()) if fisher_info else 1.0
        
        for param_name, fisher_value in fisher_info.items():
            # Importance is proportional to Fisher information
            importance = fisher_value / max_fisher if max_fisher > 0 else 0.0
            importance_weights[param_name] = importance
        
        return importance_weights
    
    def ewc_regularized_update(self, observation: np.ndarray, 
                             action: np.ndarray,
                             reward: float) -> Dict[str, Any]:
        """
        Perform EWC-regularized model update.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            
        Returns:
            Update result with EWC information
        """
        start_time = time.time()
        
        # Record experience in current task memory
        if self.current_task:
            task_memory = self.task_memories[self.current_task]
            task_memory.observations.append(observation.copy())
            task_memory.actions.append(action.copy())
            task_memory.rewards.append(reward)
            task_memory.performance_history.append(reward)
        
        # Compute EWC penalty
        ewc_penalty = self._compute_ewc_penalty()
        
        # Standard model update
        try:
            self.base_agent.update_model(observation, action, reward)
            update_success = True
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            update_success = False
        
        # Apply EWC constraint (simplified)
        if update_success and ewc_penalty > 0:
            self._apply_ewc_constraint(ewc_penalty)
        
        update_time = time.time() - start_time
        
        # Evaluate forgetting
        forgetting_score = self._evaluate_catastrophic_forgetting()
        self.ewc_performance['catastrophic_forgetting_scores'].append(forgetting_score)
        
        return {
            'current_task': self.current_task,
            'ewc_penalty': ewc_penalty,
            'update_success': update_success,
            'forgetting_score': forgetting_score,
            'update_time': update_time,
            'n_protected_parameters': len(self.fisher_information.get(self.current_task or '', {}))
        }
    
    def _compute_ewc_penalty(self) -> float:
        """Compute EWC regularization penalty."""
        total_penalty = 0.0
        
        if not self.optimal_parameters or not self.fisher_information:
            return total_penalty
        
        current_params = self._extract_model_parameters()
        
        for task_id, optimal_params in self.optimal_parameters.items():
            if task_id == self.current_task:
                continue  # Don't penalize current task
            
            task_fisher = self.fisher_information.get(task_id, {})
            
            for param_name, optimal_value in optimal_params.items():
                if (param_name in current_params and 
                    param_name in task_fisher and 
                    isinstance(optimal_value, (int, float)) and
                    isinstance(current_params[param_name], (int, float))):
                    
                    # EWC penalty: F * (theta - theta*)^2
                    param_diff = current_params[param_name] - optimal_value
                    fisher_weight = task_fisher[param_name]
                    penalty = fisher_weight * (param_diff ** 2)
                    total_penalty += penalty
        
        return total_penalty * self.ewc_lambda
    
    def _apply_ewc_constraint(self, ewc_penalty: float):
        """Apply EWC constraint to limit parameter changes."""
        if ewc_penalty <= 0:
            return
        
        # Simplified EWC constraint application
        # In practice, this would involve gradient-based constraint satisfaction
        
        constraint_strength = min(1.0, ewc_penalty / 1000.0)  # Normalize penalty
        
        # Adjust learning rate based on constraint strength
        original_lr = self.base_agent.learning_rate
        constrained_lr = original_lr * (1.0 - 0.5 * constraint_strength)
        
        # Temporarily reduce learning rate
        self.base_agent.learning_rate = constrained_lr
        
        # Log constraint application
        self.logger.debug(f"Applied EWC constraint: penalty={ewc_penalty:.3f}, "
                         f"lr_reduction={original_lr - constrained_lr:.3f}")
    
    def _evaluate_catastrophic_forgetting(self) -> float:
        """Evaluate catastrophic forgetting across tasks."""
        if len(self.task_memories) <= 1:
            return 0.0  # No previous tasks to forget
        
        forgetting_scores = []
        
        for task_id, task_memory in self.task_memories.items():
            if task_id == self.current_task or not task_memory.performance_history:
                continue
            
            # Evaluate current performance on previous task data
            if task_memory.observations:
                try:
                    # Test on a sample of previous task observations
                    n_test = min(10, len(task_memory.observations))
                    test_indices = np.random.choice(len(task_memory.observations), n_test, replace=False)
                    
                    current_performance = []
                    for idx in test_indices:
                        obs = task_memory.observations[idx]
                        # Simplified performance measure
                        action = self.base_agent.act(obs)
                        performance = -np.linalg.norm(action)  # Placeholder metric
                        current_performance.append(performance)
                    
                    avg_current_performance = np.mean(current_performance)
                    original_performance = np.mean(task_memory.performance_history[-n_test:])
                    
                    # Forgetting score: how much performance degraded
                    forgetting = max(0, original_performance - avg_current_performance)
                    forgetting_scores.append(forgetting)
                    
                except Exception as e:
                    self.logger.debug(f"Forgetting evaluation failed for task {task_id}: {e}")
        
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def evaluate_transfer_learning(self) -> Dict[str, Any]:
        """Evaluate forward and backward transfer learning."""
        if len(self.task_memories) < 2:
            return {'forward_transfer': 0.0, 'backward_transfer': 0.0}
        
        # Simplified transfer evaluation
        task_performances = []
        for task_memory in self.task_memories.values():
            if task_memory.performance_history:
                avg_performance = np.mean(task_memory.performance_history)
                task_performances.append(avg_performance)
        
        if len(task_performances) < 2:
            return {'forward_transfer': 0.0, 'backward_transfer': 0.0}
        
        # Forward transfer: improvement in learning speed for later tasks
        forward_transfer = task_performances[-1] - task_performances[0] if len(task_performances) > 1 else 0.0
        
        # Backward transfer: maintained performance on earlier tasks
        backward_transfer = -self._evaluate_catastrophic_forgetting()
        
        self.ewc_performance['forward_transfer'].append(forward_transfer)
        self.ewc_performance['backward_transfer'].append(backward_transfer)
        
        return {
            'forward_transfer': forward_transfer,
            'backward_transfer': backward_transfer,
            'n_tasks_evaluated': len(task_performances)
        }
    
    def get_ewc_statistics(self) -> Dict[str, Any]:
        """Get comprehensive EWC statistics."""
        return {
            'current_task': self.current_task,
            'n_tasks_learned': len(self.task_memories),
            'n_task_switches': len(self.task_switch_history),
            'avg_catastrophic_forgetting': np.mean(self.ewc_performance['catastrophic_forgetting_scores']) if self.ewc_performance['catastrophic_forgetting_scores'] else 0,
            'avg_forward_transfer': np.mean(self.ewc_performance['forward_transfer']) if self.ewc_performance['forward_transfer'] else 0,
            'avg_backward_transfer': np.mean(self.ewc_performance['backward_transfer']) if self.ewc_performance['backward_transfer'] else 0,
            'total_protected_parameters': sum(len(fisher) for fisher in self.fisher_information.values()),
            'ewc_lambda': self.ewc_lambda,
            'task_memory_sizes': {task_id: len(memory.observations) for task_id, memory in self.task_memories.items()}
        }


class MemoryAugmentedActiveInference:
    """
    Memory-Augmented Active Inference with episodic replay.
    
    Maintains episodic memory for experience replay and uses
    memory-based learning to improve sample efficiency and
    reduce catastrophic forgetting.
    """
    
    def __init__(self, 
                 base_agent: ActiveInferenceAgent,
                 memory_size: int = 10000,
                 replay_batch_size: int = 32,
                 replay_frequency: int = 10):
        """
        Initialize Memory-Augmented Active Inference.
        
        Args:
            base_agent: Base Active Inference agent
            memory_size: Maximum size of episodic memory
            replay_batch_size: Batch size for experience replay
            replay_frequency: Frequency of replay updates
        """
        self.base_agent = base_agent
        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.replay_frequency = replay_frequency
        self.logger = logging.getLogger("MemoryAugmentedAI")
        
        # Episodic memory
        self.episodic_memory: deque = deque(maxlen=memory_size)
        self.memory_index: Dict[str, List[int]] = defaultdict(list)  # Context-based indexing
        
        # Replay mechanisms
        self.replay_count = 0
        self.last_replay_time = time.time()
        
        # Memory management
        self.importance_sampling = True
        self.memory_consolidation_threshold = 0.1
        
        # Performance tracking
        self.memory_performance = {
            'replay_effectiveness': [],
            'memory_utilization': [],
            'forgetting_prevention': [],
            'sample_efficiency_gains': []
        }
    
    def store_experience(self, 
                        observation: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Store experience in episodic memory.
        
        Args:
            observation: Observed state
            action: Action taken
            reward: Reward received
            context: Additional context information
            
        Returns:
            Storage result information
        """
        start_time = time.time()
        context = context or {}
        
        # Get current beliefs
        beliefs = self.base_agent.beliefs.get_all_beliefs()
        
        # Compute importance score
        importance_score = self._compute_importance_score(observation, action, reward, beliefs)
        
        # Create episodic memory entry
        memory_entry = EpisodicMemory(
            observation=observation.copy(),
            action=action.copy(),
            reward=reward,
            beliefs=beliefs,
            context=context.copy(),
            timestamp=time.time(),
            importance_score=importance_score
        )
        
        # Store in memory
        memory_index = len(self.episodic_memory)
        self.episodic_memory.append(memory_entry)
        
        # Update context-based index
        context_key = self._generate_context_key(context)
        self.memory_index[context_key].append(memory_index % self.memory_size)
        
        # Memory consolidation
        consolidation_result = None
        if len(self.episodic_memory) > self.memory_size * 0.9:
            consolidation_result = self._consolidate_memory()
        
        storage_time = time.time() - start_time
        
        return {
            'memory_size': len(self.episodic_memory),
            'importance_score': importance_score,
            'context_key': context_key,
            'consolidation_result': consolidation_result,
            'storage_time': storage_time
        }
    
    def _compute_importance_score(self, 
                                observation: np.ndarray,
                                action: np.ndarray,
                                reward: float,
                                beliefs: Dict[str, Any]) -> float:
        """Compute importance score for memory entry."""
        # Multiple factors contribute to importance
        
        # 1. Reward magnitude (high rewards are important)
        reward_importance = abs(reward)
        
        # 2. Surprise/novelty (high prediction error is important)
        prediction_error = self.base_agent.get_statistics().get('current_free_energy', 0.0)
        surprise_importance = min(1.0, prediction_error)
        
        # 3. Action magnitude (significant actions are important)
        action_importance = np.linalg.norm(action) / 2.0  # Normalize
        
        # 4. Belief uncertainty (uncertain situations are important)
        uncertainty_importance = 0.5  # Placeholder - would compute from beliefs
        if beliefs:
            uncertainties = []
            for belief in beliefs.values():
                if hasattr(belief, 'variance'):
                    variance = belief.variance if isinstance(belief.variance, float) else np.mean(belief.variance)
                    uncertainties.append(variance)
            if uncertainties:
                uncertainty_importance = min(1.0, np.mean(uncertainties))
        
        # 5. Temporal importance (recent experiences are more important)
        temporal_importance = 1.0  # Most recent, so highest temporal importance
        
        # Weighted combination
        importance = (
            0.3 * reward_importance +
            0.25 * surprise_importance +
            0.2 * action_importance +
            0.15 * uncertainty_importance +
            0.1 * temporal_importance
        )
        
        return min(1.0, importance)
    
    def _generate_context_key(self, context: Dict[str, Any]) -> str:
        """Generate key for context-based memory indexing."""
        if not context:
            return 'default'
        
        # Create hash-like key from context
        key_parts = []
        for k, v in sorted(context.items()):
            if isinstance(v, (int, float, str, bool)):
                key_parts.append(f"{k}:{v}")
            elif isinstance(v, (list, tuple)) and len(v) <= 3:
                key_parts.append(f"{k}:{'-'.join(map(str, v))}")
        
        context_key = '_'.join(key_parts[:5])  # Limit key size
        return context_key if context_key else 'default'
    
    def _consolidate_memory(self) -> Dict[str, Any]:
        """Consolidate memory by removing less important entries."""
        if len(self.episodic_memory) < self.memory_size * 0.5:
            return {'status': 'no_consolidation_needed'}
        
        start_time = time.time()
        
        # Sort memories by importance score
        memory_list = list(self.episodic_memory)
        memory_list.sort(key=lambda m: m.importance_score, reverse=True)
        
        # Keep top memories and recent memories
        n_keep_important = int(self.memory_size * 0.7)
        n_keep_recent = int(self.memory_size * 0.3)
        
        # Get most important memories
        important_memories = memory_list[:n_keep_important]
        
        # Get most recent memories
        recent_memories = sorted(memory_list, key=lambda m: m.timestamp, reverse=True)[:n_keep_recent]
        
        # Combine and deduplicate
        consolidated_memories = list({id(m): m for m in important_memories + recent_memories}.values())
        
        # Update memory
        old_size = len(self.episodic_memory)
        self.episodic_memory.clear()
        self.episodic_memory.extend(consolidated_memories)
        
        # Rebuild index
        self.memory_index.clear()
        for i, memory in enumerate(self.episodic_memory):
            context_key = self._generate_context_key(memory.context)
            self.memory_index[context_key].append(i)
        
        consolidation_time = time.time() - start_time
        
        return {
            'status': 'consolidation_completed',
            'old_size': old_size,
            'new_size': len(self.episodic_memory),
            'memories_removed': old_size - len(self.episodic_memory),
            'consolidation_time': consolidation_time
        }
    
    def experience_replay(self, current_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform experience replay for continual learning.
        
        Args:
            current_context: Current context for contextual replay
            
        Returns:
            Replay result information
        """
        if len(self.episodic_memory) < self.replay_batch_size:
            return {'status': 'insufficient_memory', 'memory_size': len(self.episodic_memory)}
        
        start_time = time.time()
        current_context = current_context or {}
        
        # Select memories for replay
        replay_memories = self._select_replay_memories(current_context)
        
        # Perform replay updates
        replay_results = []
        original_stats = self.base_agent.get_statistics()
        
        for memory in replay_memories:
            try:
                # Replay the experience
                memory.replay_count += 1
                
                # Update model with replayed experience
                self.base_agent.update_model(memory.observation, memory.action, memory.reward)
                
                replay_results.append({
                    'memory_timestamp': memory.timestamp,
                    'importance_score': memory.importance_score,
                    'replay_count': memory.replay_count,
                    'success': True
                })
                
            except Exception as e:
                replay_results.append({
                    'error': str(e),
                    'success': False
                })
        
        # Measure replay effectiveness
        post_replay_stats = self.base_agent.get_statistics()
        effectiveness = self._measure_replay_effectiveness(original_stats, post_replay_stats)
        self.memory_performance['replay_effectiveness'].append(effectiveness)
        
        self.replay_count += 1
        self.last_replay_time = time.time()
        replay_time = time.time() - start_time
        
        return {
            'n_replayed_memories': len(replay_memories),
            'successful_replays': sum(1 for r in replay_results if r.get('success', False)),
            'replay_effectiveness': effectiveness,
            'replay_time': replay_time,
            'total_replay_count': self.replay_count
        }
    
    def _select_replay_memories(self, current_context: Dict[str, Any]) -> List[EpisodicMemory]:
        """Select memories for replay based on importance and context."""
        if not self.episodic_memory:
            return []
        
        # Context-aware selection
        context_key = self._generate_context_key(current_context)
        contextual_indices = self.memory_index.get(context_key, [])
        
        # Importance-based selection
        memory_list = list(self.episodic_memory)
        
        if self.importance_sampling:
            # Sample based on importance scores
            importance_scores = np.array([m.importance_score for m in memory_list])
            
            # Add small probability for all memories to avoid complete neglect
            importance_scores += 0.1
            probabilities = importance_scores / np.sum(importance_scores)
            
            # Sample memories
            n_select = min(self.replay_batch_size, len(memory_list))
            selected_indices = np.random.choice(
                len(memory_list), 
                size=n_select, 
                replace=False, 
                p=probabilities
            )
            
            selected_memories = [memory_list[i] for i in selected_indices]
        else:
            # Random selection
            n_select = min(self.replay_batch_size, len(memory_list))
            selected_memories = np.random.choice(memory_list, size=n_select, replace=False)
        
        # Boost contextual memories
        if contextual_indices:
            # Replace some random memories with contextual ones
            n_contextual = min(len(contextual_indices), len(selected_memories) // 2)
            contextual_memory_indices = np.random.choice(contextual_indices, size=n_contextual, replace=False)
            contextual_memories = [self.episodic_memory[i] for i in contextual_memory_indices]
            
            # Replace random selections with contextual ones
            selected_memories[:len(contextual_memories)] = contextual_memories
        
        return selected_memories
    
    def _measure_replay_effectiveness(self, 
                                    pre_stats: Dict[str, Any],
                                    post_stats: Dict[str, Any]) -> float:
        """Measure effectiveness of experience replay."""
        # Simple effectiveness measure: improvement in key metrics
        
        effectiveness_factors = []
        
        # 1. Free energy reduction
        pre_fe = pre_stats.get('current_free_energy', 0.0)
        post_fe = post_stats.get('current_free_energy', 0.0)
        if pre_fe > 0:
            fe_improvement = max(0, (pre_fe - post_fe) / pre_fe)
            effectiveness_factors.append(fe_improvement)
        
        # 2. Belief confidence improvement
        pre_conf = pre_stats.get('belief_confidence', 0.5)
        post_conf = post_stats.get('belief_confidence', 0.5)
        conf_improvement = max(0, post_conf - pre_conf)
        effectiveness_factors.append(conf_improvement)
        
        # 3. Error rate improvement (inverse of health status)
        pre_health = 1.0 if pre_stats.get('health_status') == 'healthy' else 0.5
        post_health = 1.0 if post_stats.get('health_status') == 'healthy' else 0.5
        health_improvement = max(0, post_health - pre_health)
        effectiveness_factors.append(health_improvement)
        
        # Overall effectiveness
        if effectiveness_factors:
            effectiveness = np.mean(effectiveness_factors)
        else:
            effectiveness = 0.0
        
        return min(1.0, effectiveness)
    
    def adaptive_replay(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Perform adaptive replay based on current situation.
        
        Args:
            observation: Current observation
            
        Returns:
            Adaptive replay result
        """
        # Decide whether to replay based on current situation
        current_stats = self.base_agent.get_statistics()
        
        # Replay triggers
        should_replay = False
        replay_reason = "none"
        
        # 1. High uncertainty (need more experience)
        if current_stats.get('belief_confidence', 1.0) < 0.3:
            should_replay = True
            replay_reason = "high_uncertainty"
        
        # 2. High prediction error (encountering something difficult)
        elif current_stats.get('current_free_energy', 0.0) > 1.0:
            should_replay = True
            replay_reason = "high_prediction_error"
        
        # 3. Periodic replay
        elif time.time() - self.last_replay_time > self.replay_frequency:
            should_replay = True
            replay_reason = "periodic"
        
        # 4. Performance degradation
        elif current_stats.get('health_status') != 'healthy':
            should_replay = True
            replay_reason = "performance_degradation"
        
        if should_replay:
            # Create context from current observation
            context = {
                'observation_magnitude': float(np.linalg.norm(observation)),
                'replay_trigger': replay_reason
            }
            
            replay_result = self.experience_replay(context)
            replay_result['replay_reason'] = replay_reason
            replay_result['adaptive_replay'] = True
        else:
            replay_result = {
                'adaptive_replay': False,
                'replay_reason': replay_reason,
                'status': 'no_replay_needed'
            }
        
        return replay_result
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        # Memory utilization
        memory_utilization = len(self.episodic_memory) / self.memory_size
        self.memory_performance['memory_utilization'].append(memory_utilization)
        
        # Importance score distribution
        if self.episodic_memory:
            importance_scores = [m.importance_score for m in self.episodic_memory]
            avg_importance = np.mean(importance_scores)
            importance_std = np.std(importance_scores)
        else:
            avg_importance = importance_std = 0.0
        
        # Context diversity
        n_contexts = len(self.memory_index)
        
        # Replay statistics
        if self.episodic_memory:
            replay_counts = [m.replay_count for m in self.episodic_memory]
            avg_replay_count = np.mean(replay_counts)
            max_replay_count = max(replay_counts)
        else:
            avg_replay_count = max_replay_count = 0
        
        return {
            'memory_size': len(self.episodic_memory),
            'memory_capacity': self.memory_size,
            'memory_utilization': memory_utilization,
            'avg_importance_score': avg_importance,
            'importance_score_std': importance_std,
            'n_contexts': n_contexts,
            'total_replay_count': self.replay_count,
            'avg_memory_replay_count': avg_replay_count,
            'max_memory_replay_count': max_replay_count,
            'avg_replay_effectiveness': np.mean(self.memory_performance['replay_effectiveness']) if self.memory_performance['replay_effectiveness'] else 0,
            'time_since_last_replay': time.time() - self.last_replay_time
        }


class ProgressiveNeuralNetworks:
    """
    Progressive Neural Networks for hierarchical skill acquisition.
    
    Grows network capacity as new tasks are learned, preventing
    catastrophic forgetting while enabling transfer learning.
    """
    
    def __init__(self, base_agent: ActiveInferenceAgent):
        self.base_agent = base_agent
        self.logger = logging.getLogger("ProgressiveNeuralNetworks")
        
        # Network columns (one per task)
        self.network_columns: List[Dict[str, Any]] = []
        self.lateral_connections: Dict[str, List[float]] = {}
        
        # Task-specific components
        self.task_heads: Dict[str, Dict[str, Any]] = {}
        self.column_task_mapping: Dict[int, str] = {}
        
        # Progressive learning
        self.current_column = -1
        self.capacity_expansion_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.progressive_performance = {
            'transfer_benefits': [],
            'capacity_utilization': [],
            'lateral_connection_strengths': [],
            'task_interference': []
        }
    
    def add_new_task_column(self, task_id: str) -> Dict[str, Any]:
        """
        Add a new column for learning a new task.
        
        Args:
            task_id: Identifier for the new task
            
        Returns:
            Column creation result
        """
        start_time = time.time()
        
        # Create new network column
        column_index = len(self.network_columns)
        new_column = self._create_network_column(task_id, column_index)
        
        self.network_columns.append(new_column)
        self.column_task_mapping[column_index] = task_id
        self.current_column = column_index
        
        # Initialize lateral connections from previous columns
        if column_index > 0:
            lateral_connections = self._initialize_lateral_connections(column_index)
            self.lateral_connections[task_id] = lateral_connections
        
        # Create task-specific head
        task_head = self._create_task_head(task_id, column_index)
        self.task_heads[task_id] = task_head
        
        # Record capacity expansion
        expansion_record = {
            'task_id': task_id,
            'column_index': column_index,
            'expansion_time': start_time,
            'n_total_columns': len(self.network_columns),
            'lateral_connections_created': len(lateral_connections) if column_index > 0 else 0
        }
        self.capacity_expansion_history.append(expansion_record)
        
        creation_time = time.time() - start_time
        
        self.logger.info(f"Added new task column for {task_id} (column {column_index})")
        
        return {
            'task_id': task_id,
            'column_index': column_index,
            'n_total_columns': len(self.network_columns),
            'lateral_connections': len(lateral_connections) if column_index > 0 else 0,
            'creation_time': creation_time
        }
    
    def _create_network_column(self, task_id: str, column_index: int) -> Dict[str, Any]:
        """Create a new network column for the task."""
        # Simplified column structure
        column = {
            'task_id': task_id,
            'column_index': column_index,
            'parameters': self._initialize_column_parameters(),
            'frozen': False,  # New columns start unfrozen
            'creation_time': time.time(),
            'update_count': 0
        }
        
        return column
    
    def _initialize_column_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize parameters for a network column."""
        # Simplified parameter initialization
        # In practice, this would be more sophisticated
        
        parameters = {
            'weights_layer1': np.random.randn(8, 16) * 0.1,
            'biases_layer1': np.zeros(16),
            'weights_layer2': np.random.randn(16, 8) * 0.1,
            'biases_layer2': np.zeros(8),
            'output_weights': np.random.randn(8, 2) * 0.1,
            'output_biases': np.zeros(2)
        }
        
        return parameters
    
    def _initialize_lateral_connections(self, target_column_index: int) -> List[float]:
        """Initialize lateral connections from previous columns."""
        lateral_connections = []
        
        # Create connections from all previous columns
        for source_column_index in range(target_column_index):
            # Random initialization of lateral connection strength
            connection_strength = np.random.uniform(0.1, 0.5)
            lateral_connections.append(connection_strength)
        
        return lateral_connections
    
    def _create_task_head(self, task_id: str, column_index: int) -> Dict[str, Any]:
        """Create task-specific output head."""
        task_head = {
            'task_id': task_id,
            'column_index': column_index,
            'head_parameters': {
                'weights': np.random.randn(8, 2) * 0.1,  # Assume 8 features to 2 actions
                'biases': np.zeros(2)
            },
            'specialization_level': 0.0  # How specialized this head is
        }
        
        return task_head
    
    def progressive_forward_pass(self, 
                               observation: np.ndarray,
                               task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Forward pass through progressive network.
        
        Args:
            observation: Input observation
            task_id: Task identifier (uses current task if None)
            
        Returns:
            Progressive forward pass result
        """
        if not self.network_columns:
            # Fall back to base agent
            action = self.base_agent.act(observation)
            return {
                'action': action,
                'progressive_pass': False,
                'fallback_reason': 'no_columns'
            }
        
        start_time = time.time()
        
        # Determine target task
        if task_id is None:
            task_id = self.column_task_mapping.get(self.current_column, 'unknown')
        
        # Forward pass through all columns
        column_outputs = []
        for column_index, column in enumerate(self.network_columns):
            column_output = self._column_forward_pass(observation, column)
            column_outputs.append(column_output)
        
        # Apply lateral connections
        enhanced_outputs = self._apply_lateral_connections(column_outputs, task_id)
        
        # Generate final action using task head
        final_action = self._generate_action_from_outputs(enhanced_outputs, task_id)
        
        # Measure transfer benefits
        transfer_benefit = self._measure_transfer_benefit(column_outputs, enhanced_outputs)
        self.progressive_performance['transfer_benefits'].append(transfer_benefit)
        
        forward_time = time.time() - start_time
        
        return {
            'action': final_action,
            'progressive_pass': True,
            'n_columns_used': len(column_outputs),
            'transfer_benefit': transfer_benefit,
            'task_id': task_id,
            'forward_time': forward_time
        }
    
    def _column_forward_pass(self, observation: np.ndarray, column: Dict[str, Any]) -> np.ndarray:
        """Forward pass through a single column."""
        params = column['parameters']
        
        # Simple feedforward computation
        x = observation
        
        # Ensure input size matches first layer
        if x.shape[0] != params['weights_layer1'].shape[0]:
            # Pad or truncate to match
            if x.shape[0] < params['weights_layer1'].shape[0]:
                x = np.pad(x, (0, params['weights_layer1'].shape[0] - x.shape[0]))
            else:
                x = x[:params['weights_layer1'].shape[0]]
        
        # Layer 1
        x = np.dot(params['weights_layer1'].T, x) + params['biases_layer1']
        x = np.tanh(x)  # Activation function
        
        # Layer 2
        x = np.dot(params['weights_layer2'].T, x) + params['biases_layer2']
        x = np.tanh(x)
        
        return x
    
    def _apply_lateral_connections(self, 
                                 column_outputs: List[np.ndarray],
                                 task_id: str) -> List[np.ndarray]:
        """Apply lateral connections between columns."""
        if len(column_outputs) <= 1 or task_id not in self.lateral_connections:
            return column_outputs
        
        enhanced_outputs = []
        lateral_strengths = self.lateral_connections[task_id]
        
        for i, output in enumerate(column_outputs):
            enhanced_output = output.copy()
            
            # Add contributions from previous columns
            for j in range(i):
                if j < len(lateral_strengths):
                    lateral_strength = lateral_strengths[j]
                    lateral_contribution = lateral_strength * column_outputs[j]
                    
                    # Ensure shapes match
                    if lateral_contribution.shape == enhanced_output.shape:
                        enhanced_output += lateral_contribution
                    else:
                        # Resize lateral contribution
                        min_size = min(len(lateral_contribution), len(enhanced_output))
                        enhanced_output[:min_size] += lateral_contribution[:min_size]
            
            enhanced_outputs.append(enhanced_output)
        
        return enhanced_outputs
    
    def _generate_action_from_outputs(self, 
                                    column_outputs: List[np.ndarray],
                                    task_id: str) -> np.ndarray:
        """Generate final action from column outputs."""
        if not column_outputs:
            return np.zeros(2)  # Default action
        
        # Use the most recent (current task) column output
        primary_output = column_outputs[-1]
        
        # Apply task head if available
        if task_id in self.task_heads:
            task_head = self.task_heads[task_id]
            head_params = task_head['head_parameters']
            
            # Ensure output size matches head input
            if len(primary_output) != head_params['weights'].shape[0]:
                if len(primary_output) < head_params['weights'].shape[0]:
                    primary_output = np.pad(primary_output, (0, head_params['weights'].shape[0] - len(primary_output)))
                else:
                    primary_output = primary_output[:head_params['weights'].shape[0]]
            
            # Apply task head
            action = np.dot(head_params['weights'].T, primary_output) + head_params['biases']
        else:
            # Default action generation
            if len(primary_output) >= 2:
                action = primary_output[:2]
            else:
                action = np.pad(primary_output, (0, 2 - len(primary_output)))
        
        return action
    
    def _measure_transfer_benefit(self, 
                                original_outputs: List[np.ndarray],
                                enhanced_outputs: List[np.ndarray]) -> float:
        """Measure transfer learning benefit from lateral connections."""
        if len(original_outputs) != len(enhanced_outputs) or len(original_outputs) <= 1:
            return 0.0
        
        # Compare outputs with and without lateral connections
        transfer_benefits = []
        
        for orig, enhanced in zip(original_outputs, enhanced_outputs):
            # Measure difference
            improvement = np.linalg.norm(enhanced) - np.linalg.norm(orig)
            normalized_improvement = improvement / (np.linalg.norm(orig) + 1e-6)
            transfer_benefits.append(max(0, normalized_improvement))  # Only positive transfer
        
        return np.mean(transfer_benefits) if transfer_benefits else 0.0
    
    def freeze_previous_columns(self, except_task: Optional[str] = None) -> Dict[str, Any]:
        """
        Freeze parameters of previous columns to prevent catastrophic forgetting.
        
        Args:
            except_task: Task to exclude from freezing
            
        Returns:
            Freezing operation result
        """
        frozen_columns = []
        
        for column in self.network_columns:
            task_id = column['task_id']
            
            if except_task is None or task_id != except_task:
                column['frozen'] = True
                frozen_columns.append(task_id)
        
        self.logger.info(f"Frozen {len(frozen_columns)} columns: {frozen_columns}")
        
        return {
            'frozen_columns': frozen_columns,
            'n_frozen': len(frozen_columns),
            'n_active': len(self.network_columns) - len(frozen_columns)
        }
    
    def adaptive_lateral_connection_update(self, 
                                         task_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Adaptively update lateral connection strengths based on performance.
        
        Args:
            task_performance: Performance scores for different tasks
            
        Returns:
            Connection update result
        """
        if not self.lateral_connections or not task_performance:
            return {'status': 'no_updates_needed'}
        
        updates_made = []
        
        for task_id, connections in self.lateral_connections.items():
            if task_id in task_performance:
                performance = task_performance[task_id]
                
                # Update connection strengths based on performance
                for i, strength in enumerate(connections):
                    if performance > 0.7:
                        # Good performance: slightly increase connections
                        new_strength = min(1.0, strength * 1.05)
                    elif performance < 0.3:
                        # Poor performance: decrease connections
                        new_strength = max(0.1, strength * 0.95)
                    else:
                        # Average performance: no change
                        new_strength = strength
                    
                    connections[i] = new_strength
                
                updates_made.append(task_id)
        
        return {
            'tasks_updated': updates_made,
            'n_tasks_updated': len(updates_made)
        }
    
    def get_progressive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive progressive network statistics."""
        # Capacity utilization
        total_parameters = sum(
            sum(p.size for p in column['parameters'].values()) 
            for column in self.network_columns
        )
        
        # Active vs frozen columns
        frozen_columns = [col for col in self.network_columns if col.get('frozen', False)]
        active_columns = [col for col in self.network_columns if not col.get('frozen', False)]
        
        # Lateral connection statistics
        if self.lateral_connections:
            all_connections = []
            for connections in self.lateral_connections.values():
                all_connections.extend(connections)
            avg_connection_strength = np.mean(all_connections) if all_connections else 0
            connection_std = np.std(all_connections) if all_connections else 0
        else:
            avg_connection_strength = connection_std = 0
        
        return {
            'n_total_columns': len(self.network_columns),
            'n_frozen_columns': len(frozen_columns),
            'n_active_columns': len(active_columns),
            'current_column': self.current_column,
            'total_parameters': total_parameters,
            'n_lateral_connections': len(self.lateral_connections),
            'avg_lateral_connection_strength': avg_connection_strength,
            'lateral_connection_std': connection_std,
            'avg_transfer_benefit': np.mean(self.progressive_performance['transfer_benefits']) if self.progressive_performance['transfer_benefits'] else 0,
            'capacity_expansions': len(self.capacity_expansion_history),
            'tasks_learned': list(self.column_task_mapping.values())
        }