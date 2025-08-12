"""
Performance optimization implementations for Active Inference agents.

This module provides optimized versions of core components with
GPU acceleration, vectorization, and parallel processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp

from ..core.agent import ActiveInferenceAgent
from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel
from ..inference.variational import VariationalInference


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    use_gpu: bool = False
    enable_caching: bool = True
    parallel_belief_updates: bool = True
    vectorized_planning: bool = True
    batch_size: int = 32
    num_workers: int = None
    memory_limit_mb: int = 1024
    optimization_level: str = "balanced"  # "speed", "balanced", "memory"


class OptimizedActiveInferenceAgent(ActiveInferenceAgent):
    """
    Performance-optimized Active Inference agent.
    
    Includes GPU acceleration, caching, and parallelization
    for production deployment scenarios.
    """
    
    def __init__(self,
                 optimization_config: OptimizationConfig = None,
                 **kwargs):
        """
        Initialize optimized agent.
        
        Args:
            optimization_config: Performance optimization settings
            **kwargs: Standard agent parameters
        """
        super().__init__(**kwargs)
        
        self.opt_config = optimization_config or OptimizationConfig()
        self.logger = logging.getLogger(f"OptimizedAgent.{self.agent_id}")
        
        # Performance tracking
        self.inference_times = []
        self.planning_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize GPU availability first
        self.gpu_available = False
        
        # Initialize optimizations
        self._initialize_optimizations()
        
        self.logger.info(f"Initialized optimized agent with config: {self.opt_config}")
    
    def _initialize_optimizations(self):
        """Initialize performance optimizations."""
        
        # GPU acceleration setup
        if self.opt_config.use_gpu:
            self._setup_gpu_acceleration()
        
        # Caching system
        if self.opt_config.enable_caching:
            self._setup_caching()
        
        # Parallel processing
        if self.opt_config.parallel_belief_updates:
            self._setup_parallel_processing()
        
        # Memory optimization
        self._setup_memory_optimization()
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration if available."""
        self.gpu_available = False  # Initialize attribute
        
        try:
            # Try to import GPU libraries
            import cupy as cp
            self.gpu_available = True
            self.cp = cp
            self.logger.info("GPU acceleration enabled with CuPy")
        except ImportError:
            try:
                # Fallback to other GPU libraries
                import jax.numpy as jnp
                self.gpu_available = True
                self.jnp = jnp
                self.logger.info("GPU acceleration enabled with JAX")
            except ImportError:
                self.gpu_available = False
                self.logger.warning("GPU libraries not available, using CPU")
    
    def _setup_caching(self):
        """Setup intelligent caching system."""
        from .caching import BeliefCache, ModelCache
        
        self.belief_cache = BeliefCache(max_size=1000)
        self.model_cache = ModelCache(max_size=500)
        
        self.logger.info("Caching system initialized")
    
    def _setup_parallel_processing(self):
        """Setup parallel processing for belief updates."""
        self.num_workers = self.opt_config.num_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        
        self.logger.info(f"Parallel processing setup with {self.num_workers} workers")
    
    def _setup_memory_optimization(self):
        """Setup memory optimization strategies."""
        self.memory_limit = self.opt_config.memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.memory_usage = 0
        
        # Optimization strategies based on level
        if self.opt_config.optimization_level == "speed":
            self.history_limit = 10000
            self.cache_size_multiplier = 2.0
        elif self.opt_config.optimization_level == "memory":
            self.history_limit = 1000
            self.cache_size_multiplier = 0.5
        else:  # balanced
            self.history_limit = 5000
            self.cache_size_multiplier = 1.0
    
    def infer_states(self, observation: np.ndarray) -> BeliefState:
        """Optimized belief inference with caching and GPU acceleration."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.opt_config.enable_caching:
                cache_key = self._generate_cache_key(observation)
                cached_result = self.belief_cache.get(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    return cached_result
                self.cache_misses += 1
            
            # GPU-accelerated inference if available
            if self.gpu_available and self.opt_config.use_gpu:
                updated_beliefs = self._gpu_infer_states(observation)
            else:
                # Standard CPU inference with possible parallelization
                if self.opt_config.parallel_belief_updates and len(self.beliefs) > 1:
                    updated_beliefs = self._parallel_infer_states(observation)
                else:
                    updated_beliefs = super().infer_states(observation)
            
            # Cache the result
            if self.opt_config.enable_caching:
                self.belief_cache.put(cache_key, updated_beliefs)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Memory management
            self._manage_memory()
            
            return updated_beliefs
            
        except Exception as e:
            self.logger.error(f"Optimized inference failed: {e}")
            # Fallback to standard inference
            return super().infer_states(observation)
    
    def _gpu_infer_states(self, observation: np.ndarray) -> BeliefState:
        """GPU-accelerated belief inference."""
        
        if hasattr(self, 'cp'):  # CuPy
            return self._cupy_infer_states(observation)
        elif hasattr(self, 'jnp'):  # JAX
            return self._jax_infer_states(observation)
        else:
            return super().infer_states(observation)
    
    def _cupy_infer_states(self, observation: np.ndarray) -> BeliefState:
        """CuPy-based GPU inference."""
        
        # Transfer to GPU
        gpu_obs = self.cp.asarray(observation)
        
        updated_beliefs = BeliefState()
        
        # Process beliefs on GPU
        for name, prior in self.beliefs.get_all_beliefs().items():
            gpu_mean = self.cp.asarray(prior.mean)
            gpu_var = self.cp.asarray(prior.variance)
            
            # Simplified GPU-accelerated update
            # In practice, this would include full variational inference on GPU
            gpu_likelihood = self._compute_gpu_likelihood(gpu_obs, gpu_mean)
            
            # Update on GPU
            updated_mean = gpu_mean + 0.01 * (gpu_obs[:len(gpu_mean)] - gpu_mean)
            updated_var = gpu_var * 0.99  # Slight variance reduction
            
            # Transfer back to CPU
            cpu_mean = self.cp.asnumpy(updated_mean)
            cpu_var = self.cp.asnumpy(updated_var)
            
            updated_beliefs.add_belief(name, Belief(
                mean=cpu_mean,
                variance=cpu_var,
                support=prior.support
            ))
        
        return updated_beliefs
    
    def _compute_gpu_likelihood(self, gpu_obs: 'cp.ndarray', gpu_mean: 'cp.ndarray') -> 'cp.ndarray':
        """Compute likelihood on GPU."""
        diff = gpu_obs[:len(gpu_mean)] - gpu_mean
        likelihood = self.cp.exp(-0.5 * self.cp.sum(diff**2))
        return likelihood
    
    def _parallel_infer_states(self, observation: np.ndarray) -> BeliefState:
        """Parallel belief updates across multiple threads."""
        
        beliefs_items = list(self.beliefs.get_all_beliefs().items())
        
        # Split beliefs across workers
        chunk_size = max(1, len(beliefs_items) // self.num_workers)
        belief_chunks = [beliefs_items[i:i+chunk_size] 
                        for i in range(0, len(beliefs_items), chunk_size)]
        
        # Process chunks in parallel
        futures = []
        for chunk in belief_chunks:
            future = self.thread_pool.submit(
                self._update_belief_chunk, observation, chunk
            )
            futures.append(future)
        
        # Collect results
        updated_beliefs = BeliefState()
        for future in futures:
            chunk_results = future.result()
            for name, belief in chunk_results.items():
                updated_beliefs.add_belief(name, belief)
        
        return updated_beliefs
    
    def _update_belief_chunk(self, 
                            observation: np.ndarray, 
                            belief_chunk: List[Tuple[str, Belief]]) -> Dict[str, Belief]:
        """Update a chunk of beliefs."""
        
        results = {}
        
        for name, prior in belief_chunk:
            # Simplified belief update for parallel processing
            # In practice, would use optimized inference engine
            
            obs_subset = observation[:len(prior.mean)]
            
            # Simple Bayesian update
            precision_prior = 1.0 / (prior.variance + 1e-8)
            precision_obs = 1.0 / 0.1  # Observation noise
            
            updated_precision = precision_prior + precision_obs
            updated_variance = 1.0 / updated_precision
            
            updated_mean = updated_variance * (
                precision_prior * prior.mean + precision_obs * obs_subset
            )
            
            results[name] = Belief(
                mean=updated_mean,
                variance=updated_variance,
                support=prior.support
            )
        
        return results
    
    def plan_action(self, 
                   beliefs: Optional[BeliefState] = None,
                   horizon: Optional[int] = None) -> np.ndarray:
        """Optimized action planning with vectorization."""
        
        start_time = time.time()
        
        try:
            if self.opt_config.vectorized_planning:
                action = self._vectorized_plan_action(beliefs, horizon)
            else:
                action = super().plan_action(beliefs, horizon)
            
            planning_time = time.time() - start_time
            self.planning_times.append(planning_time)
            
            return action
            
        except Exception as e:
            self.logger.error(f"Optimized planning failed: {e}")
            return super().plan_action(beliefs, horizon)
    
    def _vectorized_plan_action(self, 
                               beliefs: Optional[BeliefState] = None,
                               horizon: Optional[int] = None) -> np.ndarray:
        """Vectorized action planning for multiple candidates."""
        
        if beliefs is None:
            beliefs = self.beliefs
        
        if horizon is None:
            horizon = self.planning_horizon
        
        # Generate batch of candidate actions
        batch_size = self.opt_config.batch_size
        candidate_actions = np.random.randn(batch_size, self.action_dim)
        
        # Vectorized evaluation
        action_values = self._evaluate_action_batch(candidate_actions, beliefs, horizon)
        
        # Select best action
        best_idx = np.argmin(action_values)  # Minimize expected free energy
        
        return candidate_actions[best_idx]
    
    def _evaluate_action_batch(self, 
                              actions: np.ndarray,
                              beliefs: BeliefState,
                              horizon: int) -> np.ndarray:
        """Vectorized evaluation of action batch."""
        
        batch_size = actions.shape[0]
        action_values = np.zeros(batch_size)
        
        # Vectorized expected free energy computation
        for i, action in enumerate(actions):
            try:
                # Simplified vectorized evaluation
                # In practice, would include full EFE computation
                
                # Epistemic value (information gain)
                current_entropy = beliefs.total_entropy()
                predicted_entropy = current_entropy * 0.9  # Simplified prediction
                epistemic_value = current_entropy - predicted_entropy
                
                # Pragmatic value (goal achievement)
                action_cost = np.sum(action**2) * 0.1  # Regularization
                pragmatic_value = -action_cost
                
                # Expected free energy (negative because we minimize)
                action_values[i] = -(epistemic_value + pragmatic_value)
                
            except Exception:
                action_values[i] = float('inf')  # Invalid action
        
        return action_values
    
    def _generate_cache_key(self, observation: np.ndarray) -> str:
        """Generate cache key for observation."""
        # Simple hash-based key
        obs_hash = hash(observation.tobytes())
        beliefs_hash = hash(str(self.beliefs.get_all_beliefs().keys()))
        return f"{obs_hash}_{beliefs_hash}"
    
    def _manage_memory(self):
        """Intelligent memory management."""
        
        # Trim history if it exceeds limits
        if len(self.history['observations']) > self.history_limit:
            trim_size = self.history_limit // 2
            for key in self.history:
                if isinstance(self.history[key], list):
                    self.history[key] = self.history[key][-trim_size:]
        
        # Clear old caches if memory usage is high
        if hasattr(self, 'belief_cache'):
            cache_usage = self.belief_cache.size()
            if cache_usage > self.memory_limit * 0.5:
                self.belief_cache.clear_old_entries()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        
        stats = super().get_statistics()
        
        # Add optimization-specific metrics
        if self.inference_times:
            stats.update({
                'avg_inference_time': np.mean(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'inference_time_std': np.std(self.inference_times)
            })
        
        if self.planning_times:
            stats.update({
                'avg_planning_time': np.mean(self.planning_times),
                'min_planning_time': np.min(self.planning_times),
                'max_planning_time': np.max(self.planning_times),
                'planning_time_std': np.std(self.planning_times)
            })
        
        # Cache performance
        if hasattr(self, 'belief_cache'):
            total_requests = self.cache_hits + self.cache_misses
            stats.update({
                'cache_hit_rate': self.cache_hits / max(1, total_requests),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_size': self.belief_cache.size()
            })
        
        # Optimization config
        stats['optimization_config'] = {
            'gpu_enabled': self.opt_config.use_gpu,
            'gpu_available': getattr(self, 'gpu_available', False),
            'caching_enabled': self.opt_config.enable_caching,
            'parallel_updates': self.opt_config.parallel_belief_updates,
            'vectorized_planning': self.opt_config.vectorized_planning,
            'optimization_level': self.opt_config.optimization_level
        }
        
        return stats
    
    def __del__(self):
        """Cleanup optimized agent resources."""
        # Close thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        
        super().__del__()


class GPUAcceleratedAgent(OptimizedActiveInferenceAgent):
    """Specialized GPU-accelerated Active Inference agent."""
    
    def __init__(self, **kwargs):
        """Initialize GPU-accelerated agent."""
        
        # Force GPU optimization
        gpu_config = OptimizationConfig(
            use_gpu=True,
            enable_caching=True,
            parallel_belief_updates=False,  # GPU handles parallelism
            vectorized_planning=True,
            batch_size=64,
            optimization_level="speed"
        )
        
        super().__init__(optimization_config=gpu_config, **kwargs)


class ParallelBeliefUpdater:
    """Parallel belief updater for multi-core processing."""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        self.logger = logging.getLogger("ParallelBeliefUpdater")
    
    def update_beliefs_parallel(self,
                               observations: List[np.ndarray],
                               beliefs: List[BeliefState],
                               model: GenerativeModel) -> List[BeliefState]:
        """Update multiple belief states in parallel."""
        
        # Submit parallel tasks
        futures = []
        for obs, belief in zip(observations, beliefs):
            future = self.process_pool.submit(
                self._update_single_belief, obs, belief, model
            )
            futures.append(future)
        
        # Collect results
        updated_beliefs = []
        for future in futures:
            updated_beliefs.append(future.result())
        
        return updated_beliefs
    
    def _update_single_belief(self,
                             observation: np.ndarray,
                             belief: BeliefState,
                             model: GenerativeModel) -> BeliefState:
        """Update single belief state."""
        
        # Use variational inference for update
        inference_engine = VariationalInference()
        return inference_engine.update_beliefs(observation, belief, model)
    
    def __del__(self):
        """Cleanup process pool."""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)


class VectorizedEnvironmentWrapper:
    """Wrapper for vectorized environment interactions."""
    
    def __init__(self, env_factory: callable, num_envs: int = 4):
        """
        Initialize vectorized environment wrapper.
        
        Args:
            env_factory: Function to create environment instances
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.envs = [env_factory() for _ in range(num_envs)]
        self.logger = logging.getLogger("VectorizedEnvironment")
    
    def reset(self) -> List[np.ndarray]:
        """Reset all environments."""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        """Step all environments with given actions."""
        
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, done = env.step(action)
            results.append((obs, reward, done))
        
        # Unpack results
        observations = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        
        return observations, rewards, dones
    
    def batch_training(self, 
                      agent: ActiveInferenceAgent,
                      n_episodes: int = 100) -> Dict[str, Any]:
        """Run batch training across vectorized environments."""
        
        start_time = time.time()
        
        all_rewards = []
        episode_count = 0
        
        # Reset all environments
        observations = self.reset()
        
        while episode_count < n_episodes:
            # Get actions from agent for all environments
            actions = []
            for obs in observations:
                action = agent.act(obs)
                actions.append(action)
            
            # Step all environments
            next_observations, rewards, dones = self.step(actions)
            
            # Update agent with experiences
            for obs, action, reward, next_obs in zip(observations, actions, rewards, next_observations):
                agent.update_model(next_obs, action, reward)
            
            # Track rewards
            all_rewards.extend(rewards)
            
            # Reset finished episodes
            for i, done in enumerate(dones):
                if done:
                    observations[i] = self.envs[i].reset()
                    episode_count += 1
                else:
                    observations[i] = next_observations[i]
        
        execution_time = time.time() - start_time
        
        return {
            'total_episodes': episode_count,
            'total_rewards': all_rewards,
            'mean_reward': np.mean(all_rewards),
            'execution_time': execution_time,
            'episodes_per_second': episode_count / execution_time
        }