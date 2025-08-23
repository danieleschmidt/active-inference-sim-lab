"""
High-performance parallel processing for Active Inference systems.
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import queue


@dataclass
class ComputationTask:
    """Represents a computational task for parallel processing."""
    task_id: str
    task_type: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None


class ParallelInferenceEngine:
    """
    High-performance parallel inference engine for Active Inference.
    
    Supports multi-threaded belief updates, batch processing, and GPU acceleration.
    """
    
    def __init__(self,
                 max_workers: int = None,
                 batch_size: int = 32,
                 use_gpu: bool = False,
                 enable_caching: bool = True):
        """
        Initialize parallel inference engine.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            batch_size: Batch size for vectorized operations
            use_gpu: Enable GPU acceleration (requires CuPy)
            enable_caching: Enable computation caching
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.enable_caching = enable_caching
        
        # Initialize computation backend
        self._init_compute_backend()
        
        # Thread pools
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(self.max_workers, multiprocessing.cpu_count() or 1))
        
        # Task queue and cache
        self.task_queue = queue.PriorityQueue()
        self.cache = {} if enable_caching else None
        
        # Performance metrics
        self.total_computations = 0
        self.cache_hits = 0
        self.avg_computation_time = 0.0
        
        # Logging
        self.logger = logging.getLogger("ParallelInferenceEngine")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"ParallelInferenceEngine initialized: workers={self.max_workers}, "
                        f"gpu={self.use_gpu}, batch_size={self.batch_size}")
    
    def _init_compute_backend(self):
        """Initialize computation backend (CPU/GPU)."""
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp  # Use CuPy for GPU
                self.logger.info("GPU acceleration enabled with CuPy")
            except ImportError:
                self.logger.warning("CuPy not available, falling back to CPU")
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
    
    def parallel_belief_update(self,
                              observations: List[np.ndarray],
                              prior_beliefs: List[Dict[str, Any]],
                              models: List[Any]) -> List[Dict[str, Any]]:
        """
        Perform parallel belief updates for multiple agents/observations.
        
        Args:
            observations: List of observation arrays
            prior_beliefs: List of prior belief states
            models: List of generative models
            
        Returns:
            List of updated belief states
        """
        if len(observations) != len(prior_beliefs) or len(observations) != len(models):
            raise ValueError("Input lists must have the same length")
        
        start_time = time.time()
        
        # Create tasks for parallel execution
        tasks = []
        for i, (obs, beliefs, model) in enumerate(zip(observations, prior_beliefs, models)):
            task = ComputationTask(
                task_id=f"belief_update_{i}",
                task_type="belief_update",
                function=self._single_belief_update,
                args=(obs, beliefs, model),
                kwargs={},
                priority=0
            )
            tasks.append(task)
        
        # Execute tasks in parallel
        results = self._execute_parallel_tasks(tasks)
        
        # Record performance metrics
        computation_time = time.time() - start_time
        self._update_performance_metrics(computation_time, len(observations))
        
        self.logger.debug(f"Parallel belief update completed: {len(observations)} updates "
                         f"in {computation_time:.3f}s")
        
        return results
    
    def batch_inference(self,
                       observations: np.ndarray,
                       model: Any,
                       batch_size: Optional[int] = None) -> np.ndarray:
        """
        Perform batch inference with automatic batching and vectorization.
        
        Args:
            observations: Array of observations (N, obs_dim)
            model: Generative model for inference
            batch_size: Override default batch size
            
        Returns:
            Batch inference results
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        n_samples = observations.shape[0]
        results = []
        
        # Process in batches
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_obs = observations[batch_start:batch_end]
            
            # Vectorized batch computation
            batch_result = self._vectorized_inference(batch_obs, model)
            results.append(batch_result)
        
        # Concatenate results
        return self.xp.concatenate(results, axis=0)
    
    def parallel_planning(self,
                         belief_states: List[Dict[str, Any]],
                         models: List[Any],
                         horizons: List[int]) -> List[np.ndarray]:
        """
        Perform parallel action planning for multiple agents.
        
        Args:
            belief_states: List of current belief states
            models: List of generative models
            horizons: List of planning horizons
            
        Returns:
            List of optimal actions
        """
        # Create planning tasks
        tasks = []
        for i, (beliefs, model, horizon) in enumerate(zip(belief_states, models, horizons)):
            task = ComputationTask(
                task_id=f"planning_{i}",
                task_type="planning",
                function=self._single_planning,
                args=(beliefs, model, horizon),
                kwargs={},
                priority=1  # Higher priority for planning
            )
            tasks.append(task)
        
        # Execute in parallel
        return self._execute_parallel_tasks(tasks)
    
    def gpu_accelerated_computation(self,
                                  data: np.ndarray,
                                  computation_fn: Callable) -> np.ndarray:
        """
        Perform GPU-accelerated computation if available.
        
        Args:
            data: Input data
            computation_fn: Computation function
            
        Returns:
            Computed results
        """
        if not self.use_gpu:
            return computation_fn(data)
        
        # Transfer to GPU
        gpu_data = self.xp.array(data)
        
        # Compute on GPU
        gpu_result = computation_fn(gpu_data)
        
        # Transfer back to CPU
        return self.xp.asnumpy(gpu_result)
    
    def _execute_parallel_tasks(self, tasks: List[ComputationTask]) -> List[Any]:
        """Execute a list of tasks in parallel."""
        results = [None] * len(tasks)
        
        # Submit tasks to thread executor
        future_to_index = {}
        for i, task in enumerate(tasks):
            # Check cache first
            if self.enable_caching:
                cache_key = self._get_cache_key(task)
                if cache_key in self.cache:
                    results[i] = self.cache[cache_key]
                    self.cache_hits += 1
                    continue
            
            # Submit to executor
            future = self.thread_executor.submit(
                self._execute_task, task
            )
            future_to_index[future] = (i, task)
        
        # Collect results
        for future in as_completed(future_to_index.keys()):
            index, task = future_to_index[future]
            try:
                result = future.result(timeout=task.timeout)
                results[index] = result
                
                # Cache result
                if self.enable_caching:
                    cache_key = self._get_cache_key(task)
                    self.cache[cache_key] = result
                    
            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed: {e}")
                results[index] = None
        
        return results
    
    def _execute_task(self, task: ComputationTask) -> Any:
        """Execute a single computational task."""
        try:
            return task.function(*task.args, **task.kwargs)
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            raise
    
    def _single_belief_update(self,
                             observation: np.ndarray,
                             prior_beliefs: Dict[str, Any],
                             model: Any) -> Dict[str, Any]:
        """Single threaded belief update (to be called in parallel)."""
        # Simplified belief update - replace with actual implementation
        updated_beliefs = prior_beliefs.copy()
        
        # Simulate belief update computation
        if 'mean' in updated_beliefs and 'variance' in updated_beliefs:
            # Simple Kalman filter-like update
            prior_mean = updated_beliefs['mean']
            prior_var = updated_beliefs['variance']
            
            # Observation model (simplified)
            obs_var = 0.1
            innovation = observation[:len(prior_mean)] - prior_mean
            
            # Update
            gain = prior_var / (prior_var + obs_var)
            updated_beliefs['mean'] = prior_mean + gain * innovation
            updated_beliefs['variance'] = (1 - gain) * prior_var
        
        return updated_beliefs
    
    def _single_planning(self,
                        belief_state: Dict[str, Any],
                        model: Any,
                        horizon: int) -> np.ndarray:
        """Single threaded planning (to be called in parallel)."""
        # Simplified planning - replace with actual implementation
        action_dim = getattr(model, 'action_dim', 2)
        
        # Random planning for demonstration
        return np.random.randn(action_dim) * 0.1
    
    def _vectorized_inference(self,
                             batch_observations: np.ndarray,
                             model: Any) -> np.ndarray:
        """Vectorized inference for batch processing."""
        # Simplified vectorized inference
        batch_size, obs_dim = batch_observations.shape
        
        # Simulate inference computation
        if self.use_gpu:
            # GPU computation
            gpu_obs = self.xp.array(batch_observations)
            # Simplified: just normalize observations
            gpu_result = gpu_obs / (self.xp.linalg.norm(gpu_obs, axis=1, keepdims=True) + 1e-8)
            return self.xp.asnumpy(gpu_result)
        else:
            # CPU computation
            return batch_observations / (np.linalg.norm(batch_observations, axis=1, keepdims=True) + 1e-8)
    
    def _get_cache_key(self, task: ComputationTask) -> str:
        """Generate cache key for a task."""
        # Simple cache key based on task type and hashed arguments
        args_hash = hash(str(task.args) + str(task.kwargs))
        return f"{task.task_type}_{args_hash}"
    
    def _update_performance_metrics(self, computation_time: float, n_operations: int):
        """Update performance tracking metrics."""
        self.total_computations += n_operations
        
        # Exponential moving average for computation time
        alpha = 0.1
        self.avg_computation_time = (alpha * computation_time + 
                                   (1 - alpha) * self.avg_computation_time)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        cache_hit_rate = (self.cache_hits / max(1, self.total_computations)) * 100
        
        return {
            'total_computations': self.total_computations,
            'cache_hits': self.cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'avg_computation_time': self.avg_computation_time,
            'using_gpu': self.use_gpu,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }
    
    def clear_cache(self):
        """Clear computation cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Computation cache cleared")
    
    def shutdown(self):
        """Shutdown thread pools."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        self.logger.info("ParallelInferenceEngine shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class AdaptiveBatchProcessor:
    """
    Adaptive batch processing with dynamic batch size optimization.
    """
    
    def __init__(self,
                 initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 1024,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive batch processor.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            adaptation_rate: Rate of batch size adaptation
        """
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.batch_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=50)
        
        # Logging
        self.logger = logging.getLogger("AdaptiveBatchProcessor")
    
    def process_batch(self,
                     data: List[Any],
                     processing_fn: Callable) -> List[Any]:
        """
        Process data with adaptive batching.
        
        Args:
            data: List of data items to process
            processing_fn: Function to process each batch
            
        Returns:
            List of processed results
        """
        if not data:
            return []
        
        results = []
        total_items = len(data)
        processed_items = 0
        
        while processed_items < total_items:
            # Determine current batch
            batch_end = min(processed_items + self.batch_size, total_items)
            batch_data = data[processed_items:batch_end]
            
            # Process batch and measure time
            start_time = time.time()
            batch_results = processing_fn(batch_data)
            batch_time = time.time() - start_time
            
            # Record metrics
            self.batch_times.append(batch_time)
            batch_throughput = len(batch_data) / (batch_time + 1e-8)
            self.throughput_history.append(batch_throughput)
            
            results.extend(batch_results)
            processed_items = batch_end
            
            # Adapt batch size
            self._adapt_batch_size()
        
        return results
    
    def _adapt_batch_size(self):
        """Adapt batch size based on performance history."""
        if len(self.throughput_history) < 5:
            return  # Need more data points
        
        # Calculate recent performance trend
        recent_throughput = list(self.throughput_history)[-5:]
        avg_throughput = np.mean(recent_throughput)
        throughput_trend = np.mean(np.diff(recent_throughput))
        
        # Adapt based on trend
        if throughput_trend > 0:
            # Performance improving, try larger batch
            new_batch_size = int(self.batch_size * (1 + self.adaptation_rate))
        else:
            # Performance declining, try smaller batch
            new_batch_size = int(self.batch_size * (1 - self.adaptation_rate))
        
        # Clamp to bounds
        self.batch_size = max(self.min_batch_size, 
                            min(new_batch_size, self.max_batch_size))
        
        self.logger.debug(f"Adapted batch size to {self.batch_size} "
                         f"(throughput: {avg_throughput:.2f} items/s)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch processing metrics."""
        if not self.batch_times:
            return {'batch_size': self.batch_size, 'avg_throughput': 0}
        
        return {
            'current_batch_size': self.batch_size,
            'avg_batch_time': np.mean(self.batch_times),
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'total_batches_processed': len(self.batch_times)
        }