"""
Performance Optimization Module for Active Inference Framework

This module provides advanced performance optimization capabilities for the
Active Inference framework including intelligent caching, memory pools,
and computational optimizations.

Author: Terragon Labs
Generation: 3 (MAKE IT SCALE)
"""

import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import functools
import weakref
from collections import OrderedDict, deque
import queue
import hashlib
import pickle


class LRUCache:
    """Thread-safe Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'max_size': self.max_size
            }


class MemoryPool:
    """Memory pool for efficient numpy array allocation."""
    
    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.pools = {}  # shape -> deque of arrays
        self.lock = threading.RLock()
        self.allocations = 0
        self.pool_hits = 0
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        key = (shape, dtype)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                array = self.pools[key].popleft()
                array.fill(0)  # Clear the array
                self.pool_hits += 1
                return array
            else:
                self.allocations += 1
                return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        key = (array.shape, array.dtype)
        
        with self.lock:
            if key not in self.pools:
                self.pools[key] = deque(maxlen=self.pool_size)
            
            if len(self.pools[key]) < self.pool_size:
                self.pools[key].append(array)
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.allocations + self.pool_hits
            pool_hit_rate = self.pool_hits / total if total > 0 else 0.0
            return {
                'allocations': self.allocations,
                'pool_hits': self.pool_hits,
                'pool_hit_rate': pool_hit_rate,
                'active_pools': len(self.pools)
            }


class ComputationBatcher:
    """Batches similar computations for improved efficiency."""
    
    def __init__(self, batch_size: int = 32, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_batches = {}  # function_name -> list of (args, future)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def batch_compute(self, func: Callable, func_key: str, *args) -> Future:
        """Submit computation to be batched."""
        future = Future()
        
        with self.lock:
            if func_key not in self.pending_batches:
                self.pending_batches[func_key] = []
            
            self.pending_batches[func_key].append((args, future))
            
            # Check if batch is ready
            if len(self.pending_batches[func_key]) >= self.batch_size:
                self._execute_batch(func, func_key)
        
        return future
    
    def _execute_batch(self, func: Callable, func_key: str) -> None:
        """Execute a batch of computations."""
        batch = self.pending_batches.pop(func_key, [])
        
        if batch:
            def batch_worker():
                try:
                    # Extract args and batch them
                    all_args = [args for args, _ in batch]
                    futures = [future for _, future in batch]
                    
                    # Execute batched computation
                    results = func(all_args)
                    
                    # Set results
                    for future, result in zip(futures, results):
                        future.set_result(result)
                        
                except Exception as e:
                    # Set exception for all futures
                    for _, future in batch:
                        future.set_exception(e)
            
            self.executor.submit(batch_worker)


class AdaptiveOptimizer:
    """Adaptive optimization system that learns from performance patterns."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.optimization_strategies = {
            'cache_size': {'min': 100, 'max': 10000, 'current': 1000},
            'batch_size': {'min': 8, 'max': 128, 'current': 32},
            'thread_pool_size': {'min': 2, 'max': 16, 'current': 4}
        }
        self.lock = threading.RLock()
    
    def record_performance(self, operation: str, duration: float, success: bool) -> None:
        """Record performance metrics for optimization."""
        with self.lock:
            self.performance_history.append({
                'operation': operation,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
    
    def suggest_optimizations(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations."""
        with self.lock:
            if len(self.performance_history) < 10:
                return {}
            
            # Analyze recent performance
            recent_metrics = list(self.performance_history)[-100:]
            avg_duration = np.mean([m['duration'] for m in recent_metrics])
            success_rate = np.mean([m['success'] for m in recent_metrics])
            
            suggestions = {}
            
            # Suggest cache size adjustment
            if avg_duration > 1.0:  # Slow operations
                suggestions['cache_size'] = min(
                    self.optimization_strategies['cache_size']['current'] * 1.5,
                    self.optimization_strategies['cache_size']['max']
                )
            
            # Suggest batch size adjustment
            if success_rate > 0.95 and avg_duration < 0.1:  # Fast, successful operations
                suggestions['batch_size'] = min(
                    self.optimization_strategies['batch_size']['current'] * 1.2,
                    self.optimization_strategies['batch_size']['max']
                )
            
            return suggestions


class PerformanceOptimizedActiveInference:
    """
    Performance-optimized Active Inference agent with advanced caching,
    memory management, and computational optimizations.
    """
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize optimization components
        self.cache = LRUCache(max_size=1000)
        self.memory_pool = MemoryPool(pool_size=100)
        self.batcher = ComputationBatcher(batch_size=32)
        self.optimizer = AdaptiveOptimizer()
        
        # Thread pool for parallel computation
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.operation_times = {}
        self.lock = threading.RLock()
        
        # Initialize model parameters with memory pool
        self.beliefs = self.memory_pool.get_array((state_dim,))
        self.preferences = self.memory_pool.get_array((state_dim,))
        self.precision = 1.0
        
        # Cached computations
        self._cached_transitions = {}
        self._cached_observations = {}
    
    def _cache_key(self, prefix: str, *args) -> str:
        """Generate cache key for given arguments."""
        key_data = f"{prefix}_{args}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def update_beliefs(self, observation: np.ndarray) -> np.ndarray:
        """Update beliefs with caching and optimization."""
        start_time = time.time()
        operation_name = 'belief_update'
        
        try:
            cache_key = self._cache_key("belief", observation.tobytes())
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Perform belief update computation
            prediction_error = observation - self.beliefs
            beliefs_update = self.precision * prediction_error
            
            # Use memory pool for intermediate computations
            temp_array = self.memory_pool.get_array(self.beliefs.shape)
            temp_array[:] = self.beliefs + 0.1 * beliefs_update
            
            # Cache result
            result = temp_array.copy()
            self.cache.put(cache_key, result)
            
            # Return array to pool
            self.memory_pool.return_array(temp_array)
            
            # Record timing
            duration = time.time() - start_time
            self.optimizer.record_performance(operation_name, duration, True)
            
            with self.lock:
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = deque(maxlen=100)
                self.operation_times[operation_name].append(duration)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.optimizer.record_performance(operation_name, duration, False)
            raise
    
    def select_action(self, beliefs: np.ndarray) -> np.ndarray:
        """Select action with performance optimization."""
        start_time = time.time()
        operation_name = 'action_selection'
        
        try:
            cache_key = self._cache_key("action", beliefs.tobytes())
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Compute expected free energy for each action
            expected_free_energies = self.memory_pool.get_array((self.action_dim,))
            
            for i in range(self.action_dim):
                # Simulate action consequences
                action_vector = self.memory_pool.get_array((self.action_dim,))
                action_vector[i] = 1.0
                
                # Compute free energy (simplified)
                free_energy = np.sum(np.square(beliefs - self.preferences)) + 0.1 * i
                expected_free_energies[i] = free_energy
                
                self.memory_pool.return_array(action_vector)
            
            # Select action with minimum expected free energy
            best_action_idx = np.argmin(expected_free_energies)
            action = self.memory_pool.get_array((self.action_dim,))
            action[best_action_idx] = 1.0
            
            # Cache result
            result = action.copy()
            self.cache.put(cache_key, result)
            
            # Return arrays to pool
            self.memory_pool.return_array(expected_free_energies)
            self.memory_pool.return_array(action)
            
            # Record timing
            duration = time.time() - start_time
            self.optimizer.record_performance(operation_name, duration, True)
            
            with self.lock:
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = deque(maxlen=100)
                self.operation_times[operation_name].append(duration)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.optimizer.record_performance(operation_name, duration, False)
            raise
    
    def parallel_inference(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """Perform parallel inference on multiple observations."""
        def process_observation(obs):
            return self.update_beliefs(obs)
        
        # Use thread pool for parallel processing
        futures = [self.thread_pool.submit(process_observation, obs) 
                  for obs in observations]
        
        results = [future.result() for future in futures]
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.lock:
            stats = {
                'cache_stats': self.cache.stats(),
                'memory_pool_stats': self.memory_pool.stats(),
                'operation_times': {},
                'optimization_suggestions': self.optimizer.suggest_optimizations()
            }
            
            # Compute operation statistics
            for op_name, times in self.operation_times.items():
                if times:
                    stats['operation_times'][op_name] = {
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'count': len(times)
                    }
            
            return stats
    
    def optimize_configuration(self) -> None:
        """Apply optimization suggestions."""
        suggestions = self.optimizer.suggest_optimizations()
        
        if 'cache_size' in suggestions:
            new_cache = LRUCache(max_size=int(suggestions['cache_size']))
            # Migrate existing cache entries
            with self.cache.lock:
                for key, value in self.cache.cache.items():
                    new_cache.put(key, value)
            self.cache = new_cache
        
        if 'batch_size' in suggestions:
            self.batcher.batch_size = int(suggestions['batch_size'])
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.cache.clear()


class ScalableActiveInferenceFramework:
    """
    Scalable framework managing multiple optimized Active Inference agents
    with load balancing and auto-scaling capabilities.
    """
    
    def __init__(self, initial_agents: int = 4, max_agents: int = 16):
        self.agents = []
        self.max_agents = max_agents
        self.load_balancer_idx = 0
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.request_queue = queue.Queue()
        self.performance_monitor = threading.Thread(target=self._monitor_performance)
        self.performance_monitor.daemon = True
        self.performance_monitor.start()
        
        # Initialize agents
        for _ in range(initial_agents):
            agent = PerformanceOptimizedActiveInference()
            self.agents.append(agent)
    
    def get_agent(self) -> PerformanceOptimizedActiveInference:
        """Get agent using round-robin load balancing."""
        with self.lock:
            agent = self.agents[self.load_balancer_idx]
            self.load_balancer_idx = (self.load_balancer_idx + 1) % len(self.agents)
            return agent
    
    def scale_up(self) -> bool:
        """Add new agent if under capacity."""
        with self.lock:
            if len(self.agents) < self.max_agents:
                new_agent = PerformanceOptimizedActiveInference()
                self.agents.append(new_agent)
                return True
            return False
    
    def scale_down(self) -> bool:
        """Remove agent if overprovisioned."""
        with self.lock:
            if len(self.agents) > 1:
                agent = self.agents.pop()
                agent.cleanup()
                return True
            return False
    
    def process_request(self, observation: np.ndarray) -> np.ndarray:
        """Process inference request with load balancing."""
        self.request_queue.put(time.time())
        agent = self.get_agent()
        beliefs = agent.update_beliefs(observation)
        action = agent.select_action(beliefs)
        return action
    
    def _monitor_performance(self) -> None:
        """Monitor performance and trigger auto-scaling."""
        while True:
            try:
                # Check request rate
                current_time = time.time()
                recent_requests = []
                
                # Collect recent requests
                while not self.request_queue.empty():
                    try:
                        request_time = self.request_queue.get_nowait()
                        if current_time - request_time < 60:  # Last minute
                            recent_requests.append(request_time)
                    except queue.Empty:
                        break
                
                requests_per_minute = len(recent_requests)
                
                # Auto-scaling logic
                if requests_per_minute > len(self.agents) * 10:  # High load
                    self.scale_up()
                elif requests_per_minute < len(self.agents) * 2:  # Low load
                    self.scale_down()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception:
                time.sleep(30)
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """Get framework-wide statistics."""
        with self.lock:
            agent_stats = []
            for i, agent in enumerate(self.agents):
                stats = agent.get_performance_stats()
                stats['agent_id'] = i
                agent_stats.append(stats)
            
            return {
                'total_agents': len(self.agents),
                'max_agents': self.max_agents,
                'agent_stats': agent_stats,
                'load_balancer_position': self.load_balancer_idx
            }
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        for agent in self.agents:
            agent.cleanup()


def performance_benchmark(agent: PerformanceOptimizedActiveInference, 
                         num_iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark agent performance."""
    print(f"Running performance benchmark with {num_iterations} iterations...")
    
    # Generate test data
    observations = [np.random.rand(8) for _ in range(num_iterations)]
    
    start_time = time.time()
    
    # Sequential processing
    sequential_results = []
    sequential_start = time.time()
    for obs in observations[:100]:  # Smaller sample for comparison
        beliefs = agent.update_beliefs(obs)
        action = agent.select_action(beliefs)
        sequential_results.append(action)
    sequential_time = time.time() - sequential_start
    
    # Parallel processing
    parallel_start = time.time()
    parallel_beliefs = agent.parallel_inference(observations[:100])
    parallel_actions = [agent.select_action(beliefs) for beliefs in parallel_beliefs]
    parallel_time = time.time() - parallel_start
    
    total_time = time.time() - start_time
    
    # Get performance statistics
    stats = agent.get_performance_stats()
    
    return {
        'total_time': total_time,
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': sequential_time / parallel_time if parallel_time > 0 else 1.0,
        'performance_stats': stats,
        'iterations_completed': len(sequential_results)
    }