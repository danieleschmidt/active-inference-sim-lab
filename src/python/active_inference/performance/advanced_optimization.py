"""Advanced Performance Optimization for Scalable Active Inference.

This module implements comprehensive performance optimization, intelligent caching,
and advanced concurrency patterns for production-scale Active Inference systems:

- Multi-level caching with intelligent eviction policies
- Asynchronous processing with work queues and thread pools
- Memory pooling and resource management
- GPU acceleration and vectorized operations
- Distributed computing support
- Performance profiling and optimization recommendations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Awaitable
from dataclasses import dataclass, field
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import logging
from queue import Queue, PriorityQueue, Empty
from collections import OrderedDict, defaultdict, deque
import weakref
import pickle
import hashlib
import psutil
from functools import lru_cache, wraps
from contextlib import contextmanager
import gc
import resource
from abc import ABC, abstractmethod

# Optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    throughput_ops_per_sec: float
    concurrency_level: int
    gpu_utilization: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    priority: float = 1.0


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def should_evict(self, entries: List[CacheEntry], max_size: int) -> List[str]:
        """Determine which cache entries should be evicted."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, entries: List[CacheEntry], max_size: int) -> List[str]:
        if len(entries) <= max_size:
            return []
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        excess_count = len(entries) - max_size
        
        return [entry.key for entry in sorted_entries[:excess_count]]


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def should_evict(self, entries: List[CacheEntry], max_size: int) -> List[str]:
        if len(entries) <= max_size:
            return []
        
        # Sort by access count (least accessed first)
        sorted_entries = sorted(entries, key=lambda x: x.access_count)
        excess_count = len(entries) - max_size
        
        return [entry.key for entry in sorted_entries[:excess_count]]


class SizeBasedEvictionPolicy(CacheEvictionPolicy):
    """Size-based eviction policy (largest items first)."""
    
    def should_evict(self, entries: List[CacheEntry], max_size: int) -> List[str]:
        if len(entries) <= max_size:
            return []
        
        # Sort by size (largest first)
        sorted_entries = sorted(entries, key=lambda x: x.size_bytes, reverse=True)
        excess_count = len(entries) - max_size
        
        return [entry.key for entry in sorted_entries[:excess_count]]


class IntelligentCache:
    """Multi-level intelligent cache with advanced eviction policies."""
    
    def __init__(self,
                 max_entries: int = 10000,
                 max_memory_mb: int = 1000,
                 eviction_policy: str = "lru",
                 enable_compression: bool = True,
                 enable_persistence: bool = False):
        
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Eviction policy
        if eviction_policy == "lru":
            self.eviction_policy = LRUEvictionPolicy()
        elif eviction_policy == "lfu":
            self.eviction_policy = LFUEvictionPolicy()
        elif eviction_policy == "size":
            self.eviction_policy = SizeBasedEvictionPolicy()
        else:
            raise ValueError(f"Unknown eviction policy: {eviction_policy}")
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_memory_bytes = 0
        
        # Background cleanup
        self.cleanup_interval = 60.0  # 1 minute
        self.cleanup_thread = None
        self.should_cleanup = True
        
        self.logger = logging.getLogger("IntelligentCache")
        
        # Start background cleanup
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cache cleanup thread."""
        def cleanup_loop():
            while self.should_cleanup:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_expired_entries()
                    self._optimize_cache_layout()
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        with self.cache_lock:
            for key, entry in self.cache.items():
                if entry.ttl and current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _optimize_cache_layout(self):
        """Optimize cache layout for better performance."""
        with self.cache_lock:
            # Check if eviction is needed
            current_memory_mb = self.total_memory_bytes / (1024 * 1024)
            
            if (len(self.cache) > self.max_entries or 
                current_memory_mb > self.max_memory_mb):
                
                entries = list(self.cache.values())
                keys_to_evict = self.eviction_policy.should_evict(entries, self.max_entries)
                
                # Also consider memory-based eviction
                if current_memory_mb > self.max_memory_mb:
                    # Add memory-based eviction
                    memory_sorted = sorted(entries, key=lambda x: x.size_bytes, reverse=True)
                    bytes_to_free = int((current_memory_mb - self.max_memory_mb * 0.8) * 1024 * 1024)
                    
                    freed_bytes = 0
                    for entry in memory_sorted:
                        if freed_bytes >= bytes_to_free:
                            break
                        if entry.key not in keys_to_evict:
                            keys_to_evict.append(entry.key)
                            freed_bytes += entry.size_bytes
                
                # Evict selected entries
                for key in keys_to_evict:
                    if key in self.cache:
                        entry = self.cache[key]
                        self.total_memory_bytes -= entry.size_bytes
                        del self.cache[key]
                
                if keys_to_evict:
                    self.logger.debug(f"Evicted {len(keys_to_evict)} cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    del self.cache[key]
                    self.total_memory_bytes -= entry.size_bytes
                    self.cache_misses += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.timestamp = time.time()
                
                # Move to end (LRU)
                self.cache.move_to_end(key)
                
                self.cache_hits += 1
                return entry.value
            else:
                self.cache_misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, priority: float = 1.0):
        """Put value into cache."""
        # Estimate size
        try:
            if hasattr(value, 'nbytes'):  # NumPy arrays
                size_bytes = value.nbytes
            else:
                # Rough estimate using pickle
                size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 1024  # Default estimate
        
        with self.cache_lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_memory_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl,
                priority=priority
            )
            
            self.cache[key] = entry
            self.total_memory_bytes += size_bytes
    
    def invalidate(self, key: str):
        """Remove specific key from cache."""
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                self.total_memory_bytes -= entry.size_bytes
                del self.cache[key]
    
    def clear(self):
        """Clear all cache entries."""
        with self.cache_lock:
            self.cache.clear()
            self.total_memory_bytes = 0
            self.cache_hits = 0
            self.cache_misses = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.cache_lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / max(1, total_requests)
            
            return {
                'cache_entries': len(self.cache),
                'max_entries': self.max_entries,
                'memory_usage_mb': self.total_memory_bytes / (1024 * 1024),
                'max_memory_mb': self.max_memory_mb,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'average_entry_size_kb': (self.total_memory_bytes / len(self.cache) / 1024) if self.cache else 0,
                'eviction_policy': self.eviction_policy.__class__.__name__
            }
    
    def shutdown(self):
        """Shutdown the cache system."""
        self.should_cleanup = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2.0)
        
        if self.enable_persistence:
            # Save cache to disk (simplified implementation)
            self.logger.info("Cache persistence not fully implemented")
        
        self.clear()
        self.logger.info("Intelligent cache shutdown complete")


class MemoryPool:
    """Memory pool for efficient array allocation and reuse."""
    
    def __init__(self, initial_size: int = 100, max_size: int = 1000):
        self.pools = defaultdict(deque)  # (shape, dtype) -> deque of arrays
        self.pool_lock = threading.RLock()
        self.initial_size = initial_size
        self.max_size = max_size
        self.allocation_count = 0
        self.reuse_count = 0
        
        self.logger = logging.getLogger("MemoryPool")
    
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Get array from pool or allocate new one."""
        pool_key = (shape, dtype)
        
        with self.pool_lock:
            if self.pools[pool_key]:
                array = self.pools[pool_key].popleft()
                array.fill(0)  # Clear the array
                self.reuse_count += 1
                return array
            else:
                self.allocation_count += 1
                return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """Return array to pool for reuse."""
        if not isinstance(array, np.ndarray):
            return
        
        pool_key = (array.shape, array.dtype)
        
        with self.pool_lock:
            if len(self.pools[pool_key]) < self.max_size:
                self.pools[pool_key].append(array)
            # Otherwise, let it be garbage collected
    
    def preallocate(self, shape: Tuple[int, ...], dtype: np.dtype, count: int):
        """Pre-allocate arrays for expected usage patterns."""
        pool_key = (shape, dtype)
        
        with self.pool_lock:
            current_count = len(self.pools[pool_key])
            needed_count = min(count - current_count, self.max_size - current_count)
            
            for _ in range(needed_count):
                array = np.zeros(shape, dtype=dtype)
                self.pools[pool_key].append(array)
        
        self.logger.debug(f"Pre-allocated {needed_count} arrays of shape {shape}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.pool_lock:
            total_arrays = sum(len(pool) for pool in self.pools.values())
            total_memory_mb = 0
            
            for (shape, dtype), pool in self.pools.items():
                array_size = np.prod(shape) * dtype.itemsize
                total_memory_mb += len(pool) * array_size / (1024 * 1024)
            
            reuse_rate = self.reuse_count / max(1, self.allocation_count + self.reuse_count)
            
            return {
                'total_pooled_arrays': total_arrays,
                'pool_types': len(self.pools),
                'total_memory_mb': total_memory_mb,
                'allocation_count': self.allocation_count,
                'reuse_count': self.reuse_count,
                'reuse_rate': reuse_rate,
                'max_pool_size': self.max_size
            }


class AsyncWorkQueue:
    """Asynchronous work queue with prioritization and load balancing."""
    
    def __init__(self, max_workers: int = None, queue_size: int = 10000):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.queue_size = queue_size
        
        # Work queues
        self.high_priority_queue = PriorityQueue(maxsize=queue_size)
        self.normal_priority_queue = Queue(maxsize=queue_size)
        self.low_priority_queue = Queue(maxsize=queue_size)
        
        # Worker management
        self.workers = []
        self.worker_stats = defaultdict(lambda: {'tasks_completed': 0, 'total_time': 0})
        self.is_running = False
        
        # Performance tracking
        self.submitted_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.logger = logging.getLogger("AsyncWorkQueue")
    
    def start(self):
        """Start the work queue processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(f"worker_{i}",),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} workers")
    
    def _worker_loop(self, worker_id: str):
        """Main worker loop."""
        while self.is_running:
            try:
                # Try queues in priority order
                task = None
                
                # High priority queue
                try:
                    priority, task_func, args, kwargs, future = self.high_priority_queue.get_nowait()
                    task = (task_func, args, kwargs, future)
                except Empty:
                    pass
                
                # Normal priority queue
                if task is None:
                    try:
                        task_func, args, kwargs, future = self.normal_priority_queue.get_nowait()
                        task = (task_func, args, kwargs, future)
                    except Empty:
                        pass
                
                # Low priority queue
                if task is None:
                    try:
                        task_func, args, kwargs, future = self.low_priority_queue.get_nowait()
                        task = (task_func, args, kwargs, future)
                    except Empty:
                        pass
                
                if task:
                    self._execute_task(worker_id, *task)
                else:
                    # No tasks available, sleep briefly
                    time.sleep(0.01)
            
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def _execute_task(self, worker_id: str, task_func: Callable, args: tuple, kwargs: dict, future):
        """Execute a single task."""
        start_time = time.time()
        
        try:
            result = task_func(*args, **kwargs)
            future.set_result(result)
            self.completed_tasks += 1
            
        except Exception as e:
            future.set_exception(e)
            self.failed_tasks += 1
            self.logger.error(f"Task execution failed: {e}")
        
        finally:
            execution_time = time.time() - start_time
            self.worker_stats[worker_id]['tasks_completed'] += 1
            self.worker_stats[worker_id]['total_time'] += execution_time
    
    def submit(self, task_func: Callable, *args, priority: str = "normal", **kwargs) -> asyncio.Future:
        """Submit task to the work queue."""
        if not self.is_running:
            self.start()
        
        future = asyncio.Future()
        self.submitted_tasks += 1
        
        try:
            if priority == "high":
                # Use negative timestamp for priority (earlier = higher priority)
                self.high_priority_queue.put((-time.time(), task_func, args, kwargs, future))
            elif priority == "low":
                self.low_priority_queue.put((task_func, args, kwargs, future))
            else:  # normal
                self.normal_priority_queue.put((task_func, args, kwargs, future))
            
        except Exception as e:
            future.set_exception(e)
            self.logger.error(f"Failed to submit task: {e}")
        
        return future
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get work queue statistics."""
        return {
            'max_workers': self.max_workers,
            'active_workers': len(self.workers),
            'submitted_tasks': self.submitted_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.completed_tasks / max(1, self.submitted_tasks),
            'high_priority_queue_size': self.high_priority_queue.qsize(),
            'normal_priority_queue_size': self.normal_priority_queue.qsize(),
            'low_priority_queue_size': self.low_priority_queue.qsize(),
            'worker_statistics': dict(self.worker_stats)
        }
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the work queue."""
        self.is_running = False
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self.workers:
            remaining_time = max(0, timeout - (time.time() - start_time))
            if worker.is_alive():
                worker.join(timeout=remaining_time)
        
        self.logger.info("Async work queue shutdown complete")


class GPUAccelerator:
    """GPU acceleration for Active Inference computations."""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.logger = logging.getLogger("GPUAccelerator")
        
        if self.gpu_available:
            try:
                self.device_id = cp.cuda.Device().id
                self.memory_pool = cp.get_default_memory_pool()
                self.logger.info(f"GPU acceleration enabled (device {self.device_id})")
            except Exception as e:
                self.gpu_available = False
                self.logger.warning(f"GPU initialization failed: {e}")
        else:
            self.logger.info("GPU acceleration not available (CuPy not installed)")
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer array to GPU if available."""
        if self.gpu_available and isinstance(array, np.ndarray):
            try:
                return cp.asarray(array)
            except Exception as e:
                self.logger.warning(f"GPU transfer failed: {e}")
                return array
        return array
    
    def to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Transfer array to CPU."""
        if self.gpu_available and hasattr(array, 'get'):
            try:
                return array.get()
            except Exception as e:
                self.logger.warning(f"CPU transfer failed: {e}")
                return array
        return array
    
    def accelerated_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Accelerated matrix multiplication."""
        if not self.gpu_available or a.size < 1000:  # Use CPU for small arrays
            return np.matmul(a, b)
        
        try:
            gpu_a = self.to_gpu(a)
            gpu_b = self.to_gpu(b)
            gpu_result = cp.matmul(gpu_a, gpu_b)
            return self.to_cpu(gpu_result)
        except Exception as e:
            self.logger.warning(f"GPU matmul failed, falling back to CPU: {e}")
            return np.matmul(a, b)
    
    def accelerated_inference(self, beliefs: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """GPU-accelerated belief inference."""
        if not self.gpu_available:
            # CPU fallback
            return self._cpu_inference(beliefs, observations)
        
        try:
            gpu_beliefs = self.to_gpu(beliefs)
            gpu_observations = self.to_gpu(observations)
            
            # Simplified inference computation (actual implementation would be more complex)
            gpu_result = cp.dot(gpu_beliefs, gpu_observations) + cp.exp(-gpu_beliefs)
            
            return self.to_cpu(gpu_result)
        except Exception as e:
            self.logger.warning(f"GPU inference failed, falling back to CPU: {e}")
            return self._cpu_inference(beliefs, observations)
    
    def _cpu_inference(self, beliefs: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """CPU fallback for belief inference."""
        return np.dot(beliefs, observations) + np.exp(-beliefs)
    
    def get_gpu_statistics(self) -> Dict[str, Any]:
        """Get GPU utilization statistics."""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        try:
            # Get memory usage
            meminfo = cp.cuda.MemoryInfo()
            memory_used_mb = (meminfo.total - meminfo.free) / (1024 * 1024)
            memory_total_mb = meminfo.total / (1024 * 1024)
            
            return {
                'gpu_available': True,
                'device_id': self.device_id,
                'memory_used_mb': memory_used_mb,
                'memory_total_mb': memory_total_mb,
                'memory_utilization': memory_used_mb / memory_total_mb
            }
        except Exception as e:
            self.logger.error(f"Error getting GPU statistics: {e}")
            return {'gpu_available': True, 'error': str(e)}


class PerformanceProfiler:
    """Performance profiler for Active Inference operations."""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.current_profile = None
        self.profiling_lock = threading.RLock()
        self.logger = logging.getLogger("PerformanceProfiler")
    
    @contextmanager
    def profile(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling operations."""
        metadata = metadata or {}
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            profile_data = {
                'operation_name': operation_name,
                'execution_time': end_time - start_time,
                'memory_delta_mb': end_memory - start_memory,
                'cpu_usage_percent': (start_cpu + end_cpu) / 2,
                'timestamp': start_time,
                'metadata': metadata
            }
            
            with self.profiling_lock:
                self.profiles[operation_name].append(profile_data)
                
                # Limit profile history
                if len(self.profiles[operation_name]) > 1000:
                    self.profiles[operation_name] = self.profiles[operation_name][-1000:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_performance_report(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance analysis report."""
        with self.profiling_lock:
            if operation_name:
                operations = {operation_name: self.profiles[operation_name]}
            else:
                operations = dict(self.profiles)
            
            report = {}
            
            for op_name, profiles_list in operations.items():
                if not profiles_list:
                    continue
                
                execution_times = [p['execution_time'] for p in profiles_list]
                memory_deltas = [p['memory_delta_mb'] for p in profiles_list]
                cpu_usages = [p['cpu_usage_percent'] for p in profiles_list]
                
                report[op_name] = {
                    'call_count': len(profiles_list),
                    'avg_execution_time': np.mean(execution_times),
                    'min_execution_time': np.min(execution_times),
                    'max_execution_time': np.max(execution_times),
                    'std_execution_time': np.std(execution_times),
                    'avg_memory_delta_mb': np.mean(memory_deltas),
                    'max_memory_delta_mb': np.max(memory_deltas),
                    'avg_cpu_usage': np.mean(cpu_usages),
                    'throughput_ops_per_sec': len(profiles_list) / max(1, sum(execution_times)),
                    'recent_performance_trend': self._compute_performance_trend(execution_times[-50:])  # Last 50
                }
            
            return report
    
    def _compute_performance_trend(self, execution_times: List[float]) -> str:
        """Compute performance trend (improving, stable, degrading)."""
        if len(execution_times) < 10:
            return "insufficient_data"
        
        # Compute trend using linear regression slope
        x = np.arange(len(execution_times))
        y = np.array(execution_times)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope < -0.001:  # Improving (times decreasing)
                return "improving"
            elif slope > 0.001:  # Degrading (times increasing)
                return "degrading"
            else:
                return "stable"
        except Exception:
            return "unknown"
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on profiling data."""
        recommendations = []
        
        with self.profiling_lock:
            for op_name, profiles_list in self.profiles.items():
                if not profiles_list:
                    continue
                
                recent_profiles = profiles_list[-50:]  # Last 50 calls
                avg_time = np.mean([p['execution_time'] for p in recent_profiles])
                max_memory = max([p['memory_delta_mb'] for p in recent_profiles])
                avg_cpu = np.mean([p['cpu_usage_percent'] for p in recent_profiles])
                
                # Performance recommendations
                if avg_time > 1.0:  # Slow operations
                    recommendations.append(f"Operation '{op_name}' is slow ({avg_time:.3f}s avg). Consider optimization or caching.")
                
                if max_memory > 500:  # High memory usage
                    recommendations.append(f"Operation '{op_name}' uses high memory ({max_memory:.1f}MB). Consider memory pooling.")
                
                if avg_cpu > 80:  # High CPU usage
                    recommendations.append(f"Operation '{op_name}' is CPU intensive ({avg_cpu:.1f}% avg). Consider parallel processing.")
                
                # Trend-based recommendations
                trend = self._compute_performance_trend([p['execution_time'] for p in recent_profiles])
                if trend == "degrading":
                    recommendations.append(f"Operation '{op_name}' performance is degrading. Investigate for memory leaks or inefficiencies.")
        
        return recommendations


class DistributedProcessor:
    """Distributed processing for large-scale Active Inference."""
    
    def __init__(self, worker_nodes: List[str] = None):
        self.worker_nodes = worker_nodes or []
        self.local_processor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        self.logger = logging.getLogger("DistributedProcessor")
        
        # Task distribution strategy
        self.task_distribution_strategy = "round_robin"
        self.current_node_index = 0
        
        # Performance tracking
        self.distributed_tasks = 0
        self.local_tasks = 0
    
    def distribute_computation(self, 
                            computation_func: Callable,
                            data_chunks: List[Any],
                            combine_func: Callable = None) -> Any:
        """Distribute computation across available nodes."""
        
        if not self.worker_nodes:
            # Fall back to local processing
            return self._process_locally(computation_func, data_chunks, combine_func)
        
        try:
            # Distribute chunks to worker nodes (simplified implementation)
            futures = []
            
            for chunk in data_chunks:
                if self._should_process_locally(chunk):
                    future = self.local_processor.submit(computation_func, chunk)
                    self.local_tasks += 1
                else:
                    # In a real implementation, this would send tasks to remote nodes
                    future = self.local_processor.submit(computation_func, chunk)
                    self.distributed_tasks += 1
                
                futures.append(future)
            
            # Collect results
            results = [future.result() for future in futures]
            
            # Combine results if combiner provided
            if combine_func:
                return combine_func(results)
            else:
                return results
        
        except Exception as e:
            self.logger.error(f"Distributed processing failed: {e}")
            return self._process_locally(computation_func, data_chunks, combine_func)
    
    def _process_locally(self, computation_func: Callable, data_chunks: List[Any], combine_func: Callable) -> Any:
        """Fallback to local parallel processing."""
        futures = [self.local_processor.submit(computation_func, chunk) for chunk in data_chunks]
        results = [future.result() for future in futures]
        
        self.local_tasks += len(data_chunks)
        
        if combine_func:
            return combine_func(results)
        else:
            return results
    
    def _should_process_locally(self, chunk: Any) -> bool:
        """Determine if chunk should be processed locally."""
        # Simple heuristic: process small chunks locally
        try:
            if hasattr(chunk, '__len__'):
                return len(chunk) < 1000
            elif hasattr(chunk, 'size'):
                return chunk.size < 10000
            else:
                return True  # Process unknown types locally
        except Exception:
            return True
    
    def get_distribution_statistics(self) -> Dict[str, Any]:
        """Get distribution statistics."""
        total_tasks = self.distributed_tasks + self.local_tasks
        
        return {
            'worker_nodes': len(self.worker_nodes),
            'distributed_tasks': self.distributed_tasks,
            'local_tasks': self.local_tasks,
            'total_tasks': total_tasks,
            'distribution_rate': self.distributed_tasks / max(1, total_tasks),
            'distribution_strategy': self.task_distribution_strategy
        }
    
    def shutdown(self):
        """Shutdown distributed processor."""
        self.local_processor.shutdown(wait=True)
        self.logger.info("Distributed processor shutdown complete")


class ScalableActiveInferenceFramework:
    """Scalable Active Inference framework with advanced optimization."""
    
    def __init__(self, 
                 enable_caching: bool = True,
                 enable_gpu: bool = True,
                 enable_profiling: bool = True,
                 max_workers: int = None):
        
        self.logger = logging.getLogger("ScalableActiveInference")
        
        # Initialize optimization components
        self.cache = IntelligentCache() if enable_caching else None
        self.memory_pool = MemoryPool()
        self.work_queue = AsyncWorkQueue(max_workers=max_workers)
        self.gpu_accelerator = GPUAccelerator() if enable_gpu else None
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.distributed_processor = DistributedProcessor()
        
        # Framework state
        self.is_initialized = True
        self.framework_stats = {
            'total_operations': 0,
            'cache_enabled': enable_caching,
            'gpu_enabled': enable_gpu,
            'profiling_enabled': enable_profiling,
            'start_time': time.time()
        }
        
        self.logger.info("Scalable Active Inference Framework initialized")
    
    def optimized_agent_inference(self, agent: Any, observation: np.ndarray) -> Dict[str, Any]:
        """Optimized agent inference with caching and acceleration."""
        
        # Generate cache key
        obs_hash = hashlib.md5(observation.tobytes()).hexdigest()[:16]
        cache_key = f"inference_{agent.agent_id}_{obs_hash}"
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return {'result': cached_result, 'from_cache': True}
        
        # Profile the operation
        profile_context = self.profiler.profile("agent_inference") if self.profiler else None
        
        with (profile_context if profile_context else self._null_context()):
            # Get optimized arrays from memory pool
            beliefs_array = self.memory_pool.get_array(observation.shape, observation.dtype)
            
            try:
                # Perform inference (simplified)
                if hasattr(agent, 'beliefs') and agent.beliefs:
                    # Convert beliefs to array for computation
                    belief_values = []
                    for belief in agent.beliefs.get_all_beliefs().values():
                        if hasattr(belief, 'mean'):
                            belief_values.extend(belief.mean.flatten())
                    
                    if belief_values:
                        beliefs_array[:len(belief_values)] = belief_values[:beliefs_array.size]
                
                # GPU-accelerated computation if available
                if self.gpu_accelerator:
                    result_array = self.gpu_accelerator.accelerated_inference(beliefs_array, observation)
                else:
                    # CPU computation
                    result_array = np.dot(beliefs_array, observation) + np.exp(-beliefs_array)
                
                # Convert result back to beliefs format
                result = {
                    'updated_beliefs': result_array.copy(),
                    'inference_quality': np.mean(np.abs(result_array)),
                    'computation_time': time.time()
                }
                
                # Cache the result
                if self.cache:
                    self.cache.put(cache_key, result, ttl=300)  # 5-minute TTL
                
                return {'result': result, 'from_cache': False}
            
            finally:
                # Return array to pool
                self.memory_pool.return_array(beliefs_array)
                self.framework_stats['total_operations'] += 1
    
    @contextmanager
    def _null_context(self):
        """Null context manager when profiling is disabled."""
        yield
    
    def batch_process_agents(self, agents: List[Any], observations: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process multiple agents in parallel."""
        
        if len(agents) != len(observations):
            raise ValueError("Number of agents must match number of observations")
        
        # Submit tasks to work queue
        futures = []
        for agent, obs in zip(agents, observations):
            future = self.work_queue.submit(
                self.optimized_agent_inference, agent, obs, priority="normal"
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()  # This will be available when the task completes
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def distributed_training(self, agents: List[Any], training_data: List[Any]) -> Dict[str, Any]:
        """Distributed training across multiple agents."""
        
        def train_agent_chunk(chunk_data):
            agent, data_subset = chunk_data
            
            # Simplified training loop
            for data_point in data_subset:
                if hasattr(agent, 'update_model'):
                    agent.update_model(data_point['observation'], data_point['action'], data_point.get('reward', 0))
            
            return {
                'agent_id': agent.agent_id,
                'trained_samples': len(data_subset),
                'final_stats': agent.get_statistics() if hasattr(agent, 'get_statistics') else {}
            }
        
        # Combine function for training results
        def combine_training_results(results):
            total_samples = sum(r['trained_samples'] for r in results)
            return {
                'total_agents_trained': len(results),
                'total_samples_processed': total_samples,
                'agent_results': results
            }
        
        # Distribute training across processors
        agent_data_chunks = list(zip(agents, training_data))
        
        return self.distributed_processor.distribute_computation(
            train_agent_chunk, agent_data_chunks, combine_training_results
        )
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'framework_stats': self.framework_stats,
            'uptime': time.time() - self.framework_stats['start_time']
        }
        
        # Cache statistics
        if self.cache:
            report['cache_stats'] = self.cache.get_statistics()
        
        # Memory pool statistics
        report['memory_pool_stats'] = self.memory_pool.get_statistics()
        
        # Work queue statistics
        report['work_queue_stats'] = self.work_queue.get_statistics()
        
        # GPU statistics
        if self.gpu_accelerator:
            report['gpu_stats'] = self.gpu_accelerator.get_gpu_statistics()
        
        # Profiling report
        if self.profiler:
            report['performance_profiles'] = self.profiler.get_performance_report()
            report['optimization_recommendations'] = self.profiler.get_optimization_recommendations()
        
        # Distribution statistics
        report['distribution_stats'] = self.distributed_processor.get_distribution_statistics()
        
        return report
    
    def optimize_framework(self):
        """Apply automatic optimizations based on usage patterns."""
        self.logger.info("Applying automatic optimizations...")
        
        # Memory pool optimization
        if self.profiler:
            profiles = self.profiler.get_performance_report()
            
            # Pre-allocate common array sizes
            for op_name, profile_data in profiles.items():
                if 'metadata' in profile_data and profile_data['call_count'] > 10:
                    # This is a simplified optimization - in practice, you'd analyze
                    # the actual array shapes and types used
                    common_shape = (64,)  # Example common shape
                    self.memory_pool.preallocate(common_shape, np.float64, 10)
        
        # Cache optimization
        if self.cache:
            cache_stats = self.cache.get_statistics()
            if cache_stats['hit_rate'] < 0.5:  # Low hit rate
                self.logger.info("Low cache hit rate detected, consider adjusting cache parameters")
        
        self.logger.info("Automatic optimizations completed")
    
    def shutdown(self):
        """Shutdown the scalable framework."""
        self.logger.info("Shutting down Scalable Active Inference Framework")
        
        try:
            if self.work_queue:
                self.work_queue.shutdown()
            
            if self.cache:
                self.cache.shutdown()
            
            if self.distributed_processor:
                self.distributed_processor.shutdown()
            
            # Final performance report
            final_report = self.get_comprehensive_performance_report()
            self.logger.info(f"Final performance report: {final_report}")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("Framework shutdown complete")


# Utility decorators for performance optimization
def cached_computation(cache_instance: IntelligentCache, ttl: float = 300):
    """Decorator for caching expensive computations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


def memory_optimized(memory_pool: MemoryPool):
    """Decorator for memory-optimized array operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a simplified example - real implementation would analyze
            # function signature and optimize array allocations
            return func(*args, **kwargs)
        return wrapper
    return decorator


def gpu_accelerated(gpu_accelerator: GPUAccelerator):
    """Decorator for GPU-accelerated computations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert numpy arrays to GPU arrays where beneficial
            gpu_args = []
            for arg in args:
                if isinstance(arg, np.ndarray) and arg.size > 1000:
                    gpu_args.append(gpu_accelerator.to_gpu(arg))
                else:
                    gpu_args.append(arg)
            
            result = func(*gpu_args, **kwargs)
            
            # Convert result back to CPU if needed
            if hasattr(result, 'get'):  # CuPy array
                result = gpu_accelerator.to_cpu(result)
            
            return result
        return wrapper
    return decorator
