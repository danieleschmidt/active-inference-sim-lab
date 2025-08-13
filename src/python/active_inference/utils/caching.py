"""
Performance optimization and caching utilities for active inference.

This module implements caching mechanisms, computational optimization,
and memory-efficient algorithms for high-performance active inference.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, Union
import functools
import time
import pickle
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

from .logging import get_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    hit_count: int = 0
    computation_time: float = 0.0
    
    def __post_init__(self):
        self.hit_count = 0
        self.last_accessed = self.timestamp


class LRUCache:
    """
    Least Recently Used cache with performance monitoring.
    """
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of entries
            ttl: Time to live in seconds (None for no expiration)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: list = []
        
        # Performance stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        self.logger = get_logger("lru_cache")
    
    def _make_key(self, args: tuple, kwargs: dict) -> str:
        """Create cache key from arguments."""
        # Create a hash of the arguments
        key_data = (args, tuple(sorted(kwargs.items())))
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if self.ttl and time.time() - entry.timestamp > self.ttl:
            self._evict(key)
            self.misses += 1
            return None
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Update stats
        entry.hit_count += 1
        entry.last_accessed = time.time()
        self.hits += 1
        
        return entry.value
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Put value into cache."""
        # Remove if already exists
        if key in self.cache:
            self.access_order.remove(key)
        
        # Create new entry
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            computation_time=computation_time
        )
        
        # Add to cache
        self.cache[key] = entry
        self.access_order.append(key)
        
        # Evict if necessary
        while len(self.cache) > self.maxsize:
            oldest_key = self.access_order.pop(0)
            self._evict(oldest_key)
    
    def _evict(self, key: str) -> None:
        """Evict entry from cache."""
        if key in self.cache:
            del self.cache[key]
            self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.maxsize,
            'average_computation_time': np.mean([
                entry.computation_time for entry in self.cache.values()
                if entry.computation_time > 0
            ]) if self.cache else 0
        }


def memoize(maxsize: int = 128, ttl: Optional[float] = None, 
           typed: bool = False, ignore_self: bool = True):
    """
    Memoization decorator with LRU cache.
    
    Args:
        maxsize: Maximum cache size
        ttl: Time to live for cached entries
        typed: Whether to consider argument types in cache key
        ignore_self: Whether to ignore 'self' parameter in methods
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize=maxsize, ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Handle 'self' parameter for methods
            cache_args = args
            if ignore_self and len(args) > 0 and hasattr(args[0], func.__name__):
                cache_args = args[1:]  # Skip 'self'
            
            # Create cache key
            if typed:
                key_args = cache_args + tuple(type(arg) for arg in cache_args)
                key_kwargs = kwargs.copy()
                key_kwargs.update({f"{k}_type": type(v) for k, v in kwargs.items()})
            else:
                key_args = cache_args
                key_kwargs = kwargs
            
            key = cache._make_key(key_args, key_kwargs)
            
            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            computation_time = time.perf_counter() - start_time
            
            # Cache result
            cache.put(key, result, computation_time)
            
            return result
        
        # Attach cache for inspection
        wrapper._cache = cache
        
        return wrapper
    return decorator


class BatchProcessor:
    """
    Batch processing for vectorized computations.
    """
    
    def __init__(self, batch_size: int = 32, n_workers: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size
            n_workers: Number of worker threads (None for auto)
        """
        self.batch_size = batch_size
        self.n_workers = n_workers or min(4, os.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
        self.logger = get_logger("batch_processor")
    
    def process_batch(self, 
                     data: np.ndarray,
                     process_func: Callable,
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
        Process data in batches.
        
        Args:
            data: Input data array
            process_func: Function to apply to each batch
            batch_size: Batch size (uses default if None)
            
        Returns:
            Processed data
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        n_samples = len(data)
        results = []
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]
            
            batch_result = process_func(batch)
            results.append(batch_result)
        
        # Concatenate results
        if results:
            return np.concatenate(results, axis=0)
        else:
            return np.array([])
    
    def parallel_process(self,
                        data_batches: list,
                        process_func: Callable) -> list:
        """
        Process multiple batches in parallel.
        
        Args:
            data_batches: List of data batches
            process_func: Function to apply to each batch
            
        Returns:
            List of processed results
        """
        futures = []
        
        for batch in data_batches:
            future = self.executor.submit(process_func, batch)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                results.append(None)
        
        return results
    
    def close(self):
        """Close the thread pool executor."""
        self.executor.shutdown(wait=True)


class PrecomputedTables:
    """
    Precomputed lookup tables for common computations.
    """
    
    def __init__(self):
        """Initialize precomputed tables."""
        self.tables = {}
        self.logger = get_logger("precomputed_tables")
    
    def create_gaussian_table(self, 
                             x_range: Tuple[float, float],
                             n_points: int = 1000,
                             sigma: float = 1.0) -> None:
        """
        Create precomputed Gaussian probability table.
        
        Args:
            x_range: Range of x values (min, max)
            n_points: Number of points to precompute
            sigma: Standard deviation
        """
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, n_points)
        
        # Compute Gaussian probabilities
        gaussian_values = np.exp(-0.5 * (x_values / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        
        self.tables['gaussian'] = {
            'x_values': x_values,
            'probabilities': gaussian_values,
            'x_min': x_min,
            'x_max': x_max,
            'sigma': sigma
        }
        
        self.logger.info(f"Created Gaussian table with {n_points} points")
    
    def lookup_gaussian(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Fast Gaussian probability lookup.
        
        Args:
            x: Input values
            
        Returns:
            Gaussian probabilities
        """
        if 'gaussian' not in self.tables:
            raise ValueError("Gaussian table not created. Call create_gaussian_table first.")
        
        table = self.tables['gaussian']
        x_values = table['x_values']
        probabilities = table['probabilities']
        
        # Interpolate
        return np.interp(x, x_values, probabilities)
    
    def create_softmax_table(self,
                           x_range: Tuple[float, float],
                           n_points: int = 1000,
                           temperature: float = 1.0) -> None:
        """
        Create precomputed softmax table.
        
        Args:
            x_range: Range of input values
            n_points: Number of points
            temperature: Softmax temperature
        """
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, n_points)
        
        # Compute softmax for different input sizes
        # Store for different common sizes
        self.tables['softmax'] = {
            'x_values': x_values,
            'temperature': temperature,
            'precomputed': {}
        }
        
        for size in [2, 3, 4, 5, 10]:
            softmax_values = []
            for x in x_values:
                inputs = np.full(size, x)
                inputs[0] = x  # Vary first element
                softmax = np.exp(inputs / temperature)
                softmax = softmax / softmax.sum()
                softmax_values.append(softmax)
            
            self.tables['softmax']['precomputed'][size] = np.array(softmax_values)
        
        self.logger.info(f"Created softmax table with {n_points} points")
    
    def save_tables(self, filepath: str) -> None:
        """Save precomputed tables to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.tables, f)
        self.logger.info(f"Tables saved to {filepath}")
    
    def load_tables(self, filepath: str) -> None:
        """Load precomputed tables from file with security validation."""
        import os
        from pathlib import Path
        
        # Validate file path
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {filepath}")
        
        # Check file size (prevent loading extremely large files)
        file_size = os.path.getsize(filepath)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Cache file too large: {file_size} bytes")
        
        try:
            with open(filepath, 'rb') as f:
                # Use safe pickle loading with size limit
                data = f.read()
                if len(data) > 100 * 1024 * 1024:  # Additional size check
                    raise ValueError("Pickle data too large")
                
                import io
                data_stream = io.BytesIO(data)
                self.tables = pickle.load(data_stream)
                
            # Validate loaded data structure
            if not isinstance(self.tables, dict):
                raise ValueError("Invalid cache data: expected dictionary")
                
            self.logger.info(f"Tables safely loaded from {filepath}")
        except (pickle.UnpicklingError, EOFError, ImportError) as e:
            raise ValueError(f"Invalid or corrupted cache file: {e}")


class OptimizedOperations:
    """
    Optimized mathematical operations for active inference.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize optimized operations.
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.use_gpu = use_gpu
        
        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.logger = get_logger("optimized_ops.gpu")
                self.logger.info("GPU acceleration enabled")
            except ImportError:
                self.use_gpu = False
                self.logger = get_logger("optimized_ops.cpu")
                self.logger.warning("CuPy not available, falling back to CPU")
        else:
            self.logger = get_logger("optimized_ops.cpu")
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication."""
        if self.use_gpu:
            A_gpu = self.cp.asarray(A)
            B_gpu = self.cp.asarray(B)
            result_gpu = self.cp.dot(A_gpu, B_gpu)
            return self.cp.asnumpy(result_gpu)
        else:
            return np.dot(A, B)
    
    def eigendecomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized eigenvalue decomposition."""
        if self.use_gpu:
            matrix_gpu = self.cp.asarray(matrix)
            eigenvals_gpu, eigenvecs_gpu = self.cp.linalg.eig(matrix_gpu)
            return self.cp.asnumpy(eigenvals_gpu), self.cp.asnumpy(eigenvecs_gpu)
        else:
            return np.linalg.eig(matrix)
    
    def cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
        """Optimized Cholesky decomposition."""
        if self.use_gpu:
            matrix_gpu = self.cp.asarray(matrix)
            chol_gpu = self.cp.linalg.cholesky(matrix_gpu)
            return self.cp.asnumpy(chol_gpu)
        else:
            return np.linalg.cholesky(matrix)
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Optimized linear system solver."""
        if self.use_gpu:
            A_gpu = self.cp.asarray(A)
            b_gpu = self.cp.asarray(b)
            x_gpu = self.cp.linalg.solve(A_gpu, b_gpu)
            return self.cp.asnumpy(x_gpu)
        else:
            return np.linalg.solve(A, b)


# Global instances
_global_batch_processor = None
_global_precomputed_tables = None
_global_optimized_ops = None


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor()
    return _global_batch_processor


def get_precomputed_tables() -> PrecomputedTables:
    """Get global precomputed tables instance."""
    global _global_precomputed_tables
    if _global_precomputed_tables is None:
        _global_precomputed_tables = PrecomputedTables()
    return _global_precomputed_tables


def get_optimized_ops(use_gpu: bool = False) -> OptimizedOperations:
    """Get global optimized operations instance."""
    global _global_optimized_ops
    if _global_optimized_ops is None:
        _global_optimized_ops = OptimizedOperations(use_gpu=use_gpu)
    return _global_optimized_ops


# Cleanup function
def cleanup_resources():
    """Cleanup global resources."""
    global _global_batch_processor
    if _global_batch_processor:
        _global_batch_processor.close()
        _global_batch_processor = None