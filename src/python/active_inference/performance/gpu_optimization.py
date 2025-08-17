"""
GPU Optimization Module
Generation 3: MAKE IT SCALE (Optimized)
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU configuration parameters."""
    device_id: int = 0
    memory_limit: Optional[int] = None  # bytes
    enable_memory_growth: bool = True
    mixed_precision: bool = False
    enable_xla: bool = False


class GPUBackend(ABC):
    """Abstract GPU backend interface."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if GPU backend is available."""
        pass
    
    @abstractmethod
    def get_device_count(self) -> int:
        """Get number of available devices."""
        pass
    
    @abstractmethod
    def to_device(self, array: np.ndarray) -> Any:
        """Move array to GPU device."""
        pass
    
    @abstractmethod
    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        """Move array back to CPU."""
        pass
    
    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """GPU matrix multiplication."""
        pass
    
    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """GPU element-wise addition."""
        pass


class NumPyBackend(GPUBackend):
    """Fallback NumPy backend (CPU)."""
    
    def is_available(self) -> bool:
        return True
    
    def get_device_count(self) -> int:
        return 0  # No GPU devices
    
    def to_device(self, array: np.ndarray) -> np.ndarray:
        return array
    
    def to_cpu(self, gpu_array: np.ndarray) -> np.ndarray:
        return gpu_array
    
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)
    
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b


class CuPyBackend(GPUBackend):
    """CuPy GPU backend."""
    
    def __init__(self):
        self.cupy = None
        self._initialize()
    
    def _initialize(self):
        try:
            import cupy as cp
            self.cupy = cp
            logger.info("CuPy backend initialized successfully")
        except ImportError:
            logger.warning("CuPy not available, falling back to CPU")
    
    def is_available(self) -> bool:
        return self.cupy is not None
    
    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return self.cupy.cuda.runtime.getDeviceCount()
    
    def to_device(self, array: np.ndarray):
        if not self.is_available():
            return array
        return self.cupy.asarray(array)
    
    def to_cpu(self, gpu_array) -> np.ndarray:
        if not self.is_available():
            return gpu_array
        return self.cupy.asnumpy(gpu_array)
    
    def multiply(self, a, b):
        if not self.is_available():
            return np.dot(a, b)
        return self.cupy.dot(a, b)
    
    def add(self, a, b):
        if not self.is_available():
            return a + b
        return a + b


class GPUOptimizer:
    """
    GPU optimization manager for Active Inference computations.
    
    Provides automatic GPU acceleration with fallback to CPU,
    memory management, and performance monitoring.
    """
    
    def __init__(self, 
                 config: Optional[GPUConfig] = None,
                 auto_select_backend: bool = True):
        """
        Initialize GPU optimizer.
        
        Args:
            config: GPU configuration
            auto_select_backend: Automatically select best available backend
        """
        self.config = config or GPUConfig()
        self.backend = None
        self.performance_stats = {
            'gpu_operations': 0,
            'cpu_operations': 0,
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'memory_transfers': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # GPU memory cache
        self.gpu_cache = {}
        self.cache_size_limit = 1024 * 1024 * 1024  # 1GB default
        self.current_cache_size = 0
        
        if auto_select_backend:
            self._select_best_backend()
        else:
            self.backend = NumPyBackend()
        
        logger.info(f"GPU Optimizer initialized with {type(self.backend).__name__}")
    
    def _select_best_backend(self):
        """Automatically select the best available GPU backend."""
        # Try CuPy first
        cupy_backend = CuPyBackend()
        if cupy_backend.is_available():
            self.backend = cupy_backend
            logger.info("Selected CuPy backend for GPU acceleration")
            return
        
        # Fallback to NumPy
        self.backend = NumPyBackend()
        logger.info("Selected NumPy backend (CPU only)")
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.backend.is_available() and self.backend.get_device_count() > 0
    
    def optimize_matrix_operations(self, 
                                 matrices: List[np.ndarray],
                                 operation: str = "auto") -> List[np.ndarray]:
        """
        Optimize matrix operations using GPU acceleration.
        
        Args:
            matrices: List of input matrices
            operation: Type of operation ("multiply", "add", "auto")
            
        Returns:
            List of result matrices
        """
        start_time = time.time()
        
        try:
            if not self.is_gpu_available():
                # CPU fallback
                result = self._cpu_matrix_operations(matrices, operation)
                self.performance_stats['cpu_operations'] += 1
                self.performance_stats['cpu_time'] += time.time() - start_time
                return result
            
            # GPU optimization path
            gpu_matrices = []
            cache_keys = []
            
            # Check cache and move to GPU
            for i, matrix in enumerate(matrices):
                cache_key = self._get_cache_key(matrix)
                cache_keys.append(cache_key)
                
                if cache_key in self.gpu_cache:
                    gpu_matrices.append(self.gpu_cache[cache_key])
                    self.performance_stats['cache_hits'] += 1
                else:
                    gpu_matrix = self.backend.to_device(matrix)
                    gpu_matrices.append(gpu_matrix)
                    self._add_to_cache(cache_key, gpu_matrix)
                    self.performance_stats['cache_misses'] += 1
                    self.performance_stats['memory_transfers'] += 1
            
            # Perform GPU operations
            if operation == "multiply" and len(gpu_matrices) >= 2:
                result_gpu = self._gpu_matrix_multiply_chain(gpu_matrices)
            elif operation == "add" and len(gpu_matrices) >= 2:
                result_gpu = self._gpu_matrix_add_chain(gpu_matrices)
            else:
                # Auto-detect operation based on matrix shapes
                result_gpu = self._auto_gpu_operations(gpu_matrices)
            
            # Move results back to CPU
            if isinstance(result_gpu, list):
                results = [self.backend.to_cpu(gpu_result) for gpu_result in result_gpu]
            else:
                results = [self.backend.to_cpu(result_gpu)]
            
            self.performance_stats['gpu_operations'] += 1
            self.performance_stats['gpu_time'] += time.time() - start_time
            
            return results
            
        except Exception as e:
            logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
            result = self._cpu_matrix_operations(matrices, operation)
            self.performance_stats['cpu_operations'] += 1
            self.performance_stats['cpu_time'] += time.time() - start_time
            return result
    
    def _cpu_matrix_operations(self, 
                              matrices: List[np.ndarray], 
                              operation: str) -> List[np.ndarray]:
        """CPU fallback for matrix operations."""
        if operation == "multiply" and len(matrices) >= 2:
            result = matrices[0]
            for matrix in matrices[1:]:
                if result.shape[1] == matrix.shape[0]:
                    result = np.dot(result, matrix)
                else:
                    # Element-wise multiplication if shapes don't match for dot product
                    result = result * matrix
            return [result]
        
        elif operation == "add" and len(matrices) >= 2:
            result = matrices[0]
            for matrix in matrices[1:]:
                result = result + matrix
            return [result]
        
        else:
            # Auto operation
            if len(matrices) >= 2:
                # Try matrix multiplication first
                try:
                    result = matrices[0]
                    for matrix in matrices[1:]:
                        if result.shape[1] == matrix.shape[0]:
                            result = np.dot(result, matrix)
                        else:
                            result = result + matrix  # Fallback to addition
                    return [result]
                except Exception:
                    # Fallback to element-wise operations
                    return [sum(matrices)]
            else:
                return matrices
    
    def _gpu_matrix_multiply_chain(self, gpu_matrices: List) -> Any:
        """Chain matrix multiplications on GPU."""
        result = gpu_matrices[0]
        for matrix in gpu_matrices[1:]:
            result = self.backend.multiply(result, matrix)
        return result
    
    def _gpu_matrix_add_chain(self, gpu_matrices: List) -> Any:
        """Chain matrix additions on GPU."""
        result = gpu_matrices[0]
        for matrix in gpu_matrices[1:]:
            result = self.backend.add(result, matrix)
        return result
    
    def _auto_gpu_operations(self, gpu_matrices: List) -> Any:
        """Automatically determine and execute appropriate GPU operations."""
        if len(gpu_matrices) < 2:
            return gpu_matrices[0] if gpu_matrices else None
        
        # Try matrix multiplication chain
        try:
            return self._gpu_matrix_multiply_chain(gpu_matrices)
        except Exception:
            # Fallback to addition chain
            return self._gpu_matrix_add_chain(gpu_matrices)
    
    def _get_cache_key(self, matrix: np.ndarray) -> str:
        """Generate cache key for matrix."""
        # Use shape and data hash for cache key
        shape_str = "_".join(map(str, matrix.shape))
        data_hash = hash(matrix.data.tobytes())
        return f"{shape_str}_{data_hash}"
    
    def _add_to_cache(self, key: str, gpu_matrix: Any):
        """Add matrix to GPU cache with size management."""
        # Estimate GPU memory usage
        estimated_size = gpu_matrix.nbytes if hasattr(gpu_matrix, 'nbytes') else 1024
        
        # Check if we need to evict items
        while (self.current_cache_size + estimated_size > self.cache_size_limit 
               and self.gpu_cache):
            # Evict oldest item (FIFO)
            oldest_key = next(iter(self.gpu_cache))
            del self.gpu_cache[oldest_key]
            self.current_cache_size -= estimated_size  # Approximate
        
        # Add new item
        self.gpu_cache[key] = gpu_matrix
        self.current_cache_size += estimated_size
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        self.gpu_cache.clear()
        self.current_cache_size = 0
        logger.info("GPU cache cleared")
    
    def optimize_belief_update(self, 
                              prior_beliefs: np.ndarray,
                              likelihood_matrix: np.ndarray,
                              observations: np.ndarray) -> np.ndarray:
        """
        GPU-optimized belief update computation.
        
        Args:
            prior_beliefs: Prior belief state
            likelihood_matrix: Observation likelihood matrix
            observations: Current observations
            
        Returns:
            Updated belief state
        """
        try:
            matrices = [likelihood_matrix, prior_beliefs.reshape(-1, 1)]
            result_matrices = self.optimize_matrix_operations(matrices, "multiply")
            
            if result_matrices:
                # Normalize beliefs
                beliefs = result_matrices[0].flatten()
                beliefs = beliefs / (np.sum(beliefs) + 1e-8)
                return beliefs
            else:
                return prior_beliefs
                
        except Exception as e:
            logger.warning(f"GPU belief update failed: {e}")
            # CPU fallback
            beliefs = np.dot(likelihood_matrix, prior_beliefs)
            return beliefs / (np.sum(beliefs) + 1e-8)
    
    def optimize_planning_computation(self,
                                    transition_matrices: List[np.ndarray],
                                    reward_matrices: List[np.ndarray],
                                    planning_horizon: int) -> np.ndarray:
        """
        GPU-optimized planning computation.
        
        Args:
            transition_matrices: State transition matrices
            reward_matrices: Reward matrices  
            planning_horizon: Planning horizon
            
        Returns:
            Optimal action values
        """
        try:
            # Combine all matrices for batch GPU processing
            all_matrices = transition_matrices + reward_matrices
            result_matrices = self.optimize_matrix_operations(all_matrices, "auto")
            
            if result_matrices:
                return result_matrices[0]
            else:
                return np.zeros(len(transition_matrices))
                
        except Exception as e:
            logger.warning(f"GPU planning computation failed: {e}")
            # CPU fallback
            return np.sum([np.trace(matrix) for matrix in transition_matrices])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU optimization performance statistics."""
        total_ops = self.performance_stats['gpu_operations'] + self.performance_stats['cpu_operations']
        total_time = self.performance_stats['gpu_time'] + self.performance_stats['cpu_time']
        
        return {
            'gpu_available': self.is_gpu_available(),
            'backend': type(self.backend).__name__,
            'device_count': self.backend.get_device_count(),
            'performance': self.performance_stats.copy(),
            'gpu_utilization': (
                self.performance_stats['gpu_operations'] / max(total_ops, 1) * 100
            ),
            'speedup_ratio': (
                self.performance_stats['cpu_time'] / max(self.performance_stats['gpu_time'], 0.001)
                if self.performance_stats['gpu_time'] > 0 else 1.0
            ),
            'cache_hit_rate': (
                self.performance_stats['cache_hits'] / 
                max(self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'], 1) * 100
            ),
            'cache_size_mb': self.current_cache_size / 1024 / 1024
        }
    
    def benchmark_performance(self, matrix_sizes: List[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance.
        
        Args:
            matrix_sizes: List of (rows, cols) matrix sizes to test
            
        Returns:
            Performance benchmark results
        """
        if matrix_sizes is None:
            matrix_sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        results = {}
        
        for size in matrix_sizes:
            # Generate test matrices
            matrix_a = np.random.randn(*size)
            matrix_b = np.random.randn(size[1], size[0])
            
            # CPU benchmark
            start_time = time.time()
            cpu_result = np.dot(matrix_a, matrix_b)
            cpu_time = time.time() - start_time
            
            # GPU benchmark (if available)
            if self.is_gpu_available():
                start_time = time.time()
                gpu_matrices = self.optimize_matrix_operations([matrix_a, matrix_b], "multiply")
                gpu_time = time.time() - start_time
                speedup = cpu_time / max(gpu_time, 0.001)
            else:
                gpu_time = float('inf')
                speedup = 0.0
            
            results[f"{size[0]}x{size[1]}"] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
        
        return results