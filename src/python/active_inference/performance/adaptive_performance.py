"""
Adaptive Performance Optimization System
Generation 3: MAKE IT SCALE - Self-Optimizing Performance Engine
"""

import time
import logging
import threading
import psutil
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import gc

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Performance optimization modes."""
    THROUGHPUT = "throughput"        # Maximum throughput
    LATENCY = "latency"             # Minimum latency
    BALANCED = "balanced"           # Balance throughput and latency
    MEMORY_EFFICIENT = "memory"     # Minimize memory usage
    CPU_EFFICIENT = "cpu"           # Minimize CPU usage
    ADAPTIVE = "adaptive"           # Automatically adapt based on conditions


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    queue_depth: int = 0
    active_threads: int = 0
    active_processes: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    gc_collections: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for analysis."""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'throughput': self.throughput,
            'latency': self.latency,
            'queue_depth': float(self.queue_depth),
            'active_threads': float(self.active_threads),
            'active_processes': float(self.active_processes),
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate,
            'gc_collections': float(self.gc_collections)
        }


@dataclass
class OptimizationStrategy:
    """Strategy for performance optimization."""
    name: str
    target_metric: str
    optimization_function: Callable[[PerformanceMetrics], Dict[str, Any]]
    conditions: Dict[str, float] = field(default_factory=dict)
    priority: int = 1
    cooldown: float = 30.0  # Seconds between applications
    last_applied: float = 0.0


class AdaptivePerformanceOptimizer:
    """
    Self-adaptive performance optimization system that monitors system metrics
    and automatically adjusts configuration for optimal performance.
    """
    
    def __init__(self,
                 optimization_mode: OptimizationMode = OptimizationMode.ADAPTIVE,
                 monitoring_interval: float = 5.0,
                 history_length: int = 1000,
                 enable_auto_tuning: bool = True,
                 enable_predictive_scaling: bool = True):
        """
        Initialize adaptive performance optimizer.
        
        Args:
            optimization_mode: Target optimization mode
            monitoring_interval: Seconds between metric collection
            history_length: Number of historical metrics to keep
            enable_auto_tuning: Enable automatic parameter tuning
            enable_predictive_scaling: Enable predictive resource scaling
        """
        self.optimization_mode = optimization_mode
        self.monitoring_interval = monitoring_interval
        self.history_length = history_length
        self.enable_auto_tuning = enable_auto_tuning
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Performance tracking
        self._metrics_history = deque(maxlen=history_length)
        self._current_metrics = PerformanceMetrics()
        self._baseline_metrics = None
        
        # Resource management
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._optimal_thread_count = mp.cpu_count()
        self._optimal_process_count = max(1, mp.cpu_count() // 2)
        
        # Optimization strategies
        self._strategies: List[OptimizationStrategy] = []
        self._active_optimizations: Dict[str, Any] = {}
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Statistics
        self._optimization_count = 0
        self._performance_improvements = []
        self._last_optimization_time = 0.0
        
        # Setup default strategies
        self._setup_optimization_strategies()
        
        logger.info(f"Adaptive performance optimizer initialized (mode: {optimization_mode.value})")
    
    def _setup_optimization_strategies(self):
        """Setup default optimization strategies."""
        # Throughput optimization
        self._strategies.append(OptimizationStrategy(
            name="increase_parallelism",
            target_metric="throughput",
            optimization_function=self._increase_parallelism,
            conditions={'cpu_usage': 60.0, 'queue_depth': 10.0},
            priority=2
        ))
        
        # Latency optimization
        self._strategies.append(OptimizationStrategy(
            name="reduce_latency",
            target_metric="latency",
            optimization_function=self._reduce_latency,
            conditions={'latency': 1.0},
            priority=3
        ))
        
        # Memory optimization
        self._strategies.append(OptimizationStrategy(
            name="optimize_memory",
            target_metric="memory_usage",
            optimization_function=self._optimize_memory,
            conditions={'memory_usage': 80.0},
            priority=2
        ))
        
        # CPU optimization
        self._strategies.append(OptimizationStrategy(
            name="optimize_cpu",
            target_metric="cpu_usage",
            optimization_function=self._optimize_cpu,
            conditions={'cpu_usage': 85.0},
            priority=1
        ))
        
        # Garbage collection optimization
        self._strategies.append(OptimizationStrategy(
            name="optimize_gc",
            target_metric="gc_collections",
            optimization_function=self._optimize_garbage_collection,
            conditions={'memory_usage': 70.0, 'gc_collections': 5.0},
            priority=1
        ))
    
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Performance monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
        
        # Initialize thread and process pools
        self._initialize_pools()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join(timeout=10.0)
        
        # Shutdown pools
        self._shutdown_pools()
        
        logger.info("Performance monitoring stopped")
    
    def _initialize_pools(self):
        """Initialize thread and process pools."""
        try:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self._optimal_thread_count,
                thread_name_prefix="AdaptivePerf"
            )
            
            self._process_pool = ProcessPoolExecutor(
                max_workers=self._optimal_process_count
            )
            
            logger.debug(f"Initialized pools: {self._optimal_thread_count} threads, {self._optimal_process_count} processes")
            
        except Exception as e:
            logger.error(f"Failed to initialize pools: {e}")
    
    def _shutdown_pools(self):
        """Shutdown thread and process pools."""
        try:
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
            
            if self._process_pool:
                self._process_pool.shutdown(wait=True)
                self._process_pool = None
                
        except Exception as e:
            logger.error(f"Error shutting down pools: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Apply optimizations if enabled
                if self.enable_auto_tuning:
                    self._apply_optimizations()
                
                # Predictive scaling
                if self.enable_predictive_scaling:
                    self._predictive_scaling()
                
                # Sleep until next collection
                self._stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(1.0)
    
    def _collect_metrics(self) -> None:
        """Collect current system performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # GC metrics
            gc_stats = gc.get_stats()
            gc_collections = sum(stat.get('collections', 0) for stat in gc_stats)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                throughput=self._calculate_throughput(),
                latency=self._calculate_average_latency(),
                queue_depth=self._calculate_queue_depth(),
                active_threads=threading.active_count(),
                active_processes=len(psutil.Process().children(recursive=True)) + 1,
                cache_hit_rate=self._calculate_cache_hit_rate(),
                error_rate=self._calculate_error_rate(),
                gc_collections=gc_collections
            )
            
            with self._lock:
                self._current_metrics = metrics
                self._metrics_history.append(metrics)
            
            # Set baseline if not set
            if self._baseline_metrics is None:
                self._baseline_metrics = metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _calculate_throughput(self) -> float:
        """Calculate current system throughput (operations per second)."""
        # This would be implemented based on specific application metrics
        # For now, return a placeholder based on CPU efficiency
        if len(self._metrics_history) < 2:
            return 0.0
        
        recent_metrics = list(self._metrics_history)[-10:]  # Last 10 samples
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        
        # Estimate throughput based on CPU utilization (placeholder logic)
        return max(0, 100 - avg_cpu) * 2.0
    
    def _calculate_average_latency(self) -> float:
        """Calculate average response latency."""
        # Placeholder implementation - would be application-specific
        if len(self._metrics_history) < 2:
            return 0.0
        
        recent_metrics = list(self._metrics_history)[-5:]
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        
        # Estimate latency based on system load
        return 0.1 + (avg_cpu / 100.0) * 2.0
    
    def _calculate_queue_depth(self) -> int:
        """Calculate current processing queue depth."""
        # Placeholder - would be application-specific
        if self._thread_pool:
            # Rough estimation based on thread pool usage
            return max(0, threading.active_count() - self._optimal_thread_count)
        return 0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Placeholder - would be application-specific
        return 0.85  # Assume 85% hit rate
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # Placeholder - would be application-specific
        return 0.01  # Assume 1% error rate
    
    def _apply_optimizations(self) -> None:
        """Apply performance optimizations based on current metrics."""
        current_time = time.time()
        metrics_dict = self._current_metrics.to_dict()
        
        # Sort strategies by priority
        sorted_strategies = sorted(self._strategies, key=lambda s: s.priority, reverse=True)
        
        for strategy in sorted_strategies:
            # Check cooldown
            if current_time - strategy.last_applied < strategy.cooldown:
                continue
            
            # Check conditions
            should_apply = True
            for condition_metric, threshold in strategy.conditions.items():
                if metrics_dict.get(condition_metric, 0) < threshold:
                    should_apply = False
                    break
            
            if should_apply:
                try:
                    optimization_result = strategy.optimization_function(self._current_metrics)
                    
                    if optimization_result.get('applied', False):
                        strategy.last_applied = current_time
                        self._optimization_count += 1
                        self._last_optimization_time = current_time
                        
                        logger.info(f"Applied optimization '{strategy.name}': {optimization_result}")
                        
                        # Record performance improvement
                        self._record_performance_improvement(strategy.name, optimization_result)
                        
                        # Only apply one optimization per cycle to avoid conflicts
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to apply optimization '{strategy.name}': {e}")
    
    def _increase_parallelism(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Increase system parallelism to improve throughput."""
        old_thread_count = self._optimal_thread_count
        
        # Increase thread count if CPU usage is moderate and queue depth is high
        if metrics.cpu_usage < 75 and metrics.queue_depth > 5:
            new_thread_count = min(self._optimal_thread_count + 2, mp.cpu_count() * 2)
            
            if new_thread_count > old_thread_count:
                self._optimal_thread_count = new_thread_count
                self._reinitialize_thread_pool()
                
                return {
                    'applied': True,
                    'old_thread_count': old_thread_count,
                    'new_thread_count': new_thread_count,
                    'improvement_type': 'throughput'
                }
        
        return {'applied': False, 'reason': 'conditions_not_met'}
    
    def _reduce_latency(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize for reduced latency."""
        # Reduce parallelism to decrease context switching overhead
        if metrics.latency > 2.0 and self._optimal_thread_count > mp.cpu_count():
            old_thread_count = self._optimal_thread_count
            self._optimal_thread_count = max(mp.cpu_count(), self._optimal_thread_count - 1)
            self._reinitialize_thread_pool()
            
            return {
                'applied': True,
                'old_thread_count': old_thread_count,
                'new_thread_count': self._optimal_thread_count,
                'improvement_type': 'latency'
            }
        
        return {'applied': False, 'reason': 'conditions_not_met'}
    
    def _optimize_memory(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize memory usage."""
        optimizations_applied = []
        
        # Force garbage collection
        if metrics.memory_usage > 85:
            gc.collect()
            optimizations_applied.append('garbage_collection')
        
        # Reduce thread pool size to save memory
        if metrics.memory_usage > 90 and self._optimal_thread_count > 2:
            old_count = self._optimal_thread_count
            self._optimal_thread_count = max(2, self._optimal_thread_count - 1)
            self._reinitialize_thread_pool()
            optimizations_applied.append(f'reduced_threads_{old_count}_to_{self._optimal_thread_count}')
        
        if optimizations_applied:
            return {
                'applied': True,
                'optimizations': optimizations_applied,
                'improvement_type': 'memory'
            }
        
        return {'applied': False, 'reason': 'conditions_not_met'}
    
    def _optimize_cpu(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize CPU usage."""
        if metrics.cpu_usage > 90:
            # Reduce parallelism to decrease CPU contention
            if self._optimal_thread_count > 1:
                old_count = self._optimal_thread_count
                self._optimal_thread_count = max(1, self._optimal_thread_count - 1)
                self._reinitialize_thread_pool()
                
                return {
                    'applied': True,
                    'old_thread_count': old_count,
                    'new_thread_count': self._optimal_thread_count,
                    'improvement_type': 'cpu'
                }
        
        return {'applied': False, 'reason': 'conditions_not_met'}
    
    def _optimize_garbage_collection(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize garbage collection settings."""
        # Force collection and tune thresholds
        old_thresholds = gc.get_threshold()
        
        if metrics.memory_usage > 80:
            # More aggressive GC
            gc.set_threshold(700, 10, 10)  # Default is (700, 10, 10)
            gc.collect()
            
            return {
                'applied': True,
                'old_thresholds': old_thresholds,
                'new_thresholds': gc.get_threshold(),
                'improvement_type': 'memory'
            }
        
        return {'applied': False, 'reason': 'conditions_not_met'}
    
    def _reinitialize_thread_pool(self):
        """Reinitialize thread pool with new configuration."""
        try:
            if self._thread_pool:
                self._thread_pool.shutdown(wait=False)
            
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self._optimal_thread_count,
                thread_name_prefix="AdaptivePerf"
            )
            
        except Exception as e:
            logger.error(f"Failed to reinitialize thread pool: {e}")
    
    def _predictive_scaling(self):
        """Implement predictive scaling based on trends."""
        if len(self._metrics_history) < 10:
            return
        
        # Analyze trends in key metrics
        recent_metrics = list(self._metrics_history)[-10:]
        
        # CPU trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        
        # Memory trend
        memory_values = [m.memory_usage for m in recent_metrics]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        
        # Predictive adjustments
        if cpu_trend > 2.0:  # CPU usage increasing rapidly
            logger.info("Predicting high CPU usage, preparing for scale-up")
            # Could pre-allocate resources or adjust thresholds
        
        if memory_trend > 1.5:  # Memory usage increasing
            logger.info("Predicting high memory usage, preparing for cleanup")
            # Could trigger proactive garbage collection
    
    def _record_performance_improvement(self, strategy_name: str, result: Dict[str, Any]):
        """Record performance improvement for analysis."""
        improvement_record = {
            'timestamp': time.time(),
            'strategy': strategy_name,
            'result': result,
            'metrics_before': self._current_metrics.to_dict(),
            'thread_count': self._optimal_thread_count,
            'process_count': self._optimal_process_count
        }
        
        self._performance_improvements.append(improvement_record)
        
        # Keep only recent improvements
        if len(self._performance_improvements) > 100:
            self._performance_improvements = self._performance_improvements[-100:]
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return self._current_metrics
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics and performance data."""
        with self._lock:
            if not self._metrics_history:
                return {'status': 'no_data'}
            
            # Calculate performance trends
            recent_metrics = list(self._metrics_history)[-20:]  # Last 20 samples
            
            cpu_trend = np.mean([m.cpu_usage for m in recent_metrics])
            memory_trend = np.mean([m.memory_usage for m in recent_metrics])
            throughput_trend = np.mean([m.throughput for m in recent_metrics])
            latency_trend = np.mean([m.latency for m in recent_metrics])
            
            # Performance vs baseline
            improvement = {}
            if self._baseline_metrics:
                current = self._current_metrics
                baseline = self._baseline_metrics
                
                improvement = {
                    'cpu_improvement': baseline.cpu_usage - current.cpu_usage,
                    'memory_improvement': baseline.memory_usage - current.memory_usage,
                    'throughput_improvement': current.throughput - baseline.throughput,
                    'latency_improvement': baseline.latency - current.latency
                }
            
            return {
                'optimization_mode': self.optimization_mode.value,
                'optimization_count': self._optimization_count,
                'last_optimization_time': self._last_optimization_time,
                'current_metrics': self._current_metrics.to_dict(),
                'trends': {
                    'cpu_usage': cpu_trend,
                    'memory_usage': memory_trend,
                    'throughput': throughput_trend,
                    'latency': latency_trend
                },
                'performance_improvement': improvement,
                'resource_configuration': {
                    'optimal_thread_count': self._optimal_thread_count,
                    'optimal_process_count': self._optimal_process_count
                },
                'strategy_count': len(self._strategies),
                'recent_improvements': len(self._performance_improvements),
                'monitoring_active': self._monitor_thread is not None and self._monitor_thread.is_alive()
            }
    
    def get_resource_pools(self) -> Tuple[Optional[ThreadPoolExecutor], Optional[ProcessPoolExecutor]]:
        """Get access to optimized resource pools."""
        return self._thread_pool, self._process_pool
    
    def execute_optimized(self, func: Callable, use_process_pool: bool = False, *args, **kwargs):
        """Execute function using optimized resource pools."""
        if use_process_pool and self._process_pool:
            return self._process_pool.submit(func, *args, **kwargs)
        elif self._thread_pool:
            return self._thread_pool.submit(func, *args, **kwargs)
        else:
            # Fallback to direct execution
            return func(*args, **kwargs)
    
    def set_optimization_mode(self, mode: OptimizationMode):
        """Change optimization mode."""
        old_mode = self.optimization_mode
        self.optimization_mode = mode
        
        logger.info(f"Optimization mode changed from {old_mode.value} to {mode.value}")
        
        # Adjust strategies based on new mode
        self._adjust_strategies_for_mode(mode)
    
    def _adjust_strategies_for_mode(self, mode: OptimizationMode):
        """Adjust optimization strategies based on mode."""
        # This could implement mode-specific strategy prioritization
        # For now, just log the change
        logger.debug(f"Adjusting strategies for {mode.value} mode")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"AdaptivePerformanceOptimizer(mode={self.optimization_mode.value}, "
                f"optimizations={self._optimization_count}, "
                f"threads={self._optimal_thread_count})")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Global performance optimizer instance
global_performance_optimizer: Optional[AdaptivePerformanceOptimizer] = None


def get_performance_optimizer() -> Optional[AdaptivePerformanceOptimizer]:
    """Get global performance optimizer instance."""
    return global_performance_optimizer


def initialize_global_optimizer(mode: OptimizationMode = OptimizationMode.ADAPTIVE) -> AdaptivePerformanceOptimizer:
    """Initialize global performance optimizer."""
    global global_performance_optimizer
    
    if global_performance_optimizer is None:
        global_performance_optimizer = AdaptivePerformanceOptimizer(optimization_mode=mode)
        global_performance_optimizer.start_monitoring()
        logger.info("Global performance optimizer initialized")
    
    return global_performance_optimizer


def performance_optimized(use_process_pool: bool = False):
    """
    Decorator for executing functions with performance optimization.
    
    Args:
        use_process_pool: Whether to prefer process pool over thread pool
        
    Returns:
        Decorated function that uses optimized execution
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            if optimizer:
                return optimizer.execute_optimized(func, use_process_pool, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator