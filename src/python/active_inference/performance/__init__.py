"""
Performance optimization and scaling tools for Active Inference.

This module provides GPU acceleration, parallel processing, and
optimization tools for production deployment.
"""

from .optimization import (
    OptimizedActiveInferenceAgent,
    GPUAcceleratedAgent,
    ParallelBeliefUpdater,
    VectorizedEnvironmentWrapper
)

from .profiling import (
    PerformanceProfiler,
    MemoryProfiler,
    BenchmarkProfiler,
    OptimizationRecommender
)

from .caching import (
    BeliefCache,
    ModelCache,
    AdaptiveCache,
    CacheStrategy
)

from .scaling import (
    DistributedTraining,
    MultiAgentScaling,
    HierarchicalScaling,
    CloudDeployment
)

__all__ = [
    # Optimization
    'OptimizedActiveInferenceAgent',
    'GPUAcceleratedAgent', 
    'ParallelBeliefUpdater',
    'VectorizedEnvironmentWrapper',
    
    # Profiling
    'PerformanceProfiler',
    'MemoryProfiler',
    'BenchmarkProfiler',
    'OptimizationRecommender',
    
    # Caching
    'BeliefCache',
    'ModelCache',
    'AdaptiveCache',
    'CacheStrategy',
    
    # Scaling
    'DistributedTraining',
    'MultiAgentScaling',
    'HierarchicalScaling',
    'CloudDeployment',
]