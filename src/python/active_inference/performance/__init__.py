"""
Performance optimization and scaling tools for Active Inference.

This module provides GPU acceleration, parallel processing, and
optimization tools for production deployment.
"""

from .optimization import (
    OptimizedActiveInferenceAgent,
    OptimizationConfig
)

from .caching import (
    BaseCache,
    CacheStrategy
)

__all__ = [
    # Optimization
    'OptimizedActiveInferenceAgent',
    'OptimizationConfig',
    
    # Caching
    'BaseCache',
    'CacheStrategy',
]