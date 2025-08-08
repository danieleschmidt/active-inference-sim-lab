"""
Research validation and benchmarking framework for Active Inference.

This module provides tools for validating Active Inference implementations
against theoretical predictions and published results.
"""

from .validation import (
    TheoreticalValidator,
    FreeEnergyValidator,
    ConvergenceValidator,
    BehaviorValidator
)

from .benchmarks import (
    AXIOMBenchmark,
    ComparativeBenchmark,
    SampleEfficiencyBenchmark,
    ReproducibilityBenchmark
)

from .analysis import (
    StatisticalAnalyzer,
    PerformanceAnalyzer,
    BehaviorAnalyzer,
    NoveltyDetector
)

from .experiments import (
    ExperimentFramework,
    ControlledExperiment,
    AblationStudy,
    ParameterSweep
)

__all__ = [
    # Validation
    'TheoreticalValidator',
    'FreeEnergyValidator', 
    'ConvergenceValidator',
    'BehaviorValidator',
    
    # Benchmarks
    'AXIOMBenchmark',
    'ComparativeBenchmark',
    'SampleEfficiencyBenchmark',
    'ReproducibilityBenchmark',
    
    # Analysis
    'StatisticalAnalyzer',
    'PerformanceAnalyzer',
    'BehaviorAnalyzer',
    'NoveltyDetector',
    
    # Experiments
    'ExperimentFramework',
    'ControlledExperiment',
    'AblationStudy',
    'ParameterSweep',
]