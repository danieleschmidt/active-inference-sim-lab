"""
Active Inference Simulation Laboratory

A high-performance toolkit for building active inference agents based on the Free Energy Principle.
Includes C++ core for performance, Python bindings for ease of use, and integration with popular RL environments.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "info@terragonlabs.com"

# Core imports
from .core import (
    ActiveInferenceAgent,
    GenerativeModel,
    FreeEnergyObjective,
)

from .inference import (
    BeliefUpdater,
    VariationalInference,
    VariationalBeliefUpdater,
    KalmanBeliefUpdater,
    ParticleBeliefUpdater,
)

from .planning import (
    ActivePlanner,
    ExpectedFreeEnergy,
    ExpectedFreeEnergyComponents,
    TrajectoryOptimizer,
    ModelPredictiveController,
)

from .environments import (
    MockEnvironment,
    ActiveInferenceGridWorld,
    ForagingEnvironment,
    SocialDilemmaEnvironment,
    TheoryOfMindEnvironment,
)

# Module-level exports
__all__ = [
    # Core classes
    "ActiveInferenceAgent",
    "GenerativeModel", 
    "FreeEnergyObjective",
    
    # Inference
    "BeliefUpdater",
    "VariationalInference",
    "VariationalBeliefUpdater",
    "KalmanBeliefUpdater", 
    "ParticleBeliefUpdater",
    
    # Planning
    "ActivePlanner",
    "ExpectedFreeEnergy",
    "ExpectedFreeEnergyComponents",
    "TrajectoryOptimizer",
    "ModelPredictiveController",
    
    # Environments
    "MockEnvironment",
    "ActiveInferenceGridWorld",
    "ForagingEnvironment", 
    "SocialDilemmaEnvironment",
    "TheoryOfMindEnvironment",
    
    # Version info
    "__version__",
]