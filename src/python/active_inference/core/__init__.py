"""
Core active inference components.

This module contains the fundamental classes and functions for implementing
active inference agents based on the Free Energy Principle.
"""

from .agent import ActiveInferenceAgent
from .generative_model import GenerativeModel
from .free_energy import FreeEnergyObjective
from .beliefs import Belief, BeliefState
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, circuit_breaker, circuit_registry

__all__ = [
    "ActiveInferenceAgent",
    "GenerativeModel",
    "FreeEnergyObjective", 
    "Belief",
    "BeliefState",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "circuit_breaker",
    "circuit_registry",
]