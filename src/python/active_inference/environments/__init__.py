"""
Environment integrations for active inference agents.
"""

from .mock_env import MockEnvironment

# Try to import gym wrapper
try:
    from .gym_wrapper import GymWrapper
    __all__ = ["MockEnvironment", "GymWrapper"]
except ImportError:
    __all__ = ["MockEnvironment"]