"""
Environment integrations for active inference agents.
"""

from .mock_env import MockEnvironment
from .grid_world import ActiveInferenceGridWorld, ForagingEnvironment
from .social_environment import SocialDilemmaEnvironment, TheoryOfMindEnvironment

# Try to import gym wrapper
try:
    from .gym_wrapper import GymWrapper
    __all__ = ["MockEnvironment", "GymWrapper", "ActiveInferenceGridWorld", 
               "ForagingEnvironment", "SocialDilemmaEnvironment", "TheoryOfMindEnvironment"]
except ImportError:
    __all__ = ["MockEnvironment", "ActiveInferenceGridWorld", 
               "ForagingEnvironment", "SocialDilemmaEnvironment", "TheoryOfMindEnvironment"]