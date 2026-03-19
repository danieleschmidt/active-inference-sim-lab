"""
Active Inference package — Free Energy Principle agents.

Core components:
    GenerativeModel       — encodes P(o,s) = P(o|s) * P(s)
    VariationalFreeEnergy — computes F = KL[q(s)||P(s)] - E_q[log P(o|s)]
    ActiveInferenceAgent  — selects actions by minimising Expected Free Energy
"""

from .generative_model import GenerativeModel
from .free_energy import VariationalFreeEnergy
from .agent import ActiveInferenceAgent

__all__ = ["GenerativeModel", "VariationalFreeEnergy", "ActiveInferenceAgent"]
__version__ = "0.1.0"
