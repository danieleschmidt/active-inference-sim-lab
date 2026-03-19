"""
Active Inference Agent.

The agent acts by minimising Expected Free Energy (EFE):

    a* = argmin_a G(a)

where G(a) = epistemic_value(a) + pragmatic_value(a).

After acting the agent updates its beliefs using variational inference:
    q(s) ← infer_states(obs)

The prior D is also updated over time (Bayesian belief updating):
    D ← q(s)   (the posterior becomes the prior for the next step)
"""

import numpy as np
from .generative_model import GenerativeModel
from .free_energy import VariationalFreeEnergy


class ActiveInferenceAgent:
    """
    Discrete active inference agent that minimises Expected Free Energy.

    Parameters
    ----------
    model : GenerativeModel
        The agent's generative model of the environment.
    alpha : float
        Precision (inverse temperature) for action selection.
        Higher values → more greedy (default 4.0).
    """

    def __init__(self, model: GenerativeModel, alpha: float = 4.0) -> None:
        self.model = model
        self.alpha = alpha
        self.vfe = VariationalFreeEnergy()

        # Current belief over hidden states — initialise to prior
        self.q_s: np.ndarray = model.D.copy()

        # History for analysis
        self.free_energy_history: list[float] = []
        self.action_history: list[int] = []
        self.belief_history: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def infer_states(self, obs: int) -> np.ndarray:
        """
        Update belief q(s) via variational inference given observation.

        Parameters
        ----------
        obs : int
            Discrete observation index.

        Returns
        -------
        q_s : np.ndarray
            Updated posterior belief over hidden states.
        """
        self.q_s = self.model.infer_states(obs)
        self.belief_history.append(self.q_s.copy())
        return self.q_s

    def select_action(self) -> int:
        """
        Select action by minimising Expected Free Energy.

        G(a) is computed for each action; action probabilities are:
            P(a) = softmax(-alpha * G(a))

        Returns
        -------
        action : int
            Selected action index.
        """
        G = np.array([
            self.vfe.expected_free_energy(self.q_s, a, self.model)
            for a in range(self.model.n_actions)
        ])
        # Softmax over negative EFE (prefer low EFE)
        log_probs = -self.alpha * G
        log_probs -= log_probs.max()
        action_probs = np.exp(log_probs) / np.exp(log_probs).sum()
        action = int(np.random.choice(self.model.n_actions, p=action_probs))
        self.action_history.append(action)
        return action

    def update_beliefs(self, obs: int, action: int) -> None:
        """
        Update beliefs after observing a new observation.

        The posterior q(s) at time t becomes the prior for time t+1.
        The model's prior D is shifted to match the predicted next state.

        Parameters
        ----------
        obs : int
            New observation received.
        action : int
            Action that was taken (used to update the prior).
        """
        # Predict next-state prior using the transition model
        self.model.D = self.model.predict_next_state(self.q_s, action)
        self.model.D = np.clip(self.model.D, 1e-16, 1.0)
        self.model.D /= self.model.D.sum()

        # Infer new states given new observation
        self.infer_states(obs)

        # Track free energy
        F = self.vfe.compute(self.q_s, obs, self.model)
        self.free_energy_history.append(F)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def most_likely_state(self) -> int:
        """Return the MAP state estimate."""
        return int(np.argmax(self.q_s))

    def reset(self) -> None:
        """Reset beliefs to prior."""
        self.q_s = self.model.D.copy()
        self.free_energy_history.clear()
        self.action_history.clear()
        self.belief_history.clear()
