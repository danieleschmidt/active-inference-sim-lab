"""
Generative Model for Active Inference.

Encodes the joint distribution P(o, s) = P(o|s) * P(s) where:
    o — observations
    s — hidden states

The model is parameterised by:
    A[o, s]    — likelihood matrix (rows: observations, cols: states)
    B[s', s, a]— transition tensor (next_state | current_state, action)
    D[s]       — prior over initial hidden states
    C[o]       — preferred observations (log-preferences)
"""

import numpy as np


class GenerativeModel:
    """
    Discrete-state generative model for active inference.

    Parameters
    ----------
    A : np.ndarray, shape (n_obs, n_states)
        Likelihood matrix. A[o, s] = P(o | s).  Columns must sum to 1.
    B : np.ndarray, shape (n_states, n_states, n_actions)
        Transition tensor. B[:, s, a] = P(s' | s, a).  Slices must sum to 1.
    D : np.ndarray, shape (n_states,)
        Prior over hidden states.  Must sum to 1.
    C : np.ndarray or None, shape (n_obs,)
        Log-preferences over observations (log P_preferred(o)).
        If None, uniform preferences are used.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        D: np.ndarray,
        C: np.ndarray | None = None,
    ) -> None:
        self.A = A / A.sum(axis=0, keepdims=True)          # normalise columns
        self.B = B / B.sum(axis=0, keepdims=True)          # normalise each slice
        self.D = D / D.sum()                               # normalise prior
        if C is None:
            self.C = np.zeros(A.shape[0])
        else:
            self.C = C - np.log(np.sum(np.exp(C)))        # log-normalise

        self.n_obs = A.shape[0]
        self.n_states = A.shape[1]
        self.n_actions = B.shape[2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def likelihood(self, obs: int, state: int) -> float:
        """P(o | s)."""
        return float(self.A[obs, state])

    def prior(self, state: int) -> float:
        """P(s) — prior over hidden states."""
        return float(self.D[state])

    def infer_states(self, obs: int) -> np.ndarray:
        """
        Compute posterior q(s) via variational mean-field inference.

        q(s) ∝ P(o | s) * P(s)   →   softmax(log A[o,:] + log D)

        Parameters
        ----------
        obs : int
            Observed discrete observation index.

        Returns
        -------
        q_s : np.ndarray, shape (n_states,)
            Posterior belief over hidden states (sums to 1).
        """
        log_likelihood = np.log(self.A[obs, :] + 1e-16)
        log_prior = np.log(self.D + 1e-16)
        log_posterior = log_likelihood + log_prior
        return _softmax(log_posterior)

    def predict_obs(self, q_s: np.ndarray) -> np.ndarray:
        """Expected observation distribution given belief q(s).  Shape: (n_obs,)."""
        return self.A @ q_s

    def predict_next_state(self, q_s: np.ndarray, action: int) -> np.ndarray:
        """Predicted next-state distribution given belief q(s) and action."""
        return self.B[:, :, action] @ q_s


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()
