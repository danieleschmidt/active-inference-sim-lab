"""
Simple 1-D Grid World environment for FEP demo.

States:   0 — 4  (5 positions)
Actions:  0=left, 1=stay, 2=right
Observations: noisy readout of current state (same cardinality as states)
"""

import numpy as np


class GridWorld1D:
    """
    Deterministic 1-D grid with noisy observations.

    Parameters
    ----------
    n_states : int
        Number of positions (default 5).
    goal : int
        Goal position (default 4 = rightmost).
    obs_noise : float
        Probability mass on off-diagonal entries of A (default 0.1).
    """

    ACTION_LEFT  = 0
    ACTION_STAY  = 1
    ACTION_RIGHT = 2

    def __init__(self, n_states: int = 5, goal: int = 4, obs_noise: float = 0.1) -> None:
        self.n_states = n_states
        self.n_actions = 3
        self.n_obs = n_states
        self.goal = goal
        self.obs_noise = obs_noise

        self.state: int = 0  # start at leftmost cell

    def reset(self, start: int = 0) -> int:
        """Reset environment to start state; return initial observation."""
        self.state = start
        return self._observe()

    def step(self, action: int) -> tuple[int, bool]:
        """
        Apply action; return (observation, done).

        Parameters
        ----------
        action : int
            0=left, 1=stay, 2=right

        Returns
        -------
        obs : int
            Noisy observation of new state.
        done : bool
            True when agent reaches goal state.
        """
        if action == self.ACTION_LEFT:
            self.state = max(0, self.state - 1)
        elif action == self.ACTION_RIGHT:
            self.state = min(self.n_states - 1, self.state + 1)
        # ACTION_STAY: do nothing
        obs = self._observe()
        done = (self.state == self.goal)
        return obs, done

    def _observe(self) -> int:
        """Return a noisy observation of the current state."""
        probs = np.full(self.n_obs, self.obs_noise / (self.n_obs - 1))
        probs[self.state] = 1.0 - self.obs_noise
        return int(np.random.choice(self.n_obs, p=probs))

    # ------------------------------------------------------------------
    # Matrix constructors (used by GenerativeModel)
    # ------------------------------------------------------------------

    def build_A(self) -> np.ndarray:
        """Likelihood matrix A[o, s] = P(o | s)."""
        n = self.n_states
        A = np.full((n, n), self.obs_noise / (n - 1))
        np.fill_diagonal(A, 1.0 - self.obs_noise)
        return A

    def build_B(self) -> np.ndarray:
        """
        Transition tensor B[s', s, a] = P(s' | s, a).
        Deterministic transitions with small noise.
        """
        n = self.n_states
        B = np.zeros((n, n, self.n_actions))
        noise = 0.05

        for s in range(n):
            # Action 0: left
            s_left = max(0, s - 1)
            B[:, s, self.ACTION_LEFT] = noise / (n - 1)
            B[s_left, s, self.ACTION_LEFT] += 1.0 - noise

            # Action 1: stay
            B[:, s, self.ACTION_STAY] = noise / (n - 1)
            B[s, s, self.ACTION_STAY] += 1.0 - noise

            # Action 2: right
            s_right = min(n - 1, s + 1)
            B[:, s, self.ACTION_RIGHT] = noise / (n - 1)
            B[s_right, s, self.ACTION_RIGHT] += 1.0 - noise

        return B

    def build_D(self, start: int = 0) -> np.ndarray:
        """Prior over initial states — concentrated at start."""
        D = np.ones(self.n_states) * 0.05
        D[start] = 1.0 - 0.05 * (self.n_states - 1)
        return D / D.sum()

    def build_C(self, goal: int | None = None) -> np.ndarray:
        """
        Log-preferences C[o].  High preference for goal observation.
        Returns log-probabilities (not normalised — that's done in model).
        """
        if goal is None:
            goal = self.goal
        n = self.n_states
        C = np.zeros(n)
        C[goal] = 3.0   # strongly prefer goal observation
        return C
