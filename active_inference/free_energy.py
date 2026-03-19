"""
Variational Free Energy computation.

Free energy under the variational bound:

    F = E_q[ log q(s) - log P(o, s) ]
      = KL[ q(s) || P(s) ]  -  E_q[ log P(o | s) ]

where:
    KL[ q || p ] = sum_s q(s) * log( q(s) / p(s) )   (complexity)
    E_q[ log P(o|s) ] = sum_s q(s) * log A[o, s]      (accuracy)

A lower F means the agent's beliefs are both accurate and parsimonious.
"""

import numpy as np
from .generative_model import GenerativeModel


class VariationalFreeEnergy:
    """Compute the variational free energy F for discrete state-spaces."""

    @staticmethod
    def compute(q_s: np.ndarray, obs: int, model: "GenerativeModel") -> float:
        """
        Compute variational free energy.

        F = KL[ q(s) || D ] - E_q[ log A[o, s] ]

        Parameters
        ----------
        q_s : np.ndarray, shape (n_states,)
            Current posterior belief over hidden states.
        obs : int
            Current observation index.
        model : GenerativeModel
            The agent's generative model.

        Returns
        -------
        F : float
            Variational free energy (lower is better).
        """
        q_s = np.clip(q_s, 1e-16, 1.0)
        D = np.clip(model.D, 1e-16, 1.0)
        A_obs = np.clip(model.A[obs, :], 1e-16, 1.0)

        # Complexity: KL[ q(s) || D ]
        kl = float(np.sum(q_s * (np.log(q_s) - np.log(D))))

        # Accuracy: -E_q[ log P(o | s) ]
        neg_accuracy = float(-np.sum(q_s * np.log(A_obs)))

        return kl + neg_accuracy

    @staticmethod
    def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
        """KL[ q || p ] for discrete distributions."""
        q = np.clip(q, 1e-16, 1.0)
        p = np.clip(p, 1e-16, 1.0)
        return float(np.sum(q * (np.log(q) - np.log(p))))

    @staticmethod
    def expected_free_energy(
        q_s: np.ndarray,
        action: int,
        model: "GenerativeModel",
    ) -> float:
        """
        Expected Free Energy G(a) for a candidate action.

        G(a) = epistemic_value + pragmatic_value

        Epistemic value (information gain / Bayesian surprise):
            -sum_o P(o|a) * H[P(s|o,a)]
            ≈ -sum_o P(o|a) * sum_s P(s|o,a) * log P(s|o,a)

        Pragmatic value (KL from preferred observations):
            KL[ P(o|a) || C ]
            = sum_o P(o|a) * (log P(o|a) - C[o])

        Lower G is better (agent prefers actions that reduce uncertainty
        and bring observations closer to its preferred ones).

        Parameters
        ----------
        q_s : np.ndarray
            Current belief over hidden states.
        action : int
            Candidate action index.
        model : GenerativeModel
            Generative model with A, B, D, C matrices.

        Returns
        -------
        G : float
            Expected free energy for this action.
        """
        # Predicted next-state distribution under this action
        qs_next = model.predict_next_state(q_s, action)          # (n_states,)
        qs_next = np.clip(qs_next, 1e-16, 1.0)
        qs_next /= qs_next.sum()

        # Predicted observation distribution: P(o|a) = A @ qs_next
        po = model.predict_obs(qs_next)                           # (n_obs,)
        po = np.clip(po, 1e-16, 1.0)
        po /= po.sum()

        # --- Epistemic value: expected entropy reduction ---
        # For each possible observation, compute the posterior P(s|o,a)
        # and its entropy, then take expectation over P(o|a)
        epistemic = 0.0
        for o in range(model.n_obs):
            # Posterior over states given this obs and predicted next-state prior
            log_post = np.log(model.A[o, :] + 1e-16) + np.log(qs_next)
            ps_given_o = _softmax(log_post)
            # Entropy of P(s|o,a) — negative info gain contribution
            H_s_given_o = -np.sum(ps_given_o * np.log(np.clip(ps_given_o, 1e-16, 1)))
            epistemic += po[o] * H_s_given_o

        # --- Pragmatic value: KL from preferred observations ---
        # C is log-preferred, so preferred_prob ∝ exp(C)
        preferred = np.exp(model.C)
        preferred = np.clip(preferred / preferred.sum(), 1e-16, 1.0)
        pragmatic = float(np.sum(po * (np.log(po) - np.log(preferred))))

        return epistemic + pragmatic


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
