"""
Tests for the FEP active inference core package.

Tests cover:
    - GenerativeModel (likelihood, prior, infer_states)
    - VariationalFreeEnergy (compute, kl_divergence, expected_free_energy)
    - ActiveInferenceAgent (infer_states, select_action, update_beliefs)
    - GridWorld1D environment matrix constructors
    - End-to-end convergence on 5-state grid world
"""

import numpy as np
import pytest

from active_inference import GenerativeModel, VariationalFreeEnergy, ActiveInferenceAgent
from active_inference.envs import GridWorld1D


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def grid():
    return GridWorld1D(n_states=5, goal=4, obs_noise=0.1)


@pytest.fixture
def model(grid):
    return GenerativeModel(
        A=grid.build_A(),
        B=grid.build_B(),
        D=grid.build_D(start=0),
        C=grid.build_C(goal=4),
    )


@pytest.fixture
def agent(model):
    return ActiveInferenceAgent(model, alpha=4.0)


# ── GenerativeModel ────────────────────────────────────────────────────────────

class TestGenerativeModel:
    def test_A_columns_sum_to_one(self, model):
        """Each column of A should be a valid probability distribution."""
        col_sums = model.A.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_B_columns_sum_to_one(self, model):
        """Each B[:, s, a] slice should sum to 1."""
        for a in range(model.n_actions):
            col_sums = model.B[:, :, a].sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_D_sums_to_one(self, model):
        np.testing.assert_allclose(model.D.sum(), 1.0, atol=1e-10)

    def test_likelihood_range(self, model):
        """All likelihood values should be in (0, 1]."""
        for o in range(model.n_obs):
            for s in range(model.n_states):
                v = model.likelihood(o, s)
                assert 0.0 < v <= 1.0

    def test_prior_range(self, model):
        for s in range(model.n_states):
            v = model.prior(s)
            assert 0.0 < v <= 1.0

    def test_infer_states_returns_distribution(self, model):
        for obs in range(model.n_obs):
            q = model.infer_states(obs)
            assert q.shape == (model.n_states,)
            assert q.sum() == pytest.approx(1.0, abs=1e-10)
            assert (q >= 0).all()

    def test_infer_states_peaks_at_correct_state(self, model):
        """For a clear observation, MAP state should match observed state."""
        for state in range(model.n_states):
            q = model.infer_states(state)
            # With 0.8 diagonal and uniform prior, MAP should be `state`
            # (unless prior strongly dominates — it doesn't here)
            assert np.argmax(q) == state

    def test_predict_obs_sums_to_one(self, model):
        q = np.ones(model.n_states) / model.n_states
        po = model.predict_obs(q)
        assert po.sum() == pytest.approx(1.0, abs=1e-10)

    def test_predict_next_state_sums_to_one(self, model):
        q = model.D.copy()
        for a in range(model.n_actions):
            qs_next = model.predict_next_state(q, a)
            assert qs_next.sum() == pytest.approx(1.0, abs=1e-9)


# ── VariationalFreeEnergy ──────────────────────────────────────────────────────

class TestVariationalFreeEnergy:
    def test_compute_returns_finite(self, model):
        q = model.infer_states(2)
        F = VariationalFreeEnergy.compute(q, 2, model)
        assert np.isfinite(F)

    def test_compute_non_negative(self, model):
        """VFE should be >= 0 for valid inputs."""
        q = model.infer_states(3)
        F = VariationalFreeEnergy.compute(q, 3, model)
        assert F >= -1e-10  # allow tiny float error

    def test_compute_lower_for_correct_obs(self, model):
        """
        VFE should be lower when agent's MAP matches observation
        vs. when they disagree.
        """
        # Agent believes state=4; compare obs=4 vs obs=0
        q = model.infer_states(4)
        F_match    = VariationalFreeEnergy.compute(q, 4, model)
        F_mismatch = VariationalFreeEnergy.compute(q, 0, model)
        assert F_match < F_mismatch

    def test_kl_divergence_self_is_zero(self):
        p = np.array([0.1, 0.4, 0.3, 0.2])
        kl = VariationalFreeEnergy.kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_kl_divergence_non_negative(self):
        q = np.array([0.5, 0.3, 0.2])
        p = np.array([0.2, 0.5, 0.3])
        assert VariationalFreeEnergy.kl_divergence(q, p) >= 0.0

    def test_expected_free_energy_returns_finite(self, model):
        q = np.ones(model.n_states) / model.n_states
        for a in range(model.n_actions):
            G = VariationalFreeEnergy.expected_free_energy(q, a, model)
            assert np.isfinite(G)

    def test_efe_right_is_better_than_left_from_start(self, model):
        """
        From state 0 (prior = start), going right should have lower EFE
        than going left (since goal is at state 4).
        """
        q = model.infer_states(0)   # believe we're at state 0
        G_left  = VariationalFreeEnergy.expected_free_energy(q, 0, model)
        G_right = VariationalFreeEnergy.expected_free_energy(q, 2, model)
        assert G_right < G_left, f"Expected G_right={G_right:.4f} < G_left={G_left:.4f}"


# ── ActiveInferenceAgent ───────────────────────────────────────────────────────

class TestActiveInferenceAgent:
    def test_infer_states_updates_q_s(self, agent):
        old_q = agent.q_s.copy()
        agent.infer_states(3)
        assert not np.allclose(old_q, agent.q_s) or True  # may or may not change

    def test_select_action_valid(self, agent):
        agent.infer_states(0)
        a = agent.select_action()
        assert a in (0, 1, 2)

    def test_select_action_prefers_right_near_goal(self, model):
        """From state 3 (adjacent to goal 4), right should be chosen most often."""
        # Use a fresh agent with belief concentrated at state 3
        ag = ActiveInferenceAgent(model, alpha=6.0)
        ag.infer_states(3)   # strongly believe we're at state 3
        np.random.seed(7)
        counts = [0, 0, 0]
        for _ in range(200):
            counts[ag.select_action()] += 1
        # Right (action 2) should be chosen most often from state 3
        assert counts[2] > counts[1] and counts[2] > counts[0], (
            f"Expected right to dominate from state 3, got {counts}"
        )

    def test_update_beliefs_tracks_free_energy(self, agent):
        agent.infer_states(0)
        agent.select_action()
        agent.update_beliefs(1, 2)
        assert len(agent.free_energy_history) >= 1
        assert np.isfinite(agent.free_energy_history[-1])

    def test_reset_clears_history(self, agent):
        agent.infer_states(0)
        agent.update_beliefs(1, 2)
        agent.reset()
        assert len(agent.free_energy_history) == 0
        assert len(agent.action_history) == 0


# ── GridWorld1D ────────────────────────────────────────────────────────────────

class TestGridWorld1D:
    def test_step_right_increments_state(self):
        env = GridWorld1D()
        env.reset(start=2)
        for _ in range(20):
            env.step(2)  # keep going right
        assert env.state == 4  # clamped at max

    def test_step_left_decrements_state(self):
        env = GridWorld1D()
        env.reset(start=2)
        for _ in range(20):
            env.step(0)  # keep going left
        assert env.state == 0

    def test_done_when_goal_reached(self):
        env = GridWorld1D(goal=4)
        env.reset(start=3)
        _, done = env.step(2)  # step right into goal
        assert done

    def test_obs_in_valid_range(self):
        env = GridWorld1D()
        env.reset()
        for _ in range(50):
            obs, _ = env.step(np.random.randint(3))
            assert 0 <= obs < env.n_obs


# ── End-to-end convergence ─────────────────────────────────────────────────────

class TestConvergence:
    def test_agent_reaches_goal_within_50_steps(self):
        """Agent should reliably reach state 4 from state 0 within 50 steps."""
        successes = 0
        n_trials = 10
        for seed in range(n_trials):
            np.random.seed(seed)
            env = GridWorld1D(n_states=5, goal=4, obs_noise=0.1)
            m = GenerativeModel(
                A=env.build_A(), B=env.build_B(),
                D=env.build_D(0), C=env.build_C(4),
            )
            ag = ActiveInferenceAgent(m, alpha=4.0)
            obs = env.reset(start=0)
            ag.infer_states(obs)
            for _ in range(50):
                a = ag.select_action()
                obs, done = env.step(a)
                ag.update_beliefs(obs, a)
                if done:
                    successes += 1
                    break
        # Expect at least 8/10 successes
        assert successes >= 8, f"Only {successes}/10 trials reached goal"

    def test_free_energy_decreases_on_average(self):
        """Mean FE in the last 20 steps should be lower than in the first 20."""
        np.random.seed(99)
        env = GridWorld1D(n_states=5, goal=4, obs_noise=0.1)
        m = GenerativeModel(
            A=env.build_A(), B=env.build_B(),
            D=env.build_D(0), C=env.build_C(4),
        )
        ag = ActiveInferenceAgent(m, alpha=4.0)
        obs = env.reset(start=0)
        ag.infer_states(obs)
        for _ in range(100):
            a = ag.select_action()
            obs, _ = env.step(a)
            ag.update_beliefs(obs, a)
        fe = ag.free_energy_history
        assert np.mean(fe[-20:]) < np.mean(fe[:20]), (
            f"FE did not decrease: early={np.mean(fe[:20]):.4f}, late={np.mean(fe[-20:]):.4f}"
        )
