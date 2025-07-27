"""Integration tests for environment interactions."""

import numpy as np
import pytest


@pytest.mark.integration
class TestEnvironmentIntegration:
    """Test integration between agents and environments."""

    def test_basic_environment_interaction(self, mock_environment):
        """Test basic agent-environment interaction loop."""
        env = mock_environment

        # Test reset
        initial_obs = env.reset()
        assert initial_obs.shape == (4,)
        assert env.step_count == 0

        # Test multiple steps
        total_reward = 0
        for step in range(10):
            action = np.random.randn(2)
            obs, reward, done, info = env.step(action)

            assert obs.shape == (4,)
            assert isinstance(reward, (float, np.floating))
            assert isinstance(done, bool)
            assert isinstance(info, dict)

            total_reward += reward

            if done:
                break

        assert env.step_count > 0

    def test_episode_completion(self, mock_environment):
        """Test complete episode from start to finish."""
        env = mock_environment
        episode_length = 0
        episode_reward = 0

        obs = env.reset()
        done = False

        while not done and episode_length < 200:  # Safety limit
            action = np.random.randn(2)
            obs, reward, done, info = env.step(action)
            episode_length += 1
            episode_reward += reward

        assert episode_length > 0
        assert episode_length <= 100  # Should terminate at max steps

    def test_multiple_episodes(self, mock_environment):
        """Test running multiple episodes sequentially."""
        env = mock_environment
        episode_rewards = []

        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 200:
                action = np.random.randn(2)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)

        assert len(episode_rewards) == 5
        assert all(isinstance(r, (float, np.floating)) for r in episode_rewards)

    def test_environment_state_consistency(self, mock_environment):
        """Test that environment state remains consistent."""
        env = mock_environment

        # Test that reset changes state
        state1 = env.reset()
        state2 = env.reset()

        # States should be different (very low probability of being identical)
        assert not np.allclose(state1, state2, atol=1e-10)

        # Test that step changes state
        initial_state = env.reset()
        action = np.array([1.0, 0.0])
        new_state, _, _, _ = env.step(action)

        assert not np.allclose(initial_state, new_state)

    @pytest.mark.slow
    def test_long_episode_stability(self, mock_environment):
        """Test stability over long episodes."""
        env = mock_environment
        obs = env.reset()

        # Run for many steps to test numerical stability
        for step in range(1000):
            action = 0.01 * np.random.randn(2)  # Small actions
            obs, reward, done, info = env.step(action)

            # Check for numerical issues
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)

            if done:
                obs = env.reset()

        # Should complete without numerical errors

    def test_action_space_boundaries(self, mock_environment):
        """Test environment behavior with extreme actions."""
        env = mock_environment
        obs = env.reset()

        # Test with large positive actions
        large_action = np.array([100.0, 100.0])
        obs, reward, done, info = env.step(large_action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)

        # Test with large negative actions
        obs = env.reset()
        large_negative_action = np.array([-100.0, -100.0])
        obs, reward, done, info = env.step(large_negative_action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)

        # Test with zero actions
        obs = env.reset()
        zero_action = np.array([0.0, 0.0])
        obs, reward, done, info = env.step(zero_action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)