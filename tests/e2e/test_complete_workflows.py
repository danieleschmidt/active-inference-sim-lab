"""End-to-end tests for complete workflows."""

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteWorkflows:
    """End-to-end tests for complete active inference workflows."""

    def test_complete_training_workflow(
        self, mock_environment, simple_generative_model, temp_dir
    ):
        """Test a complete training workflow from start to finish."""
        env = mock_environment
        model = simple_generative_model

        # Training configuration
        num_episodes = 5
        max_steps = 50

        # Track training progress
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            # Episode loop
            while not done and episode_length < max_steps:
                # Simple belief update (placeholder)
                belief_mean = obs
                belief_cov = 0.1 * np.eye(len(obs))

                # Simple action selection
                action = -0.1 * belief_mean[:2] + 0.05 * np.random.randn(2)

                # Environment step
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                # Simple model learning (placeholder)
                # In real implementation, this would update the generative model

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Verify training completed successfully
        assert len(episode_rewards) == num_episodes
        assert len(episode_lengths) == num_episodes
        assert all(length > 0 for length in episode_lengths)

        # Should show some learning (this is a simple check)
        # In practice, you'd have more sophisticated learning metrics
        assert np.std(episode_rewards) < 1000  # Rewards shouldn't be wildly varying

    def test_inference_and_planning_pipeline(self, simple_generative_model):
        """Test the complete inference and planning pipeline."""
        model = simple_generative_model

        # Initial belief
        belief_mean = np.zeros(model.state_dim)
        belief_cov = np.eye(model.state_dim)

        # Planning horizon
        horizon = 5
        num_trajectories = 10

        # Sample planning trajectories
        trajectory_values = []

        for traj in range(num_trajectories):
            # Sample initial action sequence
            actions = [np.random.randn(model.action_dim) for _ in range(horizon)]

            # Simulate trajectory
            state = belief_mean.copy()
            trajectory_value = 0

            for t, action in enumerate(actions):
                # Predict next state
                next_state = model.predict_next_state(state, action)

                # Predict observation
                pred_obs = model.predict_observation(next_state)

                # Simple value function (negative quadratic cost)
                cost = np.sum(next_state**2) + 0.1 * np.sum(action**2)
                trajectory_value -= cost

                state = next_state

            trajectory_values.append(trajectory_value)

        # Select best trajectory
        best_trajectory_idx = np.argmax(trajectory_values)
        best_value = trajectory_values[best_trajectory_idx]

        # Verify planning worked
        assert len(trajectory_values) == num_trajectories
        assert all(np.isfinite(val) for val in trajectory_values)
        assert best_trajectory_idx >= 0

    def test_multi_environment_workflow(self, mock_environment):
        """Test workflow across multiple environment instances."""
        # Create multiple environment instances
        environments = [mock_environment for _ in range(3)]

        # Run parallel episodes
        all_results = []

        for env_idx, env in enumerate(environments):
            obs = env.reset()
            episode_results = {
                "env_id": env_idx,
                "rewards": [],
                "states": [],
                "actions": [],
            }

            for step in range(20):
                # Simple policy
                action = -0.1 * obs[:2] + 0.1 * np.random.randn(2)
                next_obs, reward, done, info = env.step(action)

                episode_results["rewards"].append(reward)
                episode_results["states"].append(obs.copy())
                episode_results["actions"].append(action.copy())

                obs = next_obs
                if done:
                    break

            all_results.append(episode_results)

        # Verify all environments ran successfully
        assert len(all_results) == 3
        for result in all_results:
            assert len(result["rewards"]) > 0
            assert len(result["states"]) == len(result["actions"])

    def test_model_learning_workflow(self, mock_environment, simple_generative_model):
        """Test model learning from interaction data."""
        env = mock_environment
        model = simple_generative_model

        # Collect interaction data
        transitions = []
        obs = env.reset()

        for step in range(50):
            action = np.random.randn(2)
            next_obs, reward, done, info = env.step(action)

            transitions.append({
                "state": obs.copy(),
                "action": action.copy(),
                "next_state": next_obs.copy(),
                "reward": reward,
            })

            obs = next_obs
            if done:
                obs = env.reset()

        # Simulate model learning (placeholder)
        # In real implementation, this would update model parameters
        states = np.array([t["state"] for t in transitions])
        actions = np.array([t["action"] for t in transitions])
        next_states = np.array([t["next_state"] for t in transitions])

        # Simple model update: compute empirical dynamics
        state_diffs = next_states - states
        action_effects = np.linalg.lstsq(actions, state_diffs, rcond=None)[0]

        # Verify model learning worked
        assert action_effects.shape == (2, 4)  # action_dim x state_dim
        assert np.all(np.isfinite(action_effects))

        # Test model predictions
        test_state = np.random.randn(4)
        test_action = np.random.randn(2)
        predicted_next_state = model.predict_next_state(test_state, test_action)

        assert predicted_next_state.shape == (4,)
        assert np.all(np.isfinite(predicted_next_state))

    def test_evaluation_workflow(self, mock_environment):
        """Test model evaluation workflow."""
        env = mock_environment

        # Evaluation configuration
        num_eval_episodes = 3
        eval_results = []

        for episode in range(num_eval_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_states = []

            done = False
            while not done and episode_length < 100:
                # Evaluation policy (deterministic)
                action = -0.1 * obs[:2]  # Simple proportional controller

                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                episode_states.append(obs.copy())

                obs = next_obs

            eval_results.append({
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "final_state": obs.copy(),
                "trajectory": episode_states,
            })

        # Verify evaluation completed
        assert len(eval_results) == num_eval_episodes
        for result in eval_results:
            assert result["length"] > 0
            assert np.isfinite(result["reward"])
            assert len(result["trajectory"]) == result["length"]

        # Compute evaluation metrics
        mean_reward = np.mean([r["reward"] for r in eval_results])
        mean_length = np.mean([r["length"] for r in eval_results])
        reward_std = np.std([r["reward"] for r in eval_results])

        assert np.isfinite(mean_reward)
        assert np.isfinite(mean_length)
        assert reward_std >= 0

    @pytest.mark.slow
    def test_long_running_workflow(self, mock_environment):
        """Test stability over long-running workflows."""
        env = mock_environment

        total_steps = 0
        num_resets = 0
        cumulative_reward = 0

        obs = env.reset()
        num_resets += 1

        # Run for extended period
        for step in range(1000):
            action = np.random.randn(2)
            obs, reward, done, info = env.step(action)

            total_steps += 1
            cumulative_reward += reward

            if done:
                obs = env.reset()
                num_resets += 1

            # Check for numerical issues periodically
            if step % 100 == 0:
                assert np.all(np.isfinite(obs))
                assert np.isfinite(reward)
                assert np.isfinite(cumulative_reward)

        # Verify long run stability
        assert total_steps == 1000
        assert num_resets > 0  # Should have had some episode terminations
        assert np.isfinite(cumulative_reward)