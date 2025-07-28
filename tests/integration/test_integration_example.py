"""
Example integration tests for active-inference-sim-lab.

Integration tests verify that different components work together correctly.
"""

import numpy as np
import pytest


@pytest.mark.integration
class TestAgentEnvironmentIntegration:
    """Test agent-environment integration."""
    
    def test_agent_environment_loop(self, mock_environment):
        """Test basic agent-environment interaction loop."""
        # Mock agent class
        class MockAgent:
            def __init__(self, obs_dim, action_dim):
                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.beliefs = np.zeros(obs_dim)
            
            def act(self, observation):
                # Simple policy: random action
                return np.random.randn(self.action_dim)
            
            def update(self, observation, action, reward):
                # Simple belief update
                self.beliefs = 0.9 * self.beliefs + 0.1 * observation
        
        agent = MockAgent(4, 2)
        env = mock_environment
        
        # Run episode
        obs = env.reset()
        total_reward = 0
        
        for step in range(10):
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward)
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        assert isinstance(total_reward, (int, float))
        assert agent.beliefs.shape == (4,)
    
    @pytest.mark.slow
    def test_training_loop_integration(self, mock_environment):
        """Test integration of training components."""
        # Mock training components
        class MockTrainer:
            def __init__(self, agent, env):
                self.agent = agent
                self.env = env
                self.metrics = []
            
            def train_episode(self):
                obs = self.env.reset()
                episode_reward = 0
                
                for _ in range(20):
                    action = self.agent.act(obs)
                    next_obs, reward, done, info = self.env.step(action)
                    self.agent.update(obs, action, reward)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                self.metrics.append(episode_reward)
                return episode_reward
        
        # Mock agent
        class MockLearningAgent:
            def __init__(self):
                self.policy_params = np.random.randn(4, 2)
                self.learning_rate = 0.01
            
            def act(self, observation):
                return np.dot(observation, self.policy_params)
            
            def update(self, observation, action, reward):
                # Simple gradient update (mock)
                gradient = np.outer(observation, action) * reward
                self.policy_params += self.learning_rate * gradient
        
        agent = MockLearningAgent()
        trainer = MockTrainer(agent, mock_environment)
        
        # Run training
        for episode in range(5):
            reward = trainer.train_episode()
            assert isinstance(reward, (int, float))
        
        assert len(trainer.metrics) == 5


@pytest.mark.integration
class TestModelInferenceIntegration:
    """Test model and inference integration."""
    
    def test_generative_model_inference(self):
        """Test integration between generative model and inference."""
        # Mock generative model
        class MockGenerativeModel:
            def __init__(self, state_dim, obs_dim):
                self.state_dim = state_dim
                self.obs_dim = obs_dim
                self.prior_mean = np.zeros(state_dim)
                self.prior_cov = np.eye(state_dim)
                self.likelihood_matrix = np.random.randn(obs_dim, state_dim)
                self.obs_noise = 0.1
            
            def sample_prior(self):
                return np.random.multivariate_normal(
                    self.prior_mean, self.prior_cov
                )
            
            def likelihood(self, observation, state):
                expected_obs = np.dot(self.likelihood_matrix, state)
                diff = observation - expected_obs
                return np.exp(-0.5 * np.sum(diff**2) / self.obs_noise**2)
        
        # Mock inference engine
        class MockInference:
            def __init__(self, model):
                self.model = model
            
            def infer_posterior(self, observation, num_samples=100):
                # Simple importance sampling
                samples = []
                weights = []
                
                for _ in range(num_samples):
                    state = self.model.sample_prior()
                    weight = self.model.likelihood(observation, state)
                    samples.append(state)
                    weights.append(weight)
                
                weights = np.array(weights)
                weights /= np.sum(weights)
                
                # Compute weighted mean as posterior estimate
                posterior_mean = np.average(samples, weights=weights, axis=0)
                return posterior_mean, samples, weights
        
        model = MockGenerativeModel(4, 3)
        inference = MockInference(model)
        
        # Test inference
        observation = np.random.randn(3)
        posterior_mean, samples, weights = inference.infer_posterior(observation)
        
        assert posterior_mean.shape == (4,)
        assert len(samples) == 100
        assert len(weights) == 100
        assert np.allclose(np.sum(weights), 1.0)
    
    def test_planning_inference_integration(self):
        """Test integration between planning and inference."""
        # Mock planner
        class MockPlanner:
            def __init__(self, horizon=5):
                self.horizon = horizon
            
            def plan(self, current_belief, goal_state):
                # Simple plan: move towards goal
                plan = []
                belief = current_belief.copy()
                
                for _ in range(self.horizon):
                    action = 0.1 * (goal_state - belief[:len(goal_state)])
                    plan.append(action)
                    # Simple dynamics
                    belief[:len(action)] += action
                
                return plan
        
        planner = MockPlanner()
        current_belief = np.random.randn(6)
        goal_state = np.array([1.0, 0.0])
        
        plan = planner.plan(current_belief, goal_state)
        
        assert len(plan) == 5
        assert all(isinstance(action, np.ndarray) for action in plan)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndIntegration:
    """Test end-to-end system integration."""
    
    def test_complete_active_inference_loop(self, mock_environment):
        """Test complete active inference agent loop."""
        # Simplified active inference agent
        class SimpleActiveInferenceAgent:
            def __init__(self, obs_dim, action_dim, state_dim):
                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.state_dim = state_dim
                
                # Simple generative model
                self.prior_beliefs = np.zeros(state_dim)
                self.beliefs = np.zeros(state_dim)
                self.observation_model = np.random.randn(obs_dim, state_dim)
                
            def perceive(self, observation):
                """Update beliefs based on observation."""
                # Simple belief update (mock Bayesian inference)
                prediction_error = observation - np.dot(
                    self.observation_model, self.beliefs
                )
                self.beliefs += 0.1 * np.dot(
                    self.observation_model.T, prediction_error
                )
                return self.beliefs
            
            def plan(self):
                """Plan action to minimize expected free energy."""
                # Simple planning: random action with bias towards zero
                return 0.1 * np.random.randn(self.action_dim)
            
            def act(self, observation):
                """Complete perception-action cycle."""
                self.perceive(observation)
                return self.plan()
        
        agent = SimpleActiveInferenceAgent(4, 2, 6)
        env = mock_environment
        
        # Run complete episodes
        episode_rewards = []
        
        for episode in range(3):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(20):
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        assert len(episode_rewards) == 3
        assert all(isinstance(r, (int, float)) for r in episode_rewards)
        
        # Check that agent beliefs evolved
        assert not np.allclose(agent.beliefs, agent.prior_beliefs)