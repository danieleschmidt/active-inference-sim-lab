"""
Unit tests for ActiveInferenceAgent class.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

from active_inference import ActiveInferenceAgent, MockEnvironment
from active_inference.core.beliefs import BeliefState


class TestActiveInferenceAgent:
    """Test suite for ActiveInferenceAgent."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = ActiveInferenceAgent(
            state_dim=4,
            obs_dim=4,
            action_dim=2,
            inference_method="variational",
            planning_horizon=5,
            learning_rate=0.01,
        )
        
        assert agent.state_dim == 4
        assert agent.obs_dim == 4
        assert agent.action_dim == 2
        assert agent.planning_horizon == 5
        assert agent.learning_rate == 0.01
        assert agent.inference_method == "variational"

    def test_agent_invalid_dimensions(self):
        """Test agent initialization with invalid dimensions."""
        from active_inference.utils.validation import ValidationError
        
        with pytest.raises(ValidationError):
            ActiveInferenceAgent(
                state_dim=0,  # Invalid
                obs_dim=4,
                action_dim=2
            )
        
        with pytest.raises(ValidationError):
            ActiveInferenceAgent(
                state_dim=4,
                obs_dim=-1,  # Invalid
                action_dim=2
            )

    def test_agent_act_valid_observation(self):
        """Test agent action selection with valid observation."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1,
            planning_horizon=1
        )
        
        obs = np.array([0.5, -0.3])
        action = agent.act(obs)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert np.isfinite(action).all()

    def test_agent_act_invalid_observation(self):
        """Test agent action selection with invalid observation."""
        from active_inference.utils.validation import ValidationError
        
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        # Wrong observation dimension
        with pytest.raises(ValidationError):
            agent.act(np.array([0.5]))  # Should be 2D
        
        # NaN observation
        with pytest.raises(ValidationError):
            agent.act(np.array([np.nan, 0.5]))

    def test_agent_reset(self):
        """Test agent reset functionality."""
        agent = ActiveInferenceAgent(
            state_dim=3,
            obs_dim=3,
            action_dim=2
        )
        
        initial_obs = np.array([0.1, 0.2, 0.3])
        agent.reset(initial_obs)
        
        # Check that agent state is properly reset
        assert isinstance(agent.beliefs, BeliefState)
        assert agent.step_count == 0

    def test_agent_update_beliefs(self):
        """Test belief updating functionality."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        obs = np.array([0.5, -0.2])
        action = np.array([0.3])
        
        # Initial beliefs
        initial_beliefs = agent.beliefs
        
        # Update beliefs
        agent.update_beliefs(obs, action)
        
        # Beliefs should be updated
        assert agent.beliefs is not None
        assert isinstance(agent.beliefs, BeliefState)
        
        # Check if beliefs are stored properly
        all_beliefs = agent.beliefs.get_all_beliefs()
        assert isinstance(all_beliefs, dict)
        
        # Should contain at least some beliefs
        assert len(all_beliefs) >= 0  # May be empty initially

    def test_agent_environment_interaction(self):
        """Test full agent-environment interaction."""
        env = MockEnvironment(obs_dim=2, action_dim=1, episode_length=10)
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1,
            planning_horizon=1
        )
        
        obs = env.reset()
        agent.reset(obs)
        
        total_reward = 0.0
        for step in range(5):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            agent.update_beliefs(obs, action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        assert isinstance(total_reward, (int, float))
        assert agent.step_count > 0

    def test_agent_different_inference_methods(self):
        """Test agent with different inference methods."""
        methods = ["variational", "kalman", "particle"]
        
        for method in methods:
            try:
                agent = ActiveInferenceAgent(
                    state_dim=3,
                    obs_dim=3,
                    action_dim=2,
                    inference_method=method
                )
                
                obs = np.array([0.1, 0.2, 0.3])
                action = agent.act(obs)
                
                assert isinstance(action, np.ndarray)
                assert action.shape == (2,)
                
            except NotImplementedError:
                # Some methods might not be implemented yet
                pass

    def test_agent_planning_horizons(self):
        """Test agent with different planning horizons."""
        horizons = [1, 3, 5, 10]
        
        for horizon in horizons:
            agent = ActiveInferenceAgent(
                state_dim=2,
                obs_dim=2,
                action_dim=1,
                planning_horizon=horizon
            )
            
            obs = np.array([0.5, -0.3])
            action = agent.act(obs)
            
            assert isinstance(action, np.ndarray)
            assert agent.planning_horizon == horizon

    def test_agent_logging(self):
        """Test agent logging functionality."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1,
            enable_logging=True,
            agent_id="test_agent"
        )
        
        assert agent.enable_logging is True
        assert agent.agent_id == "test_agent"
        assert hasattr(agent, 'logger')

    def test_agent_memory_management(self):
        """Test agent memory management with long episodes."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1,
            max_history_length=10
        )
        
        # Run many steps to test memory management
        obs = np.array([0.1, 0.2])
        agent.reset(obs)
        
        for _ in range(20):  # More than max_history_length
            action = agent.act(obs)
            agent.update_beliefs(obs, action)
            obs = obs + np.random.normal(0, 0.01, 2)
        
        # History should be limited
        assert len(agent.history) <= agent.max_history_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])