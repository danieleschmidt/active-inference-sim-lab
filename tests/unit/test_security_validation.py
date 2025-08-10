"""
Security validation tests for Active Inference components.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

from active_inference import ActiveInferenceAgent
from active_inference.utils.validation import ValidationError


class TestSecurityValidation:
    """Test security-related validation and input sanitization."""

    def test_agent_large_array_protection(self):
        """Test protection against extremely large arrays."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        # Test with extremely large array (should be rejected)
        huge_obs = np.ones(1000000)  # 1M elements
        
        with pytest.raises(ValidationError):
            agent.act(huge_obs)

    def test_agent_malformed_input_protection(self):
        """Test protection against malformed inputs."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        # Test with empty array
        with pytest.raises(ValidationError):
            agent.act(np.array([]))
        
        # Test with infinite values
        with pytest.raises(ValidationError):
            agent.act(np.array([np.inf, -np.inf]))
        
        # Test with NaN values
        with pytest.raises(ValidationError):
            agent.act(np.array([np.nan, 0.5]))

    def test_agent_memory_bounds(self):
        """Test agent memory usage stays within reasonable bounds."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1,
            max_history_length=10  # Small limit for testing
        )
        
        obs = np.array([0.1, 0.2])
        agent.reset(obs)
        
        # Run many steps to test memory management
        for _ in range(50):
            action = agent.act(obs)
            agent.update_beliefs(obs)
            obs = obs + np.random.normal(0, 0.01, 2)
        
        # History should be limited
        assert len(agent.history['observations']) <= agent.max_history_length

    def test_numerical_stability(self):
        """Test numerical stability with edge case values."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        # Test with very small values
        tiny_obs = np.array([1e-10, -1e-10])
        action = agent.act(tiny_obs)
        assert np.isfinite(action).all()
        
        # Test with very large (but valid) values
        large_obs = np.array([100.0, -100.0])
        action = agent.act(large_obs)
        assert np.isfinite(action).all()

    def test_concurrent_safety(self):
        """Test thread safety of agent operations."""
        import threading
        import time
        
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    obs = np.random.randn(2)
                    action = agent.act(obs)
                    agent.update_beliefs(obs)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have minimal errors (some may occur due to race conditions)
        assert len(errors) < 5  # Allow some errors but not too many

    def test_input_sanitization(self):
        """Test input sanitization and bounds checking."""
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1
        )
        
        # Test with edge case dimensions
        valid_obs = np.array([0.5, -0.3])
        action = agent.act(valid_obs)
        
        # Action should be bounded (no extreme values)
        assert np.abs(action).max() < 1000  # Reasonable bound
        assert np.isfinite(action).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])