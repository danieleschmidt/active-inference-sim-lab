"""
Example unit tests for active-inference-sim-lab.

This file demonstrates testing patterns and provides basic test examples.
"""

import numpy as np
import pytest
import torch


class TestExampleUnitTests:
    """Example unit test class."""
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        result = 2 + 2
        assert result == 4
    
    def test_numpy_operations(self, sample_observations):
        """Test with numpy arrays."""
        assert sample_observations.shape == (10, 4)
        assert sample_observations.dtype == np.float64
        
        # Test basic operations
        mean_obs = np.mean(sample_observations, axis=0)
        assert mean_obs.shape == (4,)
    
    def test_torch_operations(self, device):
        """Test with PyTorch tensors."""
        x = torch.randn(5, 3, device=device)
        y = torch.randn(5, 3, device=device)
        
        result = x + y
        assert result.shape == (5, 3)
        assert result.device == device
    
    @pytest.mark.parametrize("input_size,expected_output", [
        (4, 8),
        (6, 12),
        (10, 20),
    ])
    def test_parametrized(self, input_size, expected_output):
        """Test with parameters."""
        result = input_size * 2
        assert result == expected_output
    
    def test_with_mock_environment(self, mock_environment):
        """Test using mock environment."""
        obs = mock_environment.reset()
        assert obs.shape == (4,)
        
        action = np.random.randn(2)
        obs, reward, done, info = mock_environment.step(action)
        
        assert obs.shape == (4,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_exception_handling(self):
        """Test exception handling."""
        with pytest.raises(ValueError):
            raise ValueError("Test exception")
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test marked as slow."""
        # Simulate slow operation
        result = sum(i**2 for i in range(1000))
        assert result > 0


class TestMathUtils:
    """Test mathematical utility functions."""
    
    def test_matrix_operations(self):
        """Test matrix operations."""
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 3)
        
        C = np.dot(A, B)
        assert C.shape == (3, 3)
    
    def test_probability_operations(self):
        """Test probability-related operations."""
        # Test normalization
        x = np.random.rand(10)
        normalized = x / np.sum(x)
        
        assert np.allclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
    
    def test_kl_divergence_properties(self):
        """Test KL divergence properties."""
        # Mock KL divergence function
        def kl_divergence(p, q):
            return np.sum(p * np.log(p / q))
        
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        kl = kl_divergence(p, q)
        assert kl >= 0  # KL divergence is non-negative


class TestEnvironmentInterface:
    """Test environment interface."""
    
    def test_environment_reset(self, mock_environment):
        """Test environment reset functionality."""
        obs1 = mock_environment.reset()
        obs2 = mock_environment.reset()
        
        assert obs1.shape == obs2.shape
        # Reset should potentially give different observations
        # (though they might be the same due to randomness)
    
    def test_environment_step(self, mock_environment):
        """Test environment step functionality."""
        mock_environment.reset()
        
        # Test multiple steps
        for _ in range(5):
            action = np.random.randn(2)
            obs, reward, done, info = mock_environment.step(action)
            
            assert obs.shape == (4,)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert "step" in info
    
    def test_environment_episode(self, mock_environment):
        """Test full episode."""
        obs = mock_environment.reset()
        total_reward = 0
        steps = 0
        
        while steps < 10:  # Limit steps for test
            action = np.random.randn(2)
            obs, reward, done, info = mock_environment.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        assert steps > 0
        assert isinstance(total_reward, (int, float))