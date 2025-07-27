"""Sample unit tests for the active inference simulation lab."""

import numpy as np
import pytest


class TestSampleFunctionality:
    """Sample test class to demonstrate testing structure."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert 2 + 2 == 4
        assert 3 * 4 == 12

    def test_numpy_operations(self, sample_observations):
        """Test numpy operations with sample data."""
        assert sample_observations.shape == (100, 4)
        assert isinstance(sample_observations, np.ndarray)

        # Test basic statistics
        mean = np.mean(sample_observations, axis=0)
        assert len(mean) == 4

        # Test variance
        var = np.var(sample_observations, axis=0)
        assert all(var >= 0)

    def test_belief_state_properties(self, sample_beliefs):
        """Test properties of belief states."""
        assert sample_beliefs.shape == (100, 8)
        assert np.all(np.isfinite(sample_beliefs))

    def test_action_dimensions(self, sample_actions):
        """Test action array properties."""
        assert sample_actions.shape == (100, 2)
        assert sample_actions.dtype == np.float64

    def test_generative_model_interface(self, simple_generative_model):
        """Test the generative model interface."""
        model = simple_generative_model

        # Test dimensions
        assert model.state_dim == 4
        assert model.obs_dim == 4
        assert model.action_dim == 2

        # Test prediction methods
        state = np.random.randn(4)
        action = np.random.randn(2)

        next_state = model.predict_next_state(state, action)
        assert next_state.shape == (4,)

        observation = model.predict_observation(state)
        assert observation.shape == (4,)

        # Test likelihood computation
        log_like = model.log_likelihood(observation, state)
        assert isinstance(log_like, (float, np.floating))
        assert np.isfinite(log_like)

    @pytest.mark.parametrize("state_dim", [2, 4, 8])
    @pytest.mark.parametrize("action_dim", [1, 2, 3])
    def test_different_dimensions(self, state_dim, action_dim):
        """Test functionality with different state and action dimensions."""
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)

        # Test that operations work with different dimensions
        next_state = state + 0.1 * action[:state_dim] if action_dim >= state_dim else state
        assert next_state.shape == (state_dim,)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            # This should raise an error for mismatched dimensions
            a = np.array([1, 2, 3])
            b = np.array([1, 2])
            np.dot(a, b)  # Incompatible dimensions

    def test_numerical_stability(self):
        """Test numerical stability of operations."""
        # Test with very small numbers
        small_array = np.array([1e-10, 1e-15, 1e-20])
        result = np.log(small_array + 1e-8)
        assert np.all(np.isfinite(result))

        # Test with large numbers
        large_array = np.array([1e10, 1e15, 1e20])
        result = np.exp(-large_array)
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    @pytest.mark.slow
    def test_performance_characteristics(self):
        """Test performance characteristics (marked as slow)."""
        import time

        # Test computational performance
        large_matrix = np.random.randn(1000, 1000)
        start_time = time.time()
        result = np.linalg.inv(large_matrix)
        computation_time = time.time() - start_time

        assert result.shape == (1000, 1000)
        assert computation_time < 10.0  # Should complete within 10 seconds

    def test_random_seed_reproducibility(self):
        """Test that random operations can be made reproducible."""
        np.random.seed(42)
        result1 = np.random.randn(10)

        np.random.seed(42)
        result2 = np.random.randn(10)

        np.testing.assert_array_equal(result1, result2)

    def test_memory_usage(self):
        """Test memory usage patterns."""
        # Create large arrays and ensure they can be garbage collected
        large_arrays = []
        for _ in range(10):
            arr = np.random.randn(100, 100)
            large_arrays.append(arr)

        # Clear references
        large_arrays.clear()

        # This test mainly ensures no memory leaks in simple operations
        assert True  # If we get here, no memory errors occurred