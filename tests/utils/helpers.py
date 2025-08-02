"""Test helper functions and utilities."""

import contextlib
import tempfile
import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def create_test_agent(state_dim: int = 4, obs_dim: int = 8, action_dim: int = 2, **kwargs):
    """Create a test active inference agent with default parameters.
    
    Args:
        state_dim: Hidden state dimensions
        obs_dim: Observation dimensions
        action_dim: Action dimensions
        **kwargs: Additional agent parameters
        
    Returns:
        ActiveInferenceAgent: Configured test agent
    """
    from active_inference import ActiveInferenceAgent
    
    return ActiveInferenceAgent(
        state_dim=state_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        **kwargs
    )


def create_test_environment(env_type: str = "discrete", **kwargs):
    """Create a test environment for agent testing.
    
    Args:
        env_type: Type of environment ("discrete", "continuous", "grid")
        **kwargs: Additional environment parameters
        
    Returns:
        Environment: Configured test environment
    """
    from active_inference.envs import TestEnvironment
    
    return TestEnvironment(env_type=env_type, **kwargs)


def generate_test_observations(
    num_obs: int = 100,
    obs_dim: int = 8,
    noise_level: float = 0.1,
    random_seed: Optional[int] = 42
) -> np.ndarray:
    """Generate synthetic observation sequences for testing.
    
    Args:
        num_obs: Number of observations to generate
        obs_dim: Dimensionality of each observation
        noise_level: Amount of noise to add
        random_seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Generated observations [num_obs, obs_dim]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate base signal
    t = np.linspace(0, 10, num_obs)
    signal = np.sin(t[:, np.newaxis] * np.arange(1, obs_dim + 1))
    
    # Add noise
    noise = np.random.normal(0, noise_level, signal.shape)
    
    return signal + noise


def generate_test_beliefs(
    num_states: int = 4,
    num_timesteps: int = 50,
    random_seed: Optional[int] = 42
) -> np.ndarray:
    """Generate test belief sequences.
    
    Args:
        num_states: Number of hidden states
        num_timesteps: Number of timesteps
        random_seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Belief sequences [num_timesteps, num_states]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random walk beliefs that sum to 1
    beliefs = np.zeros((num_timesteps, num_states))
    beliefs[0] = np.random.dirichlet(np.ones(num_states))
    
    for t in range(1, num_timesteps):
        # Small random changes maintaining probability constraints
        change = np.random.normal(0, 0.01, num_states)
        new_beliefs = beliefs[t-1] + change
        new_beliefs = np.maximum(new_beliefs, 0.001)  # Avoid zeros
        beliefs[t] = new_beliefs / np.sum(new_beliefs)  # Normalize
    
    return beliefs


@contextlib.contextmanager
def temporary_config(config_dict: Dict[str, Any]):
    """Temporarily modify configuration for testing.
    
    Args:
        config_dict: Configuration values to set temporarily
        
    Yields:
        None
    """
    original_values = {}
    
    try:
        # Store original values and set new ones
        for key, value in config_dict.items():
            if hasattr(os.environ, key):
                original_values[key] = os.environ.get(key)
            else:
                original_values[key] = None
            os.environ[key] = str(value)
        
        yield
        
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]


@contextlib.contextmanager
def temporary_directory():
    """Create a temporary directory for testing.
    
    Yields:
        Path: Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def assert_beliefs_close(
    beliefs1: np.ndarray,
    beliefs2: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8
):
    """Assert that two belief arrays are numerically close.
    
    Args:
        beliefs1: First belief array
        beliefs2: Second belief array  
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Raises:
        AssertionError: If beliefs are not close enough
    """
    np.testing.assert_allclose(
        beliefs1, beliefs2, rtol=rtol, atol=atol,
        err_msg="Beliefs are not numerically close"
    )


def assert_free_energy_decreases(
    free_energies: List[float],
    tolerance: float = 1e-6
):
    """Assert that free energy generally decreases over time.
    
    Args:
        free_energies: List of free energy values over time
        tolerance: Tolerance for small increases due to numerical errors
        
    Raises:
        AssertionError: If free energy doesn't generally decrease
    """
    increases = []
    for i in range(1, len(free_energies)):
        if free_energies[i] > free_energies[i-1] + tolerance:
            increases.append((i, free_energies[i] - free_energies[i-1]))
    
    # Allow for small number of increases (local minima, numerical errors)
    max_allowed_increases = max(1, len(free_energies) // 10)
    
    assert len(increases) <= max_allowed_increases, (
        f"Free energy increased too many times: {len(increases)} > {max_allowed_increases}. "
        f"Increases: {increases}"
    )


def assert_probabilities_valid(
    probabilities: np.ndarray,
    axis: Optional[int] = -1,
    rtol: float = 1e-5
):
    """Assert that probabilities are valid (non-negative, sum to 1).
    
    Args:
        probabilities: Probability array
        axis: Axis along which probabilities should sum to 1
        rtol: Relative tolerance for sum check
        
    Raises:
        AssertionError: If probabilities are invalid
    """
    # Check non-negativity
    assert np.all(probabilities >= 0), "Probabilities must be non-negative"
    
    # Check sum to 1
    sums = np.sum(probabilities, axis=axis, keepdims=True)
    np.testing.assert_allclose(
        sums, 1.0, rtol=rtol,
        err_msg="Probabilities must sum to 1"
    )


def create_test_data_file(
    filepath: Union[str, Path],
    data: Union[np.ndarray, Dict[str, Any]],
    format: str = "numpy"
):
    """Create a test data file.
    
    Args:
        filepath: Path to save file
        data: Data to save
        format: File format ("numpy", "json", "csv")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "numpy":
        np.save(filepath, data)
    elif format == "json":
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == "csv":
        import pandas as pd
        pd.DataFrame(data).to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_test_data(filepath: Union[str, Path], format: str = "numpy"):
    """Load test data from file.
    
    Args:
        filepath: Path to data file
        format: File format ("numpy", "json", "csv")
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if format == "numpy":
        return np.load(filepath)
    elif format == "json":
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == "csv":
        import pandas as pd
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple[Any, float]: (result, execution_time_in_seconds)
    """
    import time
    
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    return result, execution_time


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import pytest
    
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")


def skip_if_no_mujoco():
    """Skip test if MuJoCo is not available."""
    import pytest
    
    try:
        import mujoco_py
    except ImportError:
        pytest.skip("MuJoCo not available")


def parametrize_environments():
    """Pytest parametrize decorator for different environments."""
    import pytest
    
    return pytest.mark.parametrize("env_name", [
        "discrete",
        "continuous", 
        "grid_world",
        "mountain_car"
    ])


def parametrize_agents():
    """Pytest parametrize decorator for different agent types."""
    import pytest
    
    return pytest.mark.parametrize("agent_type", [
        "basic",
        "hierarchical",
        "curiosity_driven"
    ])