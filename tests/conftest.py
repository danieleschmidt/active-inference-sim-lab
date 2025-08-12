"""
Pytest configuration and fixtures for active-inference-sim-lab tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

# Set random seeds for reproducibility
np.random.seed(42)

# Optional torch import
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
except ImportError:
    torch = None


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture providing path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_observations() -> np.ndarray:
    """Fixture providing sample observation data."""
    return np.random.randn(10, 4)


@pytest.fixture
def sample_actions() -> np.ndarray:
    """Fixture providing sample action data."""
    return np.random.randn(10, 2)


@pytest.fixture
def sample_beliefs() -> np.ndarray:
    """Fixture providing sample belief states."""
    return np.random.randn(10, 6)


@pytest.fixture
def mock_environment():
    """Fixture providing mock environment for testing."""
    
    class MockEnvironment:
        def __init__(self):
            self.observation_space_dim = 4
            self.action_space_dim = 2
            self.action_dim = 2  # Add action_dim attribute
            self.state_dim = 6
            self._state = np.zeros(self.state_dim)
            self._step_count = 0
            
        def reset(self):
            self._state = np.random.randn(self.state_dim)
            self._step_count = 0
            return self.get_observation()
            
        def step(self, action):
            # Ensure action has correct dimensionality
            action = np.atleast_1d(action)
            if len(action) != self.action_dim:
                action = action[:self.action_dim] if len(action) > self.action_dim else np.pad(action, (0, self.action_dim - len(action)))
            
            self._state += 0.1 * action + 0.05 * np.random.randn(self.state_dim)
            self._step_count += 1
            observation = self.get_observation()
            reward = -np.sum(self._state**2)  # Simple quadratic cost
            done = self._step_count >= 100
            info = {"step": self._step_count}
            return observation, reward, done, info
            
        def get_observation(self):
            # Partial observability
            noise = 0.1 * np.random.randn(self.observation_space_dim)
            return self._state[:self.observation_space_dim] + noise
    
    return MockEnvironment()


@pytest.fixture
def device():
    """Fixture providing appropriate device for testing."""
    if torch is not None:
        if torch.cuda.is_available() and not os.getenv("PYTEST_CURRENT_TEST"):
            return torch.device("cuda")
        return torch.device("cpu")
    else:
        return "cpu"  # String fallback when torch not available


@pytest.fixture(autouse=True)
def set_test_environment():
    """Automatically set environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    os.environ.update({
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
        "CUDA_VISIBLE_DEVICES": "",  # Disable CUDA by default in tests
        "WANDB_MODE": "disabled",
        "LOG_LEVEL": "WARNING",
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "mujoco: mark test as requiring MuJoCo"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests with certain patterns
        if any(keyword in item.nodeid for keyword in ["train", "optimize", "benchmark"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def anyio_backend():
    """Backend for anyio async tests."""
    return "asyncio"


class TestConfig:
    """Test configuration constants."""
    
    TOLERANCE = 1e-6
    RANDOM_SEED = 42
    MAX_ITERATIONS = 1000
    DEFAULT_BATCH_SIZE = 32
    SMALL_DATASET_SIZE = 100
    MEDIUM_DATASET_SIZE = 1000


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Fixture providing test configuration."""
    return TestConfig()


# Pytest plugins  
# pytest_plugins = [
#     "pytest_benchmark",
#     "pytest_mock", 
#     "pytest_asyncio",
# ]