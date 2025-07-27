"""Global pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_observations() -> np.ndarray:
    """Generate sample observation data for testing."""
    return np.random.randn(100, 4)


@pytest.fixture
def sample_actions() -> np.ndarray:
    """Generate sample action data for testing."""
    return np.random.randn(100, 2)


@pytest.fixture
def sample_beliefs() -> np.ndarray:
    """Generate sample belief state data for testing."""
    return np.random.randn(100, 8)


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""

    class MockEnvironment:
        def __init__(self):
            self.state_dim = 4
            self.action_dim = 2
            self.observation_dim = 4
            self.current_state = np.zeros(self.state_dim)
            self.step_count = 0

        def reset(self):
            self.current_state = np.random.randn(self.state_dim)
            self.step_count = 0
            return self.current_state.copy()

        def step(self, action):
            self.current_state += action + 0.1 * np.random.randn(self.state_dim)
            self.step_count += 1
            reward = -np.sum(self.current_state**2)
            done = self.step_count >= 100
            return self.current_state.copy(), reward, done, {}

        def render(self):
            pass

    return MockEnvironment()


@pytest.fixture
def simple_generative_model():
    """Create a simple generative model for testing."""

    class SimpleModel:
        def __init__(self, state_dim=4, obs_dim=4, action_dim=2):
            self.state_dim = state_dim
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            # Simple linear dynamics
            self.A = np.eye(state_dim) + 0.1 * np.random.randn(state_dim, state_dim)
            self.B = 0.1 * np.random.randn(state_dim, action_dim)
            self.C = np.eye(obs_dim, state_dim)
            self.Q = 0.01 * np.eye(state_dim)  # Process noise
            self.R = 0.01 * np.eye(obs_dim)  # Observation noise

        def predict_next_state(self, state, action):
            return self.A @ state + self.B @ action + np.random.multivariate_normal(
                np.zeros(self.state_dim), self.Q
            )

        def predict_observation(self, state):
            return self.C @ state + np.random.multivariate_normal(
                np.zeros(self.obs_dim), self.R
            )

        def log_likelihood(self, observation, state):
            predicted_obs = self.C @ state
            diff = observation - predicted_obs
            return -0.5 * (diff.T @ np.linalg.inv(self.R) @ diff)

    return SimpleModel()


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "num_episodes": 10,
        "max_steps_per_episode": 100,
        "timeout_seconds": 60,
        "performance_threshold": 0.8,
        "memory_limit_mb": 100,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "mujoco: mark test as requiring MuJoCo")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests in performance directory
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark MuJoCo tests
        if "mujoco" in item.name.lower():
            item.add_marker(pytest.mark.mujoco)