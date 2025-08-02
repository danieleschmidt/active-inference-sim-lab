# Testing Guide - Active Inference Sim Lab

This directory contains comprehensive tests for the Active Inference Simulation Lab project.

## Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
├── contract/       # API contract tests
├── benchmark/      # Performance benchmarks
├── load/          # Load and stress tests
├── mutation/      # Mutation testing configuration
├── conftest.py    # Shared test fixtures and configuration
└── README.md      # This file
```

## Running Tests

### Quick Test Commands

```bash
# Run all tests
make test

# Run only unit tests
pytest tests/unit/ -v

# Run with coverage
pytest --cov=active_inference --cov-report=html

# Run specific test file
pytest tests/unit/test_example.py -v

# Run tests matching pattern
pytest -k "test_free_energy" -v
```

### Test Categories

#### Unit Tests (`tests/unit/`)
Fast, isolated tests for individual functions and classes.

```bash
# Run unit tests only
pytest tests/unit/ -v

# Run unit tests with coverage
pytest tests/unit/ --cov=active_inference --cov-report=term-missing
```

#### Integration Tests (`tests/integration/`)
Tests that verify component interactions and system behavior.

```bash
# Run integration tests
pytest tests/integration/ -v -m integration

# Run integration tests with specific environment
pytest tests/integration/ -v --env=test
```

#### Benchmark Tests (`tests/benchmark/`)
Performance tests and benchmarks.

```bash
# Run benchmarks
pytest tests/benchmark/ --benchmark-only

# Run benchmarks with comparison
pytest tests/benchmark/ --benchmark-compare=previous_results.json

# Generate benchmark report
pytest tests/benchmark/ --benchmark-json=benchmark_results.json
```

#### Contract Tests (`tests/contract/`)
API contract and interface tests.

```bash
# Run contract tests
pytest tests/contract/ -v

# Run contract tests with schema validation
pytest tests/contract/ --validate-schemas
```

#### Load Tests (`tests/load/`)
Load testing and stress testing using Locust.

```bash
# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Run headless load test
locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=60s --headless
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

Key features:
- Coverage reporting with 80% minimum threshold
- Multiple output formats (terminal, HTML, XML)
- Custom markers for test categorization
- Strict configuration validation

### Tox Configuration (`tox.ini`)

Testing environments:
- `py39`, `py310`, `py311`, `py312`: Multi-Python version testing
- `lint`: Code quality checks
- `security`: Security vulnerability scanning
- `docs`: Documentation building
- `coverage-report`: Coverage analysis
- `benchmark`: Performance benchmarking
- `mutation`: Mutation testing

### Test Markers

Use markers to categorize and selectively run tests:

```python
import pytest

@pytest.mark.slow
def test_expensive_computation():
    """Test that takes a long time to run."""
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test that requires GPU."""
    pass

@pytest.mark.integration
def test_system_integration():
    """Integration test."""
    pass
```

Run tests by marker:
```bash
# Skip slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run integration tests
pytest -m integration
```

## Writing Tests

### Unit Test Example

```python
import pytest
from active_inference.core import FreeEnergyCalculator

class TestFreeEnergyCalculator:
    """Unit tests for FreeEnergyCalculator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = FreeEnergyCalculator()
    
    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator is not None
        assert self.calculator.precision == "double"
    
    @pytest.mark.parametrize("input_dim,expected", [
        (2, 4),
        (4, 16),
        (8, 64),
    ])
    def test_compute_complexity(self, input_dim, expected):
        """Test complexity computation with parameters."""
        result = self.calculator.compute_complexity(input_dim)
        assert result == expected
```

### Integration Test Example

```python
import pytest
from active_inference import ActiveInferenceAgent
from active_inference.envs import TestEnvironment

@pytest.mark.integration
class TestAgentEnvironmentIntegration:
    """Integration tests for agent-environment interaction."""
    
    def test_agent_environment_interaction(self):
        """Test complete agent-environment loop."""
        env = TestEnvironment()
        agent = ActiveInferenceAgent.from_env(env)
        
        obs = env.reset()
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        
        assert next_obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
```

### Benchmark Test Example

```python
import pytest
from active_inference.core import BeliefUpdater

class TestBeliefUpdaterBenchmarks:
    """Benchmark tests for BeliefUpdater."""
    
    def setup_method(self):
        self.updater = BeliefUpdater()
    
    def test_belief_update_performance(self, benchmark):
        """Benchmark belief update performance."""
        observations = [1.0, 2.0, 3.0]
        result = benchmark(self.updater.update, observations)
        assert result is not None
```

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

Common test fixtures are defined in `conftest.py`:

```python
import pytest
from active_inference.envs import TestEnvironment

@pytest.fixture
def test_environment():
    """Provide a test environment."""
    return TestEnvironment()

@pytest.fixture
def sample_observations():
    """Provide sample observation data."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]
```

### Test Data Organization

```
tests/
├── data/
│   ├── fixtures/       # Test data files
│   ├── models/         # Test model files
│   └── environments/   # Test environment configurations
└── utils/
    ├── helpers.py      # Test utility functions
    └── mocks.py        # Mock objects
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Nightly schedule for comprehensive testing

### Test Matrix

- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Operating systems**: Ubuntu, macOS, Windows
- **Dependencies**: Minimum and latest versions

## Coverage Requirements

- **Minimum coverage**: 80%
- **Critical paths**: 95%+ coverage required
- **Coverage reports**: Available in HTML and XML formats

### Coverage Exclusions

Certain files/patterns are excluded from coverage:
- Test files themselves
- `__init__.py` files
- Debug and development utilities
- Third-party integrations

## Performance Testing

### Benchmark Thresholds

Performance regression detection:
- **Inference speed**: < 1ms for simple models
- **Memory usage**: < 100MB baseline
- **Throughput**: > 1000 inferences/second

### Profiling

```bash
# Profile memory usage
python -m memory_profiler examples/profile_example.py

# Profile CPU usage with py-spy
py-spy record -o profile.svg -- python examples/profile_example.py

# Generate line-by-line profiling
kernprof -l -v examples/profile_example.py
```

## Security Testing

### Automated Security Scans

```bash
# Run security checks
tox -e security

# Scan for vulnerabilities
bandit -r src/

# Check dependencies
safety check
pip-audit
```

## Mutation Testing

Test the quality of your tests with mutation testing:

```bash
# Run mutation testing
tox -e mutation

# Generate mutation report
mutmut html
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes `src/`
2. **Missing Dependencies**: Run `pip install -r requirements-dev.txt`
3. **Permission Errors**: Check file permissions for test data
4. **GPU Tests Failing**: Ensure CUDA is available or skip with `-m "not gpu"`

### Debug Mode

Run tests in debug mode:
```bash
# Enable debug output
pytest -v -s --log-cli-level=DEBUG

# Drop into debugger on failure
pytest --pdb

# Debug specific test
pytest tests/unit/test_example.py::test_function -v -s --pdb
```

### Test Isolation

Ensure tests don't interfere:
```bash
# Run tests in random order
pytest --random-order

# Run tests in parallel
pytest -n auto

# Clear cache between runs
pytest --cache-clear
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Fixtures**: Use fixtures for setup/teardown and shared data
4. **Mocking**: Mock external dependencies and slow operations
5. **Assertions**: Use specific assertions with helpful error messages
6. **Documentation**: Document complex test scenarios
7. **Performance**: Keep unit tests fast (< 1 second each)
8. **Isolation**: Ensure tests can run independently

## Contributing Tests

When contributing tests:

1. Follow existing patterns and conventions
2. Include both positive and negative test cases
3. Test edge cases and error conditions
4. Maintain or improve coverage percentage
5. Add appropriate markers for test categorization
6. Update documentation for new test utilities

---

For more information, see:
- [Contributing Guide](../CONTRIBUTING.md)
- [Development Setup](../docs/DEVELOPMENT.md)
- [API Documentation](../docs/api/)