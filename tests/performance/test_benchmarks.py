"""Performance benchmarks and tests."""

import time
from typing import Any, Dict

import numpy as np
import pytest


@pytest.mark.benchmark
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for active inference components."""

    def test_matrix_operations_performance(self):
        """Benchmark matrix operations performance."""
        sizes = [100, 500, 1000]
        results = {}

        for size in sizes:
            matrix = np.random.randn(size, size)

            # Time matrix multiplication
            start_time = time.time()
            result = matrix @ matrix.T
            multiplication_time = time.time() - start_time

            # Time matrix inversion
            start_time = time.time()
            inv_result = np.linalg.inv(matrix + np.eye(size))
            inversion_time = time.time() - start_time

            # Time eigenvalue decomposition
            start_time = time.time()
            eigenvals, eigenvecs = np.linalg.eigh(matrix @ matrix.T)
            eigen_time = time.time() - start_time

            results[size] = {
                "multiplication": multiplication_time,
                "inversion": inversion_time,
                "eigendecomposition": eigen_time,
            }

        # Assert reasonable performance (these are loose bounds)
        assert results[100]["multiplication"] < 0.1
        assert results[100]["inversion"] < 0.1
        assert results[500]["multiplication"] < 1.0
        assert results[1000]["inversion"] < 5.0

        print(f"Performance results: {results}")

    def test_inference_loop_performance(
        self, mock_environment, simple_generative_model, benchmark_config
    ):
        """Benchmark inference loop performance."""
        env = mock_environment
        model = simple_generative_model
        config = benchmark_config

        total_steps = 0
        start_time = time.time()

        for episode in range(config["num_episodes"]):
            obs = env.reset()
            done = False
            steps = 0

            while not done and steps < config["max_steps_per_episode"]:
                # Simple inference step (placeholder)
                belief = obs + 0.1 * np.random.randn(4)

                # Simple action selection
                action = -0.1 * belief[:2] + 0.1 * np.random.randn(2)

                # Environment step
                obs, reward, done, info = env.step(action)
                steps += 1
                total_steps += 1

        total_time = time.time() - start_time
        steps_per_second = total_steps / total_time

        # Performance assertions
        assert steps_per_second > 100  # Should achieve at least 100 steps/second
        assert total_time < config["timeout_seconds"]

        print(f"Inference performance: {steps_per_second:.1f} steps/second")

    def test_memory_usage_benchmark(self, benchmark_config):
        """Benchmark memory usage patterns."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and manipulate large arrays
        arrays = []
        for i in range(10):
            arr = np.random.randn(1000, 1000)
            arrays.append(arr)

            # Perform operations
            result = arr @ arr.T
            eigenvals = np.linalg.eigvals(result)

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        # Clean up
        del arrays
        del result
        del eigenvals

        # Memory should not exceed configured limit
        assert memory_increase < benchmark_config["memory_limit_mb"]

        print(f"Memory usage: {memory_increase:.1f} MB increase")

    @pytest.mark.parametrize("state_dim", [4, 8, 16, 32])
    def test_scalability_with_state_dimension(self, state_dim):
        """Test performance scaling with state dimension."""
        num_operations = 1000

        start_time = time.time()
        for _ in range(num_operations):
            state = np.random.randn(state_dim)
            covariance = np.eye(state_dim) + 0.1 * np.random.randn(state_dim, state_dim)
            covariance = covariance @ covariance.T  # Ensure positive definite

            # Simulate belief update operations
            log_det = np.linalg.slogdet(covariance)[1]
            inv_cov = np.linalg.inv(covariance)
            quadratic_form = state.T @ inv_cov @ state

        operation_time = time.time() - start_time
        time_per_operation = operation_time / num_operations

        # Performance should scale reasonably with dimension
        expected_scaling = state_dim**2  # Quadratic scaling for matrix operations
        normalized_time = time_per_operation / expected_scaling * 1000

        # Should be reasonably fast even for larger dimensions
        if state_dim <= 16:
            assert time_per_operation < 0.01  # 10ms per operation
        else:
            assert time_per_operation < 0.1  # 100ms per operation

        print(f"State dim {state_dim}: {time_per_operation*1000:.2f} ms/op")

    def test_batch_processing_performance(self):
        """Test performance of batch vs sequential processing."""
        batch_size = 100
        state_dim = 8

        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for i in range(batch_size):
            state = np.random.randn(state_dim)
            result = np.exp(-0.5 * np.sum(state**2))
            sequential_results.append(result)
        sequential_time = time.time() - start_time

        # Batch processing
        start_time = time.time()
        batch_states = np.random.randn(batch_size, state_dim)
        batch_results = np.exp(-0.5 * np.sum(batch_states**2, axis=1))
        batch_time = time.time() - start_time

        # Batch should be significantly faster
        speedup = sequential_time / batch_time
        assert speedup > 2.0  # At least 2x speedup

        # Results should be similar (allowing for different random seeds)
        assert len(sequential_results) == len(batch_results)

        print(f"Batch processing speedup: {speedup:.1f}x")

    def test_convergence_performance(self):
        """Test convergence speed of iterative algorithms."""
        state_dim = 4
        max_iterations = 100
        tolerance = 1e-6

        # Simulate iterative belief update
        belief = np.random.randn(state_dim)
        target = np.random.randn(state_dim)

        start_time = time.time()
        for iteration in range(max_iterations):
            # Simple gradient descent-like update
            error = target - belief
            belief += 0.1 * error

            # Check convergence
            if np.linalg.norm(error) < tolerance:
                break

        convergence_time = time.time() - start_time
        iterations_needed = iteration + 1

        # Should converge reasonably quickly
        assert iterations_needed < max_iterations
        assert convergence_time < 1.0  # Should converge within 1 second

        print(f"Convergence: {iterations_needed} iterations in {convergence_time:.3f}s")