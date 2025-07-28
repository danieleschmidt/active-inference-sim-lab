"""
Example benchmark tests for active-inference-sim-lab.

These tests measure performance and help track regressions.
"""

import numpy as np
import pytest


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_matrix_multiplication_benchmark(self, benchmark):
        """Benchmark matrix multiplication performance."""
        
        def matrix_multiply():
            A = np.random.randn(1000, 1000)
            B = np.random.randn(1000, 1000)
            return np.dot(A, B)
        
        result = benchmark(matrix_multiply)
        assert result.shape == (1000, 1000)
    
    def test_belief_update_benchmark(self, benchmark):
        """Benchmark belief update performance."""
        
        def belief_update():
            # Simulate belief update computation
            prior = np.random.randn(100)
            observation = np.random.randn(50)
            likelihood_matrix = np.random.randn(50, 100)
            
            # Simple variational update
            for _ in range(10):
                prediction = np.dot(likelihood_matrix, prior)
                error = observation - prediction
                gradient = np.dot(likelihood_matrix.T, error)
                prior += 0.01 * gradient
            
            return prior
        
        result = benchmark(belief_update)
        assert result.shape == (100,)
    
    def test_planning_benchmark(self, benchmark):
        """Benchmark planning algorithm performance."""
        
        def planning_step():
            # Simulate planning computation
            state_dim = 20
            action_dim = 5
            horizon = 10
            num_trajectories = 100
            
            best_value = -np.inf
            best_actions = None
            
            for _ in range(num_trajectories):
                actions = np.random.randn(horizon, action_dim)
                
                # Simulate trajectory evaluation
                state = np.random.randn(state_dim)
                value = 0
                
                for action in actions:
                    state += 0.1 * action[:state_dim] + 0.01 * np.random.randn(state_dim)
                    value -= np.sum(state**2)  # Quadratic cost
                
                if value > best_value:
                    best_value = value
                    best_actions = actions
            
            return best_actions
        
        result = benchmark(planning_step)
        assert result.shape == (10, 5)
    
    @pytest.mark.slow
    def test_full_episode_benchmark(self, benchmark, mock_environment):
        """Benchmark full episode performance."""
        
        class BenchmarkAgent:
            def __init__(self):
                self.policy = np.random.randn(4, 2)
            
            def act(self, observation):
                return np.tanh(np.dot(observation, self.policy))
        
        def run_episode():
            agent = BenchmarkAgent()
            env = mock_environment
            
            obs = env.reset()
            total_reward = 0
            
            for _ in range(100):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            return total_reward
        
        result = benchmark(run_episode)
        assert isinstance(result, (int, float))


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Memory usage benchmark tests."""
    
    def test_large_array_operations(self, benchmark):
        """Benchmark operations on large arrays."""
        
        def large_array_ops():
            # Create large arrays
            x = np.random.randn(10000, 100)
            y = np.random.randn(100, 1000)
            
            # Perform operations
            result1 = np.dot(x, y)
            result2 = np.sum(result1, axis=0)
            result3 = np.mean(result2)
            
            return result3
        
        result = benchmark(large_array_ops)
        assert isinstance(result, (int, float))
    
    def test_iterative_computation(self, benchmark):
        """Benchmark iterative computations."""
        
        def iterative_computation():
            state = np.random.randn(1000)
            
            for i in range(100):
                # Simulate iterative belief updates
                noise = 0.01 * np.random.randn(1000)
                state = 0.99 * state + noise
                
                # Normalize to prevent explosion
                if i % 10 == 0:
                    state = state / np.linalg.norm(state)
            
            return state
        
        result = benchmark(iterative_computation)
        assert result.shape == (1000,)


@pytest.mark.benchmark
class TestAlgorithmicComplexity:
    """Tests for algorithmic complexity analysis."""
    
    @pytest.mark.parametrize("size", [100, 500, 1000, 2000])
    def test_matrix_operations_scaling(self, benchmark, size):
        """Test how matrix operations scale with size."""
        
        def matrix_ops():
            A = np.random.randn(size, size)
            B = np.random.randn(size, size)
            
            # Matrix multiplication
            C = np.dot(A, B)
            
            # Eigenvalue decomposition (more expensive)
            eigenvals = np.linalg.eigvals(C)
            
            return eigenvals
        
        result = benchmark(matrix_ops)
        assert len(result) == size
    
    @pytest.mark.parametrize("horizon", [5, 10, 20, 50])
    def test_planning_horizon_scaling(self, benchmark, horizon):
        """Test how planning scales with horizon length."""
        
        def planning_with_horizon():
            state_dim = 10
            action_dim = 3
            
            # Simulate planning over horizon
            states = [np.random.randn(state_dim)]
            actions = []
            
            for t in range(horizon):
                action = np.random.randn(action_dim)
                actions.append(action)
                
                # Simple dynamics
                next_state = states[-1] + 0.1 * action[:state_dim]
                states.append(next_state)
            
            # Compute trajectory cost
            cost = sum(np.sum(s**2) for s in states)
            
            return cost, actions
        
        cost, actions = benchmark(planning_with_horizon)
        assert isinstance(cost, (int, float))
        assert len(actions) == horizon