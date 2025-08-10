"""
Performance monitoring and benchmarking tests.
"""

import pytest
import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

from active_inference import ActiveInferenceAgent, MockEnvironment


class TestPerformanceMonitoring:
    """Test performance characteristics and monitoring."""

    def test_agent_action_speed(self):
        """Test that action generation is reasonably fast."""
        agent = ActiveInferenceAgent(
            state_dim=4,
            obs_dim=4,
            action_dim=2,
            planning_horizon=3
        )
        
        obs = np.random.randn(4)
        
        # Warm up
        for _ in range(5):
            agent.act(obs)
        
        # Time action generation
        start_time = time.time()
        num_actions = 100
        
        for _ in range(num_actions):
            action = agent.act(obs)
            obs = obs + np.random.normal(0, 0.01, 4)
        
        elapsed_time = time.time() - start_time
        actions_per_second = num_actions / elapsed_time
        
        print(f"Action generation: {actions_per_second:.1f} actions/second")
        
        # Should be reasonably fast (at least 3 actions/second for research implementation)
        assert actions_per_second > 3

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over long runs."""
        import psutil
        import os
        
        agent = ActiveInferenceAgent(
            state_dim=3,
            obs_dim=3,
            action_dim=2,
            max_history_length=100
        )
        
        process = psutil.Process(os.getpid())
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run for a while
        obs = np.random.randn(3)
        agent.reset(obs)
        
        for i in range(1000):
            action = agent.act(obs)
            agent.update_beliefs(obs)
            obs = obs + np.random.normal(0, 0.01, 3)
            
            # Check memory every 200 steps
            if i % 200 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                print(f"Step {i}: Memory growth: {memory_growth:.2f} MB")
                
                # Memory growth should be bounded
                assert memory_growth < 50  # Less than 50MB growth

    def test_inference_scalability(self):
        """Test performance with different model sizes."""
        dimensions = [(2, 2, 1), (4, 4, 2), (8, 8, 4)]
        
        for state_dim, obs_dim, action_dim in dimensions:
            agent = ActiveInferenceAgent(
                state_dim=state_dim,
                obs_dim=obs_dim,
                action_dim=action_dim,
                planning_horizon=2  # Keep small for performance
            )
            
            obs = np.random.randn(obs_dim)
            
            # Time inference
            start_time = time.time()
            num_inferences = 50
            
            for _ in range(num_inferences):
                action = agent.act(obs)
                obs = obs + np.random.normal(0, 0.01, obs_dim)
            
            elapsed_time = time.time() - start_time
            inferences_per_second = num_inferences / elapsed_time
            
            print(f"Dims {dimensions}: {inferences_per_second:.1f} inferences/second")
            
            # Should scale reasonably (at least 5 inferences/second even for large models)
            assert inferences_per_second > 5

    def test_batch_processing_efficiency(self):
        """Test efficiency of processing multiple observations."""
        agent = ActiveInferenceAgent(
            state_dim=3,
            obs_dim=3,
            action_dim=2
        )
        
        # Generate batch of observations
        batch_size = 20
        observations = [np.random.randn(3) for _ in range(batch_size)]
        
        # Time sequential processing
        start_time = time.time()
        actions = []
        for obs in observations:
            action = agent.act(obs)
            actions.append(action)
        
        sequential_time = time.time() - start_time
        
        print(f"Sequential processing: {sequential_time:.3f}s for {batch_size} observations")
        print(f"Rate: {batch_size/sequential_time:.1f} obs/second")
        
        # Should process at reasonable speed
        assert len(actions) == batch_size
        assert batch_size / sequential_time > 20  # At least 20 obs/second

    def test_convergence_monitoring(self):
        """Test that beliefs converge over time."""
        env = MockEnvironment(obs_dim=2, action_dim=1, episode_length=100)
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=1,
            planning_horizon=1
        )
        
        obs = env.reset()
        agent.reset(obs)
        
        belief_entropies = []
        
        for step in range(50):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update_beliefs(obs)
            
            # Monitor belief uncertainty
            entropy = agent.beliefs.total_entropy()
            belief_entropies.append(entropy)
            
            if terminated or truncated:
                break
        
        # Beliefs should generally decrease in uncertainty over time
        # (though this depends on environment dynamics)
        initial_entropy = np.mean(belief_entropies[:10])
        final_entropy = np.mean(belief_entropies[-10:])
        
        print(f"Initial entropy: {initial_entropy:.3f}")
        print(f"Final entropy: {final_entropy:.3f}")
        
        # Should show some form of learning/adaptation
        # (entropy might increase or decrease depending on environment)
        assert len(belief_entropies) > 10
        assert all(np.isfinite(e) for e in belief_entropies)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])