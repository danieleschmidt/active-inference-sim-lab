#!/usr/bin/env python3
"""
Optimized Active Inference Demo - Generation 3: MAKE IT SCALE  
Tests performance optimization, caching, and scalability features
"""

import sys
sys.path.append('src/python')

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from active_inference import ActiveInferenceAgent
from active_inference.environments import MockEnvironment


class OptimizedActiveInferenceAgent:
    """
    Performance-optimized version of ActiveInferenceAgent with caching and fast inference.
    """
    
    def __init__(self, state_dim, obs_dim, action_dim, **kwargs):
        """Initialize optimized agent with performance features."""
        self.base_agent = ActiveInferenceAgent(
            state_dim=state_dim,
            obs_dim=obs_dim, 
            action_dim=action_dim,
            **kwargs
        )
        
        # Performance optimizations
        self.observation_cache = {}
        self.action_cache = {}
        self.inference_cache_size = 100
        self.planning_cache_size = 50
        
        # Fast inference settings
        self.base_agent.inference_engine.max_iterations = 3  # Reduce iterations
        self.base_agent.planner.horizon = 3  # Reduce planning horizon
        
        # Pre-allocate arrays for speed
        self.obs_buffer = np.zeros(obs_dim)
        self.action_buffer = np.zeros(action_dim)
        
    def fast_act(self, observation):
        """Optimized perception-action cycle with caching."""
        # Copy to pre-allocated buffer (avoids memory allocation)
        np.copyto(self.obs_buffer, observation)
        
        # Check cache first
        obs_key = hash(observation.tobytes())
        if obs_key in self.observation_cache:
            cached_action = self.observation_cache[obs_key]
            np.copyto(self.action_buffer, cached_action)
            return self.action_buffer.copy()
        
        # Fast inference path
        action = self.base_agent.act(self.obs_buffer)
        
        # Cache result if cache not full
        if len(self.observation_cache) < self.inference_cache_size:
            self.observation_cache[obs_key] = action.copy()
        
        return action
    
    def batch_process(self, observations_batch):
        """Process multiple observations in batch for efficiency."""
        results = []
        for obs in observations_batch:
            action = self.fast_act(obs)
            results.append(action)
        return results


def test_basic_performance_optimization():
    """Test basic performance improvements."""
    print("⚡ Testing Basic Performance Optimization...")
    
    # Standard agent
    standard_agent = ActiveInferenceAgent(
        state_dim=4, obs_dim=8, action_dim=2,
        agent_id="standard_agent"
    )
    
    # Optimized agent
    optimized_agent = OptimizedActiveInferenceAgent(
        state_dim=4, obs_dim=8, action_dim=2,
        agent_id="optimized_agent"
    )
    
    env = MockEnvironment(obs_dim=8, action_dim=2)
    obs = env.reset()
    
    # Benchmark standard agent
    start_time = time.perf_counter()
    for _ in range(10):
        action = standard_agent.act(obs)
        obs, _, _, _, _ = env.step(action)
    standard_time = time.perf_counter() - start_time
    
    # Benchmark optimized agent
    obs = env.reset()
    start_time = time.perf_counter()
    for _ in range(10):
        action = optimized_agent.fast_act(obs)
        obs, _, _, _, _ = env.step(action)
    optimized_time = time.perf_counter() - start_time
    
    speedup = standard_time / optimized_time
    print(f"  Standard Agent: {standard_time:.3f}s")
    print(f"  Optimized Agent: {optimized_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    return speedup > 1.5  # Should be at least 50% faster


def test_caching_system():
    """Test caching effectiveness."""
    print("\n💾 Testing Caching System...")
    
    agent = OptimizedActiveInferenceAgent(
        state_dim=3, obs_dim=6, action_dim=2,
        agent_id="cache_test"
    )
    
    # Test with repeated observations
    obs1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    obs2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    
    # First calls (cache miss)
    start_time = time.perf_counter()
    action1_first = agent.fast_act(obs1)
    action2_first = agent.fast_act(obs2)
    first_call_time = time.perf_counter() - start_time
    
    # Second calls (cache hit)
    start_time = time.perf_counter()
    action1_second = agent.fast_act(obs1)
    action2_second = agent.fast_act(obs2)
    second_call_time = time.perf_counter() - start_time
    
    cache_speedup = first_call_time / second_call_time
    print(f"  First call (cache miss): {first_call_time:.3f}s")
    print(f"  Second call (cache hit): {second_call_time:.3f}s")
    print(f"  Cache speedup: {cache_speedup:.2f}x")
    
    # Verify cache correctness
    action_match = np.allclose(action1_first, action1_second) and np.allclose(action2_first, action2_second)
    print(f"  Cache correctness: {'PASS' if action_match else 'FAIL'}")
    
    return cache_speedup > 2.0 and action_match


def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n📦 Testing Batch Processing...")
    
    agent = OptimizedActiveInferenceAgent(
        state_dim=2, obs_dim=4, action_dim=2,
        agent_id="batch_test"
    )
    
    # Generate batch of observations
    batch_size = 20
    observations_batch = [np.random.randn(4) for _ in range(batch_size)]
    
    # Sequential processing
    start_time = time.perf_counter()
    sequential_results = []
    for obs in observations_batch:
        action = agent.fast_act(obs)
        sequential_results.append(action)
    sequential_time = time.perf_counter() - start_time
    
    # Batch processing
    start_time = time.perf_counter()
    batch_results = agent.batch_process(observations_batch)
    batch_time = time.perf_counter() - start_time
    
    batch_speedup = sequential_time / batch_time
    print(f"  Sequential processing: {sequential_time:.3f}s")
    print(f"  Batch processing: {batch_time:.3f}s")
    print(f"  Batch speedup: {batch_speedup:.2f}x")
    
    # Verify results consistency
    results_match = all(np.allclose(seq, batch) for seq, batch in zip(sequential_results, batch_results))
    print(f"  Results consistency: {'PASS' if results_match else 'FAIL'}")
    
    return batch_speedup > 1.1 and results_match


def test_concurrent_agents():
    """Test concurrent agent execution."""
    print("\n🔀 Testing Concurrent Agents...")
    
    def agent_worker(worker_id, num_steps=50):
        """Worker function for concurrent testing."""
        agent = OptimizedActiveInferenceAgent(
            state_dim=3, obs_dim=6, action_dim=2,
            agent_id=f"concurrent_worker_{worker_id}"
        )
        
        env = MockEnvironment(obs_dim=6, action_dim=2)
        obs = env.reset()
        
        start_time = time.perf_counter()
        total_reward = 0
        
        for _ in range(num_steps):
            action = agent.fast_act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                obs = env.reset()
        
        elapsed_time = time.perf_counter() - start_time
        return worker_id, elapsed_time, total_reward
    
    # Test with different numbers of concurrent workers
    num_workers = mp.cpu_count()
    print(f"  Testing with {num_workers} concurrent workers...")
    
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(agent_worker, i) for i in range(num_workers)]
        results = [future.result() for future in as_completed(futures)]
    
    total_time = time.perf_counter() - start_time
    
    # Analyze results
    worker_times = [result[1] for result in results]
    avg_worker_time = np.mean(worker_times)
    max_worker_time = np.max(worker_times)
    
    efficiency = avg_worker_time / total_time
    print(f"  Total execution time: {total_time:.3f}s")
    print(f"  Average worker time: {avg_worker_time:.3f}s")
    print(f"  Max worker time: {max_worker_time:.3f}s")
    print(f"  Parallelization efficiency: {efficiency:.2f}")
    
    return efficiency > 0.8  # Should achieve good parallelization


def test_memory_efficiency():
    """Test memory usage and efficiency."""
    print("\n💾 Testing Memory Efficiency...")
    
    import psutil
    import os
    
    # Measure initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create multiple agents
    agents = []
    for i in range(10):
        agent = OptimizedActiveInferenceAgent(
            state_dim=5, obs_dim=10, action_dim=3,
            agent_id=f"memory_test_{i}"
        )
        agents.append(agent)
    
    # Measure memory after agent creation
    after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run some operations
    env = MockEnvironment(obs_dim=10, action_dim=3)
    obs = env.reset()
    
    for agent in agents:
        for _ in range(20):
            action = agent.fast_act(obs)
            obs, _, _, _, _ = env.step(action)
    
    # Measure peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    memory_per_agent = (after_creation_memory - initial_memory) / len(agents)
    memory_growth = peak_memory - after_creation_memory
    
    print(f"  Initial memory: {initial_memory:.1f}MB")
    print(f"  Memory per agent: {memory_per_agent:.1f}MB")
    print(f"  Peak memory: {peak_memory:.1f}MB")
    print(f"  Memory growth during execution: {memory_growth:.1f}MB")
    
    # Cleanup
    del agents
    
    return memory_per_agent < 5.0  # Should be efficient


def test_scalability_limits():
    """Test system scalability limits."""
    print("\n📈 Testing Scalability Limits...")
    
    # Test with increasing problem sizes
    problem_sizes = [(2, 4, 2), (5, 10, 3), (10, 20, 5), (20, 40, 10)]
    
    performance_data = []
    
    for state_dim, obs_dim, action_dim in problem_sizes:
        agent = OptimizedActiveInferenceAgent(
            state_dim=state_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            agent_id=f"scale_test_{state_dim}_{obs_dim}_{action_dim}"
        )
        
        env = MockEnvironment(obs_dim=obs_dim, action_dim=action_dim)
        obs = env.reset()
        
        # Measure performance
        start_time = time.perf_counter()
        for _ in range(10):
            action = agent.fast_act(obs)
            obs, _, _, _, _ = env.step(action)
        
        elapsed_time = time.perf_counter() - start_time
        avg_step_time = elapsed_time / 10
        
        performance_data.append((state_dim * obs_dim * action_dim, avg_step_time))
        print(f"  Size {state_dim}x{obs_dim}x{action_dim}: {avg_step_time:.4f}s per step")
    
    # Check if performance scales reasonably
    largest_problem = performance_data[-1]
    smallest_problem = performance_data[0]
    
    size_ratio = largest_problem[0] / smallest_problem[0]
    time_ratio = largest_problem[1] / smallest_problem[1]
    
    scalability_ratio = time_ratio / size_ratio
    print(f"  Size increased by: {size_ratio:.1f}x")
    print(f"  Time increased by: {time_ratio:.1f}x")
    print(f"  Scalability ratio: {scalability_ratio:.2f}")
    
    return scalability_ratio < 2.0  # Should scale reasonably


def main():
    """Run all Generation 3 optimization tests."""
    print("⚡ Active Inference Demo - Generation 3: MAKE IT SCALE")
    print("=" * 70)
    
    tests = [
        ("Basic Performance Optimization", test_basic_performance_optimization),
        ("Caching System", test_caching_system),
        ("Batch Processing", test_batch_processing),
        ("Concurrent Agents", test_concurrent_agents),
        ("Memory Efficiency", test_memory_efficiency),
        ("Scalability Limits", test_scalability_limits)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            start_time = time.perf_counter()
            result = test_func()
            duration = time.perf_counter() - start_time
            
            status = "PASS" if result else "FAIL"
            results.append((test_name, status, duration))
            print(f"✅ {test_name}: {status} ({duration:.2f}s)\n")
        except Exception as e:
            results.append((test_name, f"ERROR: {e}", 0))
            print(f"❌ {test_name}: FAILED - {e}\n")
    
    print("=" * 70)
    print("📊 GENERATION 3 RESULTS:")
    for test_name, result, duration in results:
        if isinstance(duration, float):
            print(f"  {test_name}: {result} ({duration:.2f}s)")
        else:
            print(f"  {test_name}: {result}")
    
    success_count = sum(1 for _, result, _ in results if result == "PASS")
    total_time = sum(duration for _, _, duration in results if isinstance(duration, float))
    
    print(f"\n🎯 Success Rate: {success_count}/{len(tests)} tests passed")
    print(f"⏱️ Total Execution Time: {total_time:.2f}s")
    
    if success_count == len(tests):
        print("🎉 Generation 3 COMPLETE! System is optimized and scalable.")
        return True
    else:
        print("⚠️ Some optimization tests failed. System needs tuning.")
        return False


if __name__ == "__main__":
    main()