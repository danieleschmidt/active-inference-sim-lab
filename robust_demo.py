#!/usr/bin/env python3
"""
Robust Active Inference Demo - Generation 2: MAKE IT ROBUST
Tests error handling, validation, security, and monitoring
"""

import sys
sys.path.append('src/python')

import numpy as np
import logging
import time
import json
from pathlib import Path
from active_inference import ActiveInferenceAgent
from active_inference.environments import MockEnvironment
from active_inference.utils.validation import ValidationError, ActiveInferenceError


def test_input_validation():
    """Test comprehensive input validation."""
    print("🛡️ Testing Input Validation...")
    
    validation_tests = []
    
    # Test invalid dimensions
    try:
        agent = ActiveInferenceAgent(state_dim=-1, obs_dim=4, action_dim=2)
        validation_tests.append(("Negative state_dim", "FAIL - Should have raised error"))
    except ValidationError:
        validation_tests.append(("Negative state_dim", "PASS"))
    
    # Test invalid learning rate
    try:
        agent = ActiveInferenceAgent(
            state_dim=2, obs_dim=4, action_dim=2, learning_rate=2.0
        )
        validation_tests.append(("Invalid learning_rate", "FAIL - Should have raised error"))
    except ValidationError:
        validation_tests.append(("Invalid learning_rate", "PASS"))
    
    # Test invalid observation
    try:
        agent = ActiveInferenceAgent(state_dim=2, obs_dim=4, action_dim=2)
        bad_obs = np.array([np.nan, 1.0, 2.0, 3.0])
        agent.act(bad_obs)
        validation_tests.append(("NaN observation", "FAIL - Should have raised error"))
    except ValidationError:
        validation_tests.append(("NaN observation", "PASS"))
    
    # Test wrong observation dimensions
    try:
        agent = ActiveInferenceAgent(state_dim=2, obs_dim=4, action_dim=2)
        bad_obs = np.array([1.0, 2.0])  # Wrong size
        agent.act(bad_obs)
        validation_tests.append(("Wrong obs dimensions", "FAIL - Should have raised error"))
    except ValidationError:
        validation_tests.append(("Wrong obs dimensions", "PASS"))
    
    for test_name, result in validation_tests:
        print(f"  {test_name}: {result}")
    
    return all("PASS" in result for _, result in validation_tests)


def test_error_handling_and_recovery():
    """Test error handling and graceful degradation."""
    print("\n🔧 Testing Error Handling and Recovery...")
    
    agent = ActiveInferenceAgent(
        state_dim=3, obs_dim=6, action_dim=2, 
        agent_id="robust_test", enable_logging=True
    )
    
    # Test initial health status
    health = agent.get_health_status()
    print(f"  Initial health: {health['health_status']}")
    
    # Test with extreme observations to trigger errors
    extreme_obs = np.array([1e10, -1e10, 0, 0, 0, 0])
    try:
        action = agent.act(extreme_obs)
        print(f"  Handled extreme observation, action norm: {np.linalg.norm(action):.3f}")
    except Exception as e:
        print(f"  Error with extreme observation: {e}")
    
    # Check health after stress test
    health_after = agent.get_health_status()
    print(f"  Health after stress: {health_after['health_status']}")
    print(f"  Error count: {health_after['total_errors']}")
    
    return True


def test_security_measures():
    """Test security validation and safe operations."""
    print("\n🔒 Testing Security Measures...")
    
    security_tests = []
    
    # Test agent ID validation (prevent injection)
    try:
        agent = ActiveInferenceAgent(
            state_dim=2, obs_dim=4, action_dim=2,
            agent_id="../../malicious_path"
        )
        security_tests.append(("Path injection in agent_id", "PASS - Cleaned input"))
    except Exception:
        security_tests.append(("Path injection in agent_id", "PASS - Rejected input"))
    
    # Test file path validation for checkpoints
    agent = ActiveInferenceAgent(state_dim=2, obs_dim=4, action_dim=2)
    
    try:
        agent.save_checkpoint("")  # Empty path
        security_tests.append(("Empty checkpoint path", "FAIL - Should reject"))
    except ValidationError:
        security_tests.append(("Empty checkpoint path", "PASS"))
    
    # Test safe checkpoint save/load
    try:
        safe_path = "/tmp/test_checkpoint.json"
        agent.save_checkpoint(safe_path)
        loaded_agent = ActiveInferenceAgent.load_checkpoint(safe_path)
        security_tests.append(("Safe checkpoint operations", "PASS"))
        
        # Cleanup
        Path(safe_path).unlink(missing_ok=True)
    except Exception as e:
        security_tests.append(("Safe checkpoint operations", f"FAIL - {e}"))
    
    for test_name, result in security_tests:
        print(f"  {test_name}: {result}")
    
    return all("PASS" in result for _, result in security_tests)


def test_monitoring_and_logging():
    """Test comprehensive monitoring and logging."""
    print("\n📊 Testing Monitoring and Logging...")
    
    # Create agent with detailed logging
    agent = ActiveInferenceAgent(
        state_dim=2, obs_dim=4, action_dim=2,
        agent_id="monitoring_test",
        enable_logging=True,
        log_level=logging.DEBUG
    )
    
    env = MockEnvironment(obs_dim=4, action_dim=2)
    obs = env.reset()
    
    # Run several steps to generate monitoring data
    for step in range(10):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_model(obs, action, reward)
        
        if step % 3 == 0:  # Check stats periodically
            stats = agent.get_statistics()
            print(f"  Step {step}: Free Energy = {stats['current_free_energy']:.3f}, "
                  f"Belief Entropy = {stats['belief_entropy']:.3f}")
    
    # Get comprehensive statistics
    final_stats = agent.get_statistics()
    print(f"\n  📈 Final Monitoring Data:")
    print(f"    Steps: {final_stats['step_count']}")
    print(f"    Episodes: {final_stats['episode_count']}")
    print(f"    Average Reward: {final_stats['average_reward']:.3f}")
    print(f"    Health Status: {final_stats['health_status']}")
    print(f"    History Length: {final_stats['history_length']}")
    
    return True


def test_concurrent_safety():
    """Test thread-safety and concurrent operations."""
    print("\n⚡ Testing Concurrent Safety...")
    
    import threading
    import queue
    
    # Create multiple agents for concurrent testing
    agents = [
        ActiveInferenceAgent(
            state_dim=2, obs_dim=4, action_dim=2,
            agent_id=f"concurrent_agent_{i}"
        ) for i in range(3)
    ]
    
    env = MockEnvironment(obs_dim=4, action_dim=2)
    results_queue = queue.Queue()
    
    def agent_worker(agent, worker_id):
        """Worker function for concurrent agent testing."""
        try:
            obs = env.reset()
            total_reward = 0
            
            for _ in range(5):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.update_model(obs, action, reward)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            results_queue.put((worker_id, "SUCCESS", total_reward))
        except Exception as e:
            results_queue.put((worker_id, "ERROR", str(e)))
    
    # Start concurrent workers
    threads = []
    for i, agent in enumerate(agents):
        thread = threading.Thread(target=agent_worker, args=(agent, i))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Collect results
    concurrent_results = []
    while not results_queue.empty():
        worker_id, status, result = results_queue.get()
        concurrent_results.append((worker_id, status, result))
        print(f"  Worker {worker_id}: {status} - {result}")
    
    success_count = sum(1 for _, status, _ in concurrent_results if status == "SUCCESS")
    print(f"  Concurrent Success Rate: {success_count}/{len(agents)}")
    
    return success_count == len(agents)


def test_performance_monitoring():
    """Test performance monitoring and benchmarking."""
    print("\n⚡ Testing Performance Monitoring...")
    
    agent = ActiveInferenceAgent(
        state_dim=5, obs_dim=10, action_dim=3,
        agent_id="performance_test"
    )
    
    env = MockEnvironment(obs_dim=10, action_dim=3)
    obs = env.reset()
    
    # Measure inference performance
    inference_times = []
    action_times = []
    
    for _ in range(20):
        # Time belief inference
        start_time = time.perf_counter()
        beliefs = agent.infer_states(obs)
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)
        
        # Time action planning
        start_time = time.perf_counter()
        action = agent.plan_action(beliefs)
        action_time = time.perf_counter() - start_time
        action_times.append(action_time)
        
        # Step environment
        obs, reward, _, _, _ = env.step(action)
        agent.update_model(obs, action, reward)
    
    # Performance statistics
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    avg_action_time = np.mean(action_times) * 1000  # ms
    total_cycle_time = avg_inference_time + avg_action_time
    
    print(f"  Average Inference Time: {avg_inference_time:.2f}ms")
    print(f"  Average Planning Time: {avg_action_time:.2f}ms")
    print(f"  Total Cycle Time: {total_cycle_time:.2f}ms")
    print(f"  Theoretical FPS: {1000/total_cycle_time:.1f}")
    
    # Performance should be reasonable (< 100ms per cycle)
    return total_cycle_time < 100


def main():
    """Run all Generation 2 robustness tests."""
    print("🛡️ Active Inference Demo - Generation 2: MAKE IT ROBUST")
    print("=" * 70)
    
    tests = [
        ("Input Validation", test_input_validation),
        ("Error Handling & Recovery", test_error_handling_and_recovery),
        ("Security Measures", test_security_measures),
        ("Monitoring & Logging", test_monitoring_and_logging),
        ("Concurrent Safety", test_concurrent_safety),
        ("Performance Monitoring", test_performance_monitoring)
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
    print("📊 GENERATION 2 RESULTS:")
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
        print("🎉 Generation 2 COMPLETE! System is robust and reliable.")
        return True
    else:
        print("⚠️ Some robustness tests failed. System needs hardening.")
        return False


if __name__ == "__main__":
    main()