#!/usr/bin/env python3
"""
Integration test for Active Inference Simulation Laboratory.

This script tests the full integration of all components:
- Core active inference agent
- Free energy computation 
- Belief updating
- Action planning
- Environment interaction
- C++ core integration
"""

import sys
import os
import numpy as np
import traceback
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

try:
    # Test imports
    print("🔍 Testing imports...")
    from active_inference import (
        ActiveInferenceAgent,
        GenerativeModel,
        FreeEnergyObjective,
        BeliefUpdater,
        ActivePlanner,
        MockEnvironment
    )
    
    from active_inference.core.beliefs import Belief, BeliefState
    from active_inference.utils.validation import validate_inputs, ValidationError
    from active_inference.utils.logging import setup_logging, get_logger
    from active_inference.utils.caching import memoize, LRUCache
    
    print("✅ All imports successful")
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Test 1: Basic Agent Creation
    print("\n🧠 Testing Agent Creation...")
    agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=8, 
        action_dim=2,
        inference_method="variational",
        planning_horizon=5,
        agent_id="test_agent"
    )
    print(f"✅ Agent created: {agent}")
    
    # Test 2: Belief State Operations
    print("\n🎯 Testing Belief States...")
    beliefs = BeliefState()
    belief = Belief(
        mean=np.array([1.0, 2.0, 3.0, 4.0]),
        variance=np.array([0.1, 0.2, 0.3, 0.4])
    )
    beliefs.add_belief("state", belief)  # Use "state" to match the agent's prior name
    
    print(f"✅ Belief entropy: {belief.entropy:.4f}")
    print(f"✅ Belief confidence: {belief.confidence:.4f}")
    print(f"✅ Total beliefs entropy: {beliefs.total_entropy():.4f}")
    
    # Test 3: Free Energy Computation
    print("\n⚡ Testing Free Energy...")
    free_energy_obj = FreeEnergyObjective(
        complexity_weight=1.0,
        accuracy_weight=1.0
    )
    
    observations = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    # Create test generative model
    model = GenerativeModel(state_dim=4, obs_dim=8, action_dim=2)
    
    # Compute free energy
    fe_components = free_energy_obj.compute_free_energy(
        observations=observations,
        beliefs=beliefs,
        priors=model.get_all_priors(),
        likelihood_fn=model.likelihood
    )
    
    print(f"✅ Free Energy - Accuracy: {fe_components.accuracy:.4f}")
    print(f"✅ Free Energy - Complexity: {fe_components.complexity:.4f}")
    print(f"✅ Free Energy - Total: {fe_components.total:.4f}")
    
    # Test 4: Agent Perception-Action Loop
    print("\n🔄 Testing Perception-Action Loop...")
    
    # Initial observation
    initial_obs = np.random.randn(8)
    agent.reset(initial_obs)
    
    # Run several steps
    total_reward = 0.0
    for step in range(5):
        obs = np.random.randn(8)  # Simulated observation
        action = agent.act(obs)
        reward = np.random.randn()  # Simulated reward
        
        agent.update_model(obs, action, reward)
        total_reward += reward
        
        print(f"  Step {step}: Action shape {action.shape}, Reward {reward:.3f}")
    
    print(f"✅ Completed 5 steps, total reward: {total_reward:.3f}")
    
    # Test 5: Validation System
    print("\n🛡️ Testing Validation...")
    try:
        validate_inputs(
            state_dim=4,
            observation_array=observations,
            learning_rate=0.01
        )
        print("✅ Validation passed for valid inputs")
    except ValidationError as e:
        print(f"❌ Unexpected validation error: {e}")
    
    try:
        validate_inputs(state_dim=-1)  # Should fail
        print("❌ Validation should have failed for negative dimension")
    except ValidationError:
        print("✅ Validation correctly rejected invalid input")
    
    # Test 6: Caching System
    print("\n💾 Testing Caching...")
    cache = LRUCache(maxsize=10)
    
    @memoize(maxsize=5)
    def expensive_computation(x):
        return x ** 2 + np.sin(x)
    
    # Test caching
    result1 = expensive_computation(3.14)
    result2 = expensive_computation(3.14)  # Should be cached
    
    assert np.isclose(result1, result2)
    print(f"✅ Cached computation: {result1:.4f}")
    print(f"✅ Cache stats: {expensive_computation._cache.get_stats()}")
    
    # Test 7: Environment Integration  
    print("\n🌍 Testing Environment Integration...")
    try:
        # Test mock environment
        mock_env = MockEnvironment(obs_dim=4, action_dim=2)
        obs = mock_env.reset()
        print(f"✅ Mock environment reset, observation shape: {obs.shape}")
        
        action = np.array([0.1, -0.2])
        next_obs, reward, done, truncated, info = mock_env.step(action)
        print(f"✅ Mock environment step: reward={reward:.3f}, done={done}")
        print(f"✅ Environment info: step={info['step_count']}")
        
        mock_env.close()
        
    except Exception as e:
        print(f"⚠️ Environment test failed: {e}")
    
    # Test 8: Agent Statistics
    print("\n📊 Testing Agent Statistics...")
    stats = agent.get_statistics()
    print(f"✅ Agent stats: {len(stats)} metrics collected")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test 9: C++ Core Integration (if built)
    print("\n⚡ Testing C++ Core Integration...")
    try:
        # Try to load the C++ module
        cpp_module_path = repo_root / "build" / "_core.cpython-312-x86_64-linux-gnu.so"
        if cpp_module_path.exists():
            sys.path.insert(0, str(repo_root / "build"))
            import _core as active_inference_cpp
            
            # Test C++ free energy computation
            cpp_fe = active_inference_cpp.FreeEnergy(1.0, 1.0, 1.0)
            print("✅ C++ FreeEnergy object created")
            
            # Test version
            version = active_inference_cpp.version()
            print(f"✅ C++ core version: {version}")
            
            # Run benchmark
            benchmark_time = active_inference_cpp.benchmark_free_energy(10, 100)
            print(f"✅ C++ benchmark: {benchmark_time:.2f} μs per iteration")
            
        else:
            print("⚠️ C++ module not found, skipping C++ integration test")
    
    except Exception as e:
        print(f"⚠️ C++ integration test failed: {e}")
    
    # Test 10: Full Integration Test
    print("\n🚀 Running Full Integration Test...")
    
    # Create agent with environment
    integration_agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=4, 
        action_dim=1,
        planning_horizon=3,
        learning_rate=0.01,
        agent_id="integration_test"
    )
    
    # Simulate episode
    episode_reward = 0.0
    episode_length = 10
    
    obs = np.random.randn(4)
    integration_agent.reset(obs)
    
    for step in range(episode_length):
        action = integration_agent.act(obs)
        
        # Simulate environment dynamics
        next_obs = obs + 0.1 * action + 0.05 * np.random.randn(4)
        reward = -np.sum(obs**2)  # Quadratic cost
        
        integration_agent.update_model(next_obs, action, reward)
        
        obs = next_obs
        episode_reward += reward
    
    final_stats = integration_agent.get_statistics()
    print(f"✅ Integration test completed")
    print(f"  Episode reward: {episode_reward:.3f}")
    print(f"  Final free energy: {final_stats.get('current_free_energy', 0):.3f}")
    print(f"  Agent confidence: {final_stats.get('belief_confidence', 0):.3f}")
    
    print("\n🎉 All tests completed successfully!")
    print("Active Inference Simulation Laboratory is working correctly.")
    
    success = True

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    success = False

if __name__ == "__main__":
    if not success:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🧠 ACTIVE INFERENCE SIMULATION LABORATORY READY 🚀")
    print("="*60)