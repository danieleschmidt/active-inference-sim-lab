#!/usr/bin/env python3
"""
Simple Active Inference Demo - Generation 1: Make It Work
Tests basic functionality of the Active Inference Agent
"""

import sys
sys.path.append('src/python')

import numpy as np
from active_inference import ActiveInferenceAgent
from active_inference.environments import MockEnvironment


def test_basic_agent_functionality():
    """Test basic agent creation and operation."""
    print("🧠 Testing Basic Active Inference Agent...")
    
    # Create a simple agent
    agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=8, 
        action_dim=2,
        agent_id="demo_agent"
    )
    print(f"✅ Agent created: {agent}")
    
    # Create mock environment 
    env = MockEnvironment(obs_dim=8, action_dim=2)
    print(f"✅ Environment created: {env}")
    
    # Test basic perception-action cycle
    obs = env.reset()
    print(f"📊 Initial observation shape: {obs.shape}")
    
    for step in range(5):
        # Agent acts based on observation
        action = agent.act(obs)
        print(f"Step {step + 1}: Action = {action[:3]}...")  # Show first 3 values
        
        # Environment responds
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update_model(obs, action, reward)
        
        if done:
            break
    
    # Get agent statistics
    stats = agent.get_statistics()
    print(f"📈 Final Statistics:")
    print(f"  - Steps: {stats['step_count']}")
    print(f"  - Total Reward: {stats['total_reward']:.3f}")
    print(f"  - Health: {stats['health_status']}")
    
    return True


def test_agent_reset_functionality():
    """Test agent reset and episode management."""
    print("\n🔄 Testing Agent Reset...")
    
    agent = ActiveInferenceAgent(
        state_dim=2,
        obs_dim=4,
        action_dim=1,
        agent_id="reset_test"
    )
    
    env = MockEnvironment(obs_dim=4, action_dim=1)
    
    # Run first episode
    obs = env.reset()
    for _ in range(3):
        action = agent.act(obs)
        obs, _, _, _, _ = env.step(action)
    
    initial_steps = agent.step_count
    print(f"Steps after first episode: {initial_steps}")
    
    # Reset agent
    agent.reset(obs)
    print(f"✅ Agent reset, episode count: {agent.episode_count}")
    
    # Run second episode  
    for _ in range(2):
        action = agent.act(obs)
        obs, _, _, _, _ = env.step(action)
    
    final_steps = agent.step_count
    print(f"Steps after reset and second episode: {final_steps}")
    
    return final_steps < initial_steps  # Should be fewer steps after reset


def test_belief_updating():
    """Test belief state updating."""
    print("\n🧠 Testing Belief Updates...")
    
    agent = ActiveInferenceAgent(
        state_dim=3,
        obs_dim=6,
        action_dim=2,
        agent_id="belief_test"
    )
    
    # Generate some observations
    observations = [np.random.randn(6) for _ in range(3)]
    
    for i, obs in enumerate(observations):
        beliefs = agent.infer_states(obs)
        print(f"Observation {i + 1}: Beliefs updated = {len(beliefs.get_all_beliefs())} components")
    
    return True


def main():
    """Run all Generation 1 tests."""
    print("🚀 Active Inference Demo - Generation 1: MAKE IT WORK")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_agent_functionality),
        ("Reset Functionality", test_agent_reset_functionality), 
        ("Belief Updates", test_belief_updating)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
            print(f"✅ {test_name}: PASSED\n")
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
            print(f"❌ {test_name}: FAILED - {e}\n")
    
    print("=" * 60)
    print("📊 GENERATION 1 RESULTS:")
    for test_name, result in results:
        print(f"  {test_name}: {result}")
    
    success_count = sum(1 for _, result in results if result == "PASS")
    print(f"\n🎯 Success Rate: {success_count}/{len(tests)} tests passed")
    
    if success_count == len(tests):
        print("🎉 Generation 1 COMPLETE! Basic functionality works.")
        return True
    else:
        print("⚠️ Some tests failed. Need fixes before proceeding.")
        return False


if __name__ == "__main__":
    main()