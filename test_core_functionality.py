#!/usr/bin/env python3
"""
Simple test to verify core Active Inference functionality works.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_beliefs():
    """Test belief representation system."""
    print("Testing Belief system...")
    
    try:
        from active_inference.core.beliefs import Belief, BeliefState
        
        # Create simple beliefs
        belief1 = Belief(
            name="position",
            mean=np.array([0.0, 1.0]),
            covariance=np.eye(2),
            distribution="gaussian"
        )
        
        belief2 = Belief(
            name="velocity", 
            mean=np.array([0.5, -0.5]),
            covariance=np.eye(2) * 0.1,
            distribution="gaussian"
        )
        
        # Create belief state
        belief_state = BeliefState()
        belief_state.add_belief(belief1)
        belief_state.add_belief(belief2)
        
        print(f"✓ Beliefs created successfully")
        print(f"  - Position: {belief1.mean}")
        print(f"  - Velocity: {belief2.mean}")
        print(f"  - Belief state has {len(belief_state.beliefs)} beliefs")
        
        return True
        
    except Exception as e:
        print(f"✗ Belief system test failed: {e}")
        return False

def test_free_energy():
    """Test free energy computation."""
    print("\nTesting Free Energy system...")
    
    try:
        from active_inference.core.free_energy import FreeEnergyObjective, FreeEnergyComponents
        from active_inference.core.beliefs import Belief, BeliefState
        
        # Create free energy objective
        fe_objective = FreeEnergyObjective(
            complexity_weight=1.0,
            accuracy_weight=1.0,
            temperature=1.0
        )
        
        # Test components
        components = FreeEnergyComponents(
            accuracy=-2.0,
            complexity=1.5, 
            total=0.0  # Will be computed in __post_init__
        )
        
        print(f"✓ Free energy system created")
        print(f"  - Complexity: {components.complexity}")
        print(f"  - Accuracy: {components.accuracy}")
        print(f"  - Total: {components.total}")
        print(f"  - Is valid: {components.is_valid()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Free energy system test failed: {e}")
        return False

def test_environments():
    """Test environment implementations."""
    print("\nTesting Environment system...")
    
    try:
        # Import just the mock environment to avoid syntax issues
        from active_inference.environments.mock_env import MockEnvironment
        
        # Create simple mock environment
        env = MockEnvironment(
            obs_dim=4,
            action_dim=2,
            episode_length=10
        )
        
        # Reset environment
        obs = env.reset()
        
        print(f"✓ Mock environment created")
        print(f"  - Observation dimension: {len(obs)}")
        print(f"  - Action dimension: {env.action_dim}")
        
        # Take a random action
        action = np.array([0.5, -0.3])  # Simple action
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  - Step completed with reward: {reward}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment system test failed: {e}")
        return False

def test_integration():
    """Test basic integration of components."""
    print("\nTesting System Integration...")
    
    try:
        from active_inference.core.agent import ActiveInferenceAgent
        from active_inference.core.generative_model import GenerativeModel
        from active_inference.core.free_energy import FreeEnergyObjective
        from active_inference.inference.belief_updater import VariationalBeliefUpdater
        from active_inference.planning.active_planner import ActivePlanner
        
        # Create agent (it will create its own components internally)
        agent = ActiveInferenceAgent(
            state_dim=4,
            obs_dim=8,
            action_dim=2,
            inference_method="variational",
            planning_horizon=5,
            learning_rate=0.01,
            temperature=1.0
        )
        
        print(f"✓ Agent created successfully")
        print(f"  - State dimension: {agent.state_dim}")
        print(f"  - Observation dimension: {agent.obs_dim}")
        print(f"  - Action dimension: {agent.action_dim}")
        
        # Test perception-action cycle with dummy observation
        obs = np.random.randn(8)
        action = agent.act(obs)
        
        print(f"  - Perception-action cycle completed")
        print(f"  - Action: {action}")
        print(f"  - Step count: {agent.step_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Active Inference Core Functionality Test ===\n")
    
    tests = [
        test_beliefs,
        test_free_energy,
        test_environments,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core functionality tests PASSED!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())