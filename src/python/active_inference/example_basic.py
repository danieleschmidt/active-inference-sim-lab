#!/usr/bin/env python3
"""
Basic working example demonstrating Active Inference functionality.

This example creates a simple active inference agent and demonstrates
the core perception-action loop without external dependencies.
"""

import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from active_inference.core.agent import ActiveInferenceAgent
from active_inference.core.beliefs import Belief, BeliefState
from active_inference.core.generative_model import GenerativeModel
from active_inference.environments.mock_env import MockEnvironment


class SimpleEnvironment:
    """Simple test environment for demonstration."""
    
    def __init__(self, obs_dim=4, action_dim=2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state = np.random.randn(obs_dim)
        self.step_count = 0
        
    def reset(self):
        """Reset environment to initial state."""
        self.state = np.random.randn(self.obs_dim)
        self.step_count = 0
        return self.state + np.random.randn(self.obs_dim) * 0.1
    
    def step(self, action):
        """Take a step in the environment."""
        # Simple dynamics: state evolves with action influence
        self.state += action * 0.1 + np.random.randn(self.obs_dim) * 0.05
        
        # Add observation noise
        observation = self.state + np.random.randn(self.obs_dim) * 0.1
        
        # Simple reward based on staying near origin
        reward = -np.sum(self.state**2)
        
        self.step_count += 1
        done = self.step_count >= 100
        
        return observation, reward, done


def run_basic_demo():
    """Run basic active inference demo."""
    print("🧠 Active Inference Basic Demo")
    print("=" * 50)
    
    # Environment parameters
    state_dim = 4
    obs_dim = 4
    action_dim = 2
    
    try:
        # Create environment
        print("🌍 Creating environment...")
        env = SimpleEnvironment(obs_dim, action_dim)
        
        # Create agent
        print("🤖 Creating Active Inference agent...")
        agent = ActiveInferenceAgent(
            state_dim=state_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            planning_horizon=3,
            learning_rate=0.01,
            temperature=1.0,
            agent_id="demo_agent"
        )
        
        print(f"Agent initialized: {agent}")
        
        # Run episode
        print("\n🎮 Running episode...")
        obs = env.reset()
        episode_reward = 0
        
        for step in range(50):
            # Agent perception-action cycle
            action = agent.act(obs)
            
            # Environment step
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            # Update agent model
            agent.update_model(obs, action, reward)
            
            # Print progress
            if step % 10 == 0:
                stats = agent.get_statistics()
                print(f"Step {step:2d}: reward={reward:6.2f}, "
                      f"total={episode_reward:6.2f}, "
                      f"health={stats['health_status']}")
            
            if done:
                break
        
        # Final statistics
        print("\n📊 Final Statistics:")
        final_stats = agent.get_statistics()
        for key, value in final_stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Agent health: {final_stats['health_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_belief_demo():
    """Demonstrate belief system functionality."""
    print("\n🧠 Belief System Demo")
    print("=" * 30)
    
    try:
        # Create belief state
        beliefs = BeliefState()
        
        # Add some beliefs
        beliefs.add_belief("position", Belief(
            mean=np.array([0.0, 0.0]), 
            variance=np.array([1.0, 1.0])
        ))
        
        beliefs.add_belief("velocity", Belief(
            mean=np.array([0.0, 0.0]), 
            variance=np.array([0.5, 0.5])
        ))
        
        print(f"Created beliefs: {list(beliefs.get_all_beliefs().keys())}")
        
        # Sample from beliefs
        pos_samples = beliefs.get_belief("position").sample(5)
        print(f"Position samples: {pos_samples[:3]}")
        
        # Compute entropy
        entropy = beliefs.total_entropy()
        print(f"Total entropy: {entropy:.4f}")
        
        confidence = beliefs.average_confidence()
        print(f"Average confidence: {confidence:.4f}")
        
        print("✅ Belief demo completed!")
        
    except Exception as e:
        print(f"❌ Belief demo failed: {e}")
        import traceback
        traceback.print_exc()


def run_model_demo():
    """Demonstrate generative model functionality."""
    print("\n🔮 Generative Model Demo")
    print("=" * 35)
    
    try:
        # Create model
        model = GenerativeModel(state_dim=3, obs_dim=3, action_dim=2)
        
        print(f"Model created: {model.model_name}")
        print(f"Dimensions: state={model.state_dim}, obs={model.obs_dim}, action={model.action_dim}")
        
        # Get priors
        priors = model.get_all_priors()
        print(f"Available priors: {list(priors.keys())}")
        
        # Sample from prior
        state_samples = model.sample_prior("state", n_samples=3)
        print(f"State samples: {state_samples}")
        
        # Test likelihood
        test_state = np.array([0.0, 0.0, 0.0])
        test_obs = np.array([0.1, -0.1, 0.0])
        likelihood = model.likelihood(test_state, test_obs)
        print(f"Likelihood: {likelihood:.6f}")
        
        # Test dynamics
        test_action = np.array([0.5, -0.3])
        next_state = model.predict_next_state(test_state, test_action)
        print(f"Next state prediction: {next_state}")
        
        print("✅ Model demo completed!")
        
    except Exception as e:
        print(f"❌ Model demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Active Inference Simulation Lab - Basic Examples")
    print("=" * 60)
    
    # Run individual component demos
    run_belief_demo()
    run_model_demo()
    
    # Run full agent demo
    success = run_basic_demo()
    
    if success:
        print("\n🎉 All demos completed successfully!")
        print("🔬 The Active Inference framework is working!")
    else:
        print("\n❌ Some demos failed - check implementation")
        sys.exit(1)