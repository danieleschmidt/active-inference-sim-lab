#!/usr/bin/env python3
"""
Example usage of Active Inference Simulation Laboratory.

This script demonstrates how to create and use an active inference agent
with the implemented framework.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

# Import active inference components
from active_inference import (
    ActiveInferenceAgent,
    MockEnvironment
)
from active_inference.utils.logging import setup_logging

def main():
    print("🧠 Active Inference Simulation Laboratory - Example Usage")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Create environment
    print("\n🌍 Creating Environment...")
    env = MockEnvironment(obs_dim=4, action_dim=2, episode_length=20)
    print(f"✅ Environment created with {env.obs_dim}D observations, {env.action_dim}D actions")
    
    # Create agent
    print("\n🤖 Creating Active Inference Agent...")
    agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=4,
        action_dim=2,
        inference_method="variational",
        planning_horizon=3,
        learning_rate=0.01,
        temperature=0.5,
        agent_id="example_agent"
    )
    print(f"✅ Agent created: {agent}")
    
    # Run episode
    print("\n🎮 Running Episode...")
    
    obs = env.reset()
    agent.reset(obs)
    
    episode_reward = 0.0
    episode_length = 0
    
    while True:
        # Agent perceives and acts
        action = agent.act(obs)
        
        # Environment responds
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Agent learns from experience  
        agent.update_model(next_obs, action, reward)
        
        episode_reward += reward
        episode_length += 1
        
        print(f"  Step {episode_length}: Action={action[:2]:.3f}, Reward={reward:.3f}")
        
        if terminated or truncated:
            break
            
        obs = next_obs
    
    # Show results
    print(f"\n📊 Episode Results:")
    print(f"  Length: {episode_length} steps")
    print(f"  Total reward: {episode_reward:.3f}")
    print(f"  Average reward: {episode_reward/episode_length:.3f}")
    
    # Agent statistics
    stats = agent.get_statistics()
    print(f"\n🧠 Agent Statistics:")
    print(f"  Belief confidence: {stats['belief_confidence']:.3f}")
    print(f"  Belief entropy: {stats['belief_entropy']:.3f}")
    print(f"  Current free energy: {stats['current_free_energy']:.3f}")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"  ✅ Active inference agent with belief updating")
    print(f"  ✅ Free energy minimization")
    print(f"  ✅ Action planning with expected free energy")
    print(f"  ✅ Environment interaction")
    print(f"  ✅ Learning from experience")
    print(f"  ✅ Performance monitoring and logging")
    
    print(f"\n🚀 Active Inference Simulation Laboratory is working!")

if __name__ == "__main__":
    main()