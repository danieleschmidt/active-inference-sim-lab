#!/usr/bin/env python3
"""Quick functionality test."""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

from active_inference import ActiveInferenceAgent, MockEnvironment

def quick_test():
    print("🧠 Quick Active Inference Test")
    
    # Create simple environment and agent
    env = MockEnvironment(obs_dim=2, action_dim=1, episode_length=5)
    agent = ActiveInferenceAgent(
        state_dim=2,
        obs_dim=2, 
        action_dim=1,
        planning_horizon=1,
        temperature=1.0
    )
    
    # Run 1 episode
    obs = env.reset()
    print(f"Initial obs: {obs}")
    
    for step in range(5):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {step+1}: action={action}, reward={reward:.3f}")
        
        if done:
            break
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    quick_test()