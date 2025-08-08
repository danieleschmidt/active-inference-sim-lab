#!/usr/bin/env python3
"""
Basic Active Inference Agent Demonstration

This example shows how to create and use a basic active inference agent
with different environments and inference methods.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

from active_inference import (
    ActiveInferenceAgent,
    MockEnvironment,
    ActiveInferenceGridWorld,
    SocialDilemmaEnvironment
)
from active_inference.utils.logging import setup_logging


def demo_basic_agent():
    """Demonstrate basic agent functionality."""
    print("🧠 Active Inference Agent - Basic Demo")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Create environment
    env = MockEnvironment(obs_dim=4, action_dim=2, episode_length=50)
    print(f"📍 Environment: {env.__class__.__name__}")
    
    # Create agent
    agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=4,
        action_dim=2,
        inference_method="variational",
        planning_horizon=3,
        learning_rate=0.01,
        temperature=0.5,
        agent_id="demo_agent"
    )
    print(f"🤖 Agent: {agent}")
    
    # Run episodes
    total_reward = 0.0
    episodes = 5
    
    for episode in range(episodes):
        print(f"\n📊 Episode {episode + 1}/{episodes}")
        
        # Reset environment and agent
        obs = env.reset()
        agent.reset(obs)
        
        episode_reward = 0.0
        step_count = 0
        
        while True:
            # Agent acts
            action = agent.act(obs)
            
            # Environment responds
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Agent learns
            agent.update_model(next_obs, action, reward)
            
            episode_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                stats = agent.get_statistics()
                print(f"  Step {step_count}: Reward={reward:.3f}, "
                      f"Free Energy={stats.get('current_free_energy', 0):.3f}")
            
            if terminated or truncated:
                break
            
            obs = next_obs
        
        total_reward += episode_reward
        print(f"✅ Episode {episode + 1} completed: {step_count} steps, "
              f"reward={episode_reward:.3f}")
    
    # Final statistics
    avg_reward = total_reward / episodes
    final_stats = agent.get_statistics()
    
    print(f"\n📈 Final Results:")
    print(f"  Episodes: {episodes}")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Agent confidence: {final_stats.get('belief_confidence', 0):.3f}")
    print(f"  Agent entropy: {final_stats.get('belief_entropy', 0):.3f}")


def demo_grid_world():
    """Demonstrate grid world environment."""
    print("\n🌍 Grid World Demonstration")
    print("=" * 50)
    
    # Create grid world
    env = ActiveInferenceGridWorld(
        size=8,
        vision_range=2,
        observation_noise=0.1,
        uncertainty_regions=[(3, 3), (5, 5)],
        goal_locations=[(6, 6)],
        dynamic_goals=False
    )
    
    # Create agent with appropriate dimensions
    vision_range = 2
    obs_dim = (2 * vision_range + 1) ** 2 * 4  # 4 channels
    
    agent = ActiveInferenceAgent(
        state_dim=8,
        obs_dim=obs_dim,
        action_dim=2,
        inference_method="variational",
        planning_horizon=5,
        learning_rate=0.005,
        temperature=0.3,
        agent_id="grid_explorer"
    )
    
    print(f"🗺️  Grid size: 8x8")
    print(f"👁️  Vision range: {vision_range}")
    print(f"🎯 Goal location: (6, 6)")
    print(f"❓ Uncertainty regions: [(3, 3), (5, 5)]")
    
    # Run episodes
    episodes = 3
    for episode in range(episodes):
        print(f"\n🎮 Episode {episode + 1}/{episodes}")
        
        obs = env.reset()
        agent.reset(obs)
        
        episode_reward = 0.0
        step_count = 0
        
        while step_count < 100:  # Max 100 steps
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update_model(next_obs, action, reward)
            
            episode_reward += reward
            step_count += 1
            
            # Show progress
            if step_count % 20 == 0:
                pos = info.get('agent_pos', [0, 0])
                distance = info.get('distance_to_goal', 0)
                print(f"  Step {step_count}: Position=({pos[0]}, {pos[1]}), "
                      f"Distance to goal={distance:.1f}")
            
            if terminated:
                print(f"🎯 Goal reached in {step_count} steps!")
                break
            
            if truncated:
                print(f"⏰ Episode truncated at {step_count} steps")
                break
            
            obs = next_obs
        
        print(f"📊 Episode reward: {episode_reward:.3f}")
        
        # Show final statistics
        env_stats = env.get_statistics()
        print(f"  Exploration ratio: {env_stats.get('exploration_ratio', 0):.3f}")
        print(f"  Goal reached: {env_stats.get('goal_reached', False)}")


def demo_social_dilemma():
    """Demonstrate social dilemma environment."""
    print("\n🤝 Social Dilemma Demonstration")
    print("=" * 50)
    
    # Create social environment
    env = SocialDilemmaEnvironment(
        n_agents=2,
        game_type="prisoners_dilemma",
        max_rounds=20,
        reputation_system=True,
        communication=False
    )
    
    # Create two agents
    agents = []
    for i in range(2):
        agent = ActiveInferenceAgent(
            state_dim=6,
            obs_dim=12,  # Complex observation space for social environment
            action_dim=1,  # Binary choice (cooperate/defect)
            inference_method="variational",
            planning_horizon=3,
            learning_rate=0.01,
            temperature=0.8,
            agent_id=f"social_agent_{i}"
        )
        agents.append(agent)
    
    print(f"👥 Agents: {len(agents)}")
    print(f"🎲 Game: Prisoner's Dilemma")
    print(f"🔄 Max rounds: 20")
    print(f"📊 Reputation system: enabled")
    
    # Run episodes
    episodes = 3
    for episode in range(episodes):
        print(f"\n🎭 Episode {episode + 1}/{episodes}")
        
        obs_list = env.reset()
        for i, agent in enumerate(agents):
            agent.reset(obs_list[i])
        
        episode_rewards = [0.0, 0.0]
        round_count = 0
        
        while True:
            # Get actions from both agents
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(obs_list[i])
                # Convert continuous action to discrete choice
                discrete_action = 0 if action[0] > 0.0 else 1  # 0=cooperate, 1=defect
                actions.append(discrete_action)
            
            # Environment step
            next_obs_list, rewards, done, info = env.step(actions)
            
            # Agent updates
            for i, agent in enumerate(agents):
                agent.update_model(next_obs_list[i], [actions[i]], rewards[i])
                episode_rewards[i] += rewards[i]
            
            round_count += 1
            
            # Show round results
            action_names = ["Cooperate", "Defect"]
            print(f"  Round {round_count}: "
                  f"Agent0={action_names[actions[0]]}, "
                  f"Agent1={action_names[actions[1]]}, "
                  f"Rewards=[{rewards[0]:.1f}, {rewards[1]:.1f}]")
            
            if done:
                break
            
            obs_list = next_obs_list
        
        # Episode summary
        total_reward = sum(episode_rewards)
        cooperation_rate = info.get('cooperation_rate', 0)
        
        print(f"📈 Episode summary:")
        print(f"  Total rewards: {episode_rewards}")
        print(f"  Social welfare: {total_reward:.1f}")
        print(f"  Cooperation rate: {cooperation_rate:.1%}")


def main():
    """Run all demonstrations."""
    print("🚀 Active Inference Simulation Laboratory - Examples")
    print("🧠 Demonstrating core capabilities of the framework")
    print("=" * 70)
    
    try:
        # Basic agent demo
        demo_basic_agent()
        
        # Grid world demo
        demo_grid_world()
        
        # Social dilemma demo  
        demo_social_dilemma()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("💡 Try modifying the parameters to see different behaviors")
        print("📖 Check the documentation for more advanced examples")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()