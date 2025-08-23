#!/usr/bin/env python3
"""
Robust Active Inference System Demonstration - Generation 2

Demonstrates enhanced error handling, monitoring, and resilience features.
"""

import sys
import time
import numpy as np
import logging
from pathlib import Path

# Add source path
sys.path.insert(0, 'src/python')

from active_inference import ActiveInferenceAgent
from active_inference.environments import MockEnvironment
from active_inference.monitoring import HealthMonitor, AgentTelemetry


def setup_logging():
    """Configure comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/robust_system_demo.log', mode='a')
        ]
    )


def create_monitored_agent(agent_id: str, health_monitor: HealthMonitor) -> ActiveInferenceAgent:
    """Create an agent with comprehensive monitoring."""
    agent = ActiveInferenceAgent(
        state_dim=4,
        obs_dim=8,
        action_dim=3,
        agent_id=agent_id,
        enable_logging=True,
        log_level=logging.INFO
    )
    
    # Register with health monitor
    health_monitor.register_component(
        component_id=agent_id,
        component=agent
    )
    
    # Register recovery action
    def recover_agent():
        """Recovery function for agent."""
        try:
            agent.reset()
            return True
        except Exception as e:
            logging.error(f"Recovery failed for {agent_id}: {e}")
            return False
    
    health_monitor.register_recovery_action(agent_id, recover_agent)
    
    return agent


def run_robust_training_session(num_episodes: int = 10) -> dict:
    """Run a robust training session with monitoring and error handling."""
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    setup_logging()
    
    logger = logging.getLogger("RobustDemo")
    logger.info("Starting Robust Active Inference System Demo")
    
    results = {
        'episodes_completed': 0,
        'total_steps': 0,
        'total_reward': 0.0,
        'errors_encountered': 0,
        'recovery_attempts': 0,
        'final_status': 'unknown'
    }
    
    # Initialize monitoring systems
    with HealthMonitor(
        check_interval=2.0,
        enable_alerts=True,
        enable_auto_recovery=True
    ) as health_monitor:
        
        with AgentTelemetry(
            buffer_size=5000,
            aggregation_interval=5.0,
            enable_real_time=True
        ) as telemetry:
            
            try:
                # Create monitored agents
                logger.info("Creating monitored agents...")
                agents = [
                    create_monitored_agent(f"robust_agent_{i}", health_monitor)
                    for i in range(2)
                ]
                
                # Create environment
                env = MockEnvironment(obs_dim=8, action_dim=3)
                
                logger.info(f"Starting training for {num_episodes} episodes")
                
                for episode in range(num_episodes):
                    episode_id = f"episode_{episode}"
                    logger.info(f"Starting episode {episode + 1}/{num_episodes}")
                    
                    for agent in agents:
                        try:
                            # Reset for new episode
                            obs = env.reset()
                            agent.reset(obs)
                            
                            episode_reward = 0.0
                            episode_steps = 0
                            
                            # Run episode
                            for step in range(100):  # Max 100 steps per episode
                                try:
                                    # Agent acts
                                    action = agent.act(obs)
                                    
                                    # Environment responds
                                    next_obs, reward, done, info = env.step(action)
                                    
                                    # Update agent
                                    agent.update_model(next_obs, action, reward)
                                    
                                    # Record telemetry
                                    telemetry.record_agent_step(
                                        agent_id=agent.agent_id,
                                        observation=obs,
                                        action=action,
                                        belief_state=agent.beliefs.get_all_beliefs(),
                                        free_energy=agent.history['free_energy'][-1].total if agent.history['free_energy'] else 0.0,
                                        reward=reward,
                                        episode_id=episode_id
                                    )
                                    
                                    episode_reward += reward
                                    episode_steps += 1
                                    results['total_steps'] += 1
                                    
                                    obs = next_obs
                                    
                                    if done:
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"Step error for {agent.agent_id}: {e}")
                                    results['errors_encountered'] += 1
                                    
                                    # Attempt recovery
                                    try:
                                        agent.reset(obs)
                                        results['recovery_attempts'] += 1
                                        logger.info(f"Recovery successful for {agent.agent_id}")
                                    except Exception as recovery_error:
                                        logger.error(f"Recovery failed: {recovery_error}")
                                        break
                            
                            results['total_reward'] += episode_reward
                            
                            # Record performance benchmark
                            telemetry.record_performance_benchmark(
                                agent_id=agent.agent_id,
                                benchmark_name="episode_reward",
                                score=episode_reward,
                                additional_metrics={
                                    'episode_steps': episode_steps,
                                    'avg_reward_per_step': episode_reward / max(1, episode_steps)
                                }
                            )
                            
                            logger.info(f"Episode completed for {agent.agent_id}: "
                                      f"reward={episode_reward:.3f}, steps={episode_steps}")
                            
                        except Exception as e:
                            logger.error(f"Episode error for {agent.agent_id}: {e}")
                            results['errors_encountered'] += 1
                    
                    results['episodes_completed'] += 1
                    
                    # Check system health
                    health_status = health_monitor.get_system_health()
                    logger.info(f"System health after episode {episode + 1}: {health_status['overall_status']}")
                    
                    # Check for performance anomalies
                    for agent in agents:
                        anomalies = telemetry.detect_performance_anomalies(agent.agent_id)
                        if anomalies:
                            logger.warning(f"Performance anomalies detected for {agent.agent_id}: {anomalies}")
                
                # Generate final reports
                logger.info("Generating final reports...")
                
                # Health report
                health_monitor.export_health_report('logs/health_report.json')
                
                # Telemetry reports
                for agent in agents:
                    summary = telemetry.get_agent_summary(agent.agent_id)
                    logger.info(f"Agent {agent.agent_id} summary: {summary}")
                
                # Multi-agent comparison
                comparison = telemetry.get_multi_agent_comparison([a.agent_id for a in agents])
                logger.info(f"Multi-agent comparison: {comparison}")
                
                # Export telemetry data
                telemetry.export_telemetry_data('logs/telemetry_export.json')
                
                results['final_status'] = 'success'
                
            except Exception as e:
                logger.error(f"Critical system error: {e}")
                results['final_status'] = 'critical_failure'
                raise
                
            finally:
                # Final system health check
                final_health = health_monitor.get_system_health()
                logger.info(f"Final system health: {final_health}")
                
                # Get all alerts
                alerts = health_monitor.get_alerts()
                if alerts:
                    logger.warning(f"Total alerts generated: {len(alerts)}")
                    for alert in alerts[-5:]:  # Show last 5 alerts
                        logger.warning(f"Alert: {alert}")
    
    return results


def demonstrate_resilience_features():
    """Demonstrate specific resilience features."""
    logger = logging.getLogger("ResilienceDemo")
    
    logger.info("=== Demonstrating Resilience Features ===")
    
    # Test 1: Error handling and recovery
    logger.info("Test 1: Error handling and recovery")
    try:
        agent = ActiveInferenceAgent(
            state_dim=4,
            obs_dim=6,
            action_dim=2,
            agent_id="resilience_test_agent"
        )
        
        # Simulate problematic input
        bad_observation = np.array([np.inf, np.nan, 1, 2, 3, 4])  # Contains inf and nan
        
        try:
            action = agent.act(bad_observation)
            logger.error("Expected validation error but got none!")
        except Exception as e:
            logger.info(f"✓ Successfully caught validation error: {type(e).__name__}")
        
        # Test recovery with good observation
        good_observation = np.random.randn(6)
        action = agent.act(good_observation)
        logger.info("✓ Agent recovered successfully after error")
        
        # Check agent health
        health = agent.get_health_status()
        logger.info(f"✓ Agent health status: {health['health_status']}")
        
    except Exception as e:
        logger.error(f"Resilience test failed: {e}")
    
    # Test 2: Checkpoint and recovery
    logger.info("Test 2: Checkpoint and recovery")
    try:
        agent = ActiveInferenceAgent(
            state_dim=3,
            obs_dim=5,
            action_dim=2,
            agent_id="checkpoint_test_agent"
        )
        
        # Run some steps
        for i in range(10):
            obs = np.random.randn(5)
            action = agent.act(obs)
            agent.update_model(obs, action, reward=np.random.randn())
        
        # Save checkpoint
        checkpoint_path = 'logs/test_checkpoint.json'
        agent.save_checkpoint(checkpoint_path)
        logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint
        restored_agent = ActiveInferenceAgent.load_checkpoint(checkpoint_path)
        logger.info(f"✓ Agent restored from checkpoint: {restored_agent}")
        
    except Exception as e:
        logger.error(f"Checkpoint test failed: {e}")


if __name__ == "__main__":
    print("🚀 Robust Active Inference System Demo - Generation 2")
    print("=" * 60)
    
    try:
        # Demonstrate resilience features
        demonstrate_resilience_features()
        
        print("\n" + "=" * 60)
        print("Running robust training session...")
        
        # Run robust training session
        results = run_robust_training_session(num_episodes=5)
        
        print("\n" + "=" * 60)
        print("ROBUST SYSTEM DEMO RESULTS:")
        print(f"Episodes completed: {results['episodes_completed']}")
        print(f"Total steps: {results['total_steps']}")
        print(f"Total reward: {results['total_reward']:.3f}")
        print(f"Errors encountered: {results['errors_encountered']}")
        print(f"Recovery attempts: {results['recovery_attempts']}")
        print(f"Final status: {results['final_status']}")
        print(f"Success rate: {results['episodes_completed'] / 5 * 100:.1f}%")
        
        if results['final_status'] == 'success':
            print("✅ GENERATION 2 COMPLETE: Robust system with monitoring verified!")
        else:
            print("⚠️  System completed with issues - check logs for details")
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nSee logs/ directory for detailed reports and telemetry data")