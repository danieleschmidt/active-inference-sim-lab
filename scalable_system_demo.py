#!/usr/bin/env python3
"""
Scalable Active Inference System Demonstration - Generation 3

Demonstrates high-performance scaling, parallel processing, and multi-agent orchestration.
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
from active_inference.performance.parallel_processing import ParallelInferenceEngine, AdaptiveBatchProcessor
from active_inference.scalability.multi_agent_orchestrator import MultiAgentOrchestrator


def setup_logging():
    """Configure comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/scalable_system_demo.log', mode='a')
        ]
    )


def create_environment_factory():
    """Factory function for creating environment instances."""
    def factory():
        return MockEnvironment(obs_dim=6, action_dim=3)
    return factory


def demonstrate_parallel_inference():
    """Demonstrate parallel inference capabilities."""
    print("=" * 50)
    print("PARALLEL INFERENCE DEMONSTRATION")
    print("=" * 50)
    
    with ParallelInferenceEngine(max_workers=8, batch_size=16, use_gpu=False) as engine:
        # Create test data
        n_agents = 5
        observations = [np.random.randn(6) for _ in range(n_agents)]
        
        # Mock beliefs and models
        beliefs = [{'mean': np.random.randn(4), 'variance': np.ones(4) * 0.1} for _ in range(n_agents)]
        models = [type('MockModel', (), {'action_dim': 3}) for _ in range(n_agents)]
        
        start_time = time.time()
        
        # Parallel belief updates
        updated_beliefs = engine.parallel_belief_update(observations, beliefs, models)
        
        # Parallel planning
        actions = engine.parallel_planning(updated_beliefs, models, [5] * n_agents)
        
        end_time = time.time()
        
        print(f"✓ Processed {n_agents} agents in parallel")
        print(f"✓ Total processing time: {end_time - start_time:.3f}s")
        print(f"✓ Performance metrics: {engine.get_performance_metrics()}")
        
        # Batch inference demonstration
        batch_observations = np.random.randn(32, 6)
        batch_results = engine.batch_inference(batch_observations, models[0])
        print(f"✓ Batch inference: {batch_observations.shape} -> {batch_results.shape}")


def demonstrate_adaptive_batching():
    """Demonstrate adaptive batch processing."""
    print("\n" + "=" * 50)
    print("ADAPTIVE BATCH PROCESSING DEMONSTRATION")
    print("=" * 50)
    
    processor = AdaptiveBatchProcessor(initial_batch_size=8, adaptation_rate=0.2)
    
    # Simulate processing function with varying computational load
    def process_batch(batch_data):
        # Simulate variable processing time
        processing_time = len(batch_data) * np.random.uniform(0.01, 0.02)
        time.sleep(processing_time)
        return [f"processed_{item}" for item in batch_data]
    
    # Process data with adaptive batching
    test_data = list(range(100))
    
    start_time = time.time()
    results = processor.process_batch(test_data, process_batch)
    end_time = time.time()
    
    metrics = processor.get_metrics()
    print(f"✓ Processed {len(test_data)} items in {end_time - start_time:.3f}s")
    print(f"✓ Final batch size: {metrics['current_batch_size']}")
    print(f"✓ Average throughput: {metrics['avg_throughput']:.2f} items/s")
    print(f"✓ Total batches: {metrics['total_batches_processed']}")


def demonstrate_multi_agent_orchestration():
    """Demonstrate multi-agent orchestration."""
    print("\n" + "=" * 50)
    print("MULTI-AGENT ORCHESTRATION DEMONSTRATION")
    print("=" * 50)
    
    with MultiAgentOrchestrator(
        max_agents=20,
        enable_communication=True,
        enable_load_balancing=True
    ) as orchestrator:
        
        # Create and register multiple agents
        agents = []
        for i in range(8):
            agent = ActiveInferenceAgent(
                state_dim=4,
                obs_dim=6,
                action_dim=3,
                agent_id=f"scalable_agent_{i}",
                enable_logging=False  # Reduce log noise
            )
            agents.append(agent)
            
            # Register with orchestrator
            group = "group_A" if i < 4 else "group_B"
            orchestrator.register_agent(f"scalable_agent_{i}", agent, group=group)
        
        print(f"✓ Registered {len(agents)} agents")
        
        # Check orchestrator status
        status = orchestrator.get_orchestrator_status()
        print(f"✓ Orchestrator status: {status['total_agents']} agents active")
        print(f"✓ Groups: {status['groups']}")
        
        # Execute parallel episodes
        env_factory = create_environment_factory()
        
        start_time = time.time()
        results = orchestrator.execute_parallel_episodes(
            environment_factory=env_factory,
            num_episodes=3,
            episode_length=50
        )
        end_time = time.time()
        
        print(f"✓ Parallel execution completed in {end_time - start_time:.2f}s")
        print(f"✓ Success rate: {results['success_rate']:.1f}%")
        print(f"✓ Total episodes: {results['total_episodes']}")
        print(f"✓ Average reward per episode: {results['avg_reward_per_episode']:.3f}")
        print(f"✓ Best episode reward: {results['best_episode_reward']:.3f}")
        
        # Test communication
        messages_sent = orchestrator.broadcast_message(
            sender_id="scalable_agent_0",
            message_type="performance_update",
            content={"metric": "test", "value": 1.0}
        )
        print(f"✓ Broadcast message to {messages_sent} agents")
        
        # Final status check
        final_status = orchestrator.get_orchestrator_status()
        print(f"✓ Final status: {final_status['total_messages_processed']} messages processed")


def run_comprehensive_scalability_test():
    """Run comprehensive scalability test combining all features."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SCALABILITY TEST")
    print("=" * 60)
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    setup_logging()
    
    logger = logging.getLogger("ScalabilityTest")
    
    test_results = {
        'start_time': time.time(),
        'agents_tested': 0,
        'episodes_completed': 0,
        'total_computations': 0,
        'peak_throughput': 0.0,
        'success_rate': 0.0
    }
    
    try:
        # Initialize scalability systems
        with ParallelInferenceEngine(
            max_workers=16, batch_size=32, use_gpu=False, enable_caching=True
        ) as parallel_engine:
            
            with MultiAgentOrchestrator(
                max_agents=50, enable_communication=True, enable_load_balancing=True
            ) as orchestrator:
                
                logger.info("Starting comprehensive scalability test")
                
                # Create larger agent population
                num_agents = 12
                agents = []
                
                for i in range(num_agents):
                    agent = ActiveInferenceAgent(
                        state_dim=6,
                        obs_dim=8,
                        action_dim=4,
                        agent_id=f"scale_test_agent_{i}",
                        enable_logging=False
                    )
                    agents.append(agent)
                    
                    # Assign to groups for coordination
                    group = f"performance_group_{i % 3}"
                    orchestrator.register_agent(agent.agent_id, agent, group=group)
                
                test_results['agents_tested'] = num_agents
                logger.info(f"Created {num_agents} agents for scalability testing")
                
                # Phase 1: Parallel inference testing
                logger.info("Phase 1: Testing parallel inference performance")
                
                # Create batch test data
                batch_observations = [np.random.randn(8) for _ in range(num_agents * 10)]
                batch_beliefs = [
                    {'mean': np.random.randn(6), 'variance': np.ones(6) * 0.1}
                    for _ in range(num_agents * 10)
                ]
                batch_models = [agents[0]] * (num_agents * 10)  # Reuse model
                
                inference_start = time.time()
                parallel_results = parallel_engine.parallel_belief_update(
                    batch_observations, batch_beliefs, batch_models
                )
                inference_time = time.time() - inference_start
                
                inference_throughput = len(batch_observations) / inference_time
                test_results['peak_throughput'] = max(test_results['peak_throughput'], inference_throughput)
                
                logger.info(f"Parallel inference: {len(batch_observations)} updates "
                           f"in {inference_time:.3f}s ({inference_throughput:.1f} updates/s)")
                
                # Phase 2: Multi-agent coordination testing
                logger.info("Phase 2: Testing multi-agent coordination")
                
                orchestration_start = time.time()
                coordination_results = orchestrator.execute_parallel_episodes(
                    environment_factory=create_environment_factory(),
                    num_episodes=5,
                    episode_length=100
                )
                orchestration_time = time.time() - orchestration_start
                
                test_results['episodes_completed'] = coordination_results['total_episodes']
                test_results['success_rate'] = coordination_results['success_rate']
                
                logger.info(f"Multi-agent orchestration: {coordination_results['total_episodes']} episodes "
                           f"in {orchestration_time:.3f}s (success rate: {coordination_results['success_rate']:.1f}%)")
                
                # Phase 3: Load testing with adaptive batching
                logger.info("Phase 3: Load testing with adaptive processing")
                
                processor = AdaptiveBatchProcessor(initial_batch_size=16, adaptation_rate=0.15)
                
                # Simulate high-load scenario
                load_test_data = list(range(500))
                
                def simulate_inference_load(batch):
                    # Simulate varying computational load
                    load_factor = np.random.uniform(0.8, 1.2)
                    time.sleep(len(batch) * 0.001 * load_factor)
                    return [f"result_{item}" for item in batch]
                
                load_start = time.time()
                load_results = processor.process_batch(load_test_data, simulate_inference_load)
                load_time = time.time() - load_start
                
                load_metrics = processor.get_metrics()
                load_throughput = len(load_test_data) / load_time
                
                logger.info(f"Load test: {len(load_test_data)} items in {load_time:.3f}s "
                           f"({load_throughput:.1f} items/s, final batch size: {load_metrics['current_batch_size']})")
                
                # Phase 4: System integration test
                logger.info("Phase 4: Full system integration test")
                
                integration_start = time.time()
                
                # Combine all systems for integrated test
                integration_observations = np.random.randn(num_agents * 5, 8)
                integration_batch_results = parallel_engine.batch_inference(
                    integration_observations, agents[0]
                )
                
                # Final coordination round
                final_coordination = orchestrator.coordinate_multi_agent_learning(
                    learning_task="shared_experience",
                    coordination_frequency=50
                )
                
                integration_time = time.time() - integration_start
                
                logger.info(f"Integration test completed in {integration_time:.3f}s")
                
                # Collect final performance metrics
                parallel_metrics = parallel_engine.get_performance_metrics()
                orchestrator_status = orchestrator.get_orchestrator_status()
                
                test_results.update({
                    'total_computations': parallel_metrics['total_computations'],
                    'cache_hit_rate': parallel_metrics['cache_hit_rate_percent'],
                    'final_orchestrator_status': orchestrator_status['agents_by_status'],
                    'total_messages': orchestrator_status['total_messages_processed']
                })
                
                logger.info("Comprehensive scalability test completed successfully")
                
    except Exception as e:
        logger.error(f"Scalability test failed: {e}")
        raise
    
    finally:
        test_results['total_time'] = time.time() - test_results['start_time']
    
    return test_results


if __name__ == "__main__":
    print("🚀 Scalable Active Inference System Demo - Generation 3")
    print("=" * 70)
    
    try:
        # Individual component demonstrations
        demonstrate_parallel_inference()
        demonstrate_adaptive_batching()
        demonstrate_multi_agent_orchestration()
        
        # Comprehensive test
        print("\n" + "=" * 70)
        print("Running comprehensive scalability test...")
        
        results = run_comprehensive_scalability_test()
        
        print("\n" + "=" * 70)
        print("SCALABLE SYSTEM DEMO RESULTS:")
        print(f"Total execution time: {results['total_time']:.2f}s")
        print(f"Agents tested: {results['agents_tested']}")
        print(f"Episodes completed: {results['episodes_completed']}")
        print(f"Total computations: {results['total_computations']}")
        print(f"Peak throughput: {results['peak_throughput']:.1f} operations/s")
        print(f"Success rate: {results['success_rate']:.1f}%")
        print(f"Cache hit rate: {results.get('cache_hit_rate', 0):.1f}%")
        print(f"Messages processed: {results.get('total_messages', 0)}")
        
        if results['success_rate'] >= 80 and results['peak_throughput'] > 100:
            print("✅ GENERATION 3 COMPLETE: Scalable system with high-performance features verified!")
        else:
            print("⚠️  System completed but performance targets not fully met")
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nSee logs/ directory for detailed performance logs")