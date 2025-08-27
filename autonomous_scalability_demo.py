#!/usr/bin/env python3
"""
AUTONOMOUS SDLC GENERATION 3: SCALABILITY DEMONSTRATION
Real-time performance optimization and adaptive scaling

Demonstrates:
- Concurrent active inference processing
- Adaptive performance optimization
- Memory-efficient belief computation
- Real-time scaling decisions
"""

import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import multiprocessing as mp
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

# Add source to path for local testing
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from active_inference.core.agent import ActiveInferenceAgent
from active_inference.scalability.auto_scaling import ResourceMetrics as ScalingMetrics, ScalingRule as ScalingPolicy
from active_inference.performance.optimization import OptimizationConfig, OptimizedActiveInferenceAgent
from active_inference.environments.mock_env import MockEnvironment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass 
class ScalabilityBenchmark:
    """Scalability benchmark configuration."""
    num_agents: int = 10
    num_environments: int = 5
    steps_per_agent: int = 100
    concurrent_processing: bool = True
    state_dims: List[int] = None
    
    def __post_init__(self):
        if self.state_dims is None:
            self.state_dims = [4, 8, 16, 32, 64]


class AutonomousScalabilityEngine:
    """Generation 3 Scalability Engine with adaptive optimization."""
    
    def __init__(self, benchmark: ScalabilityBenchmark):
        self.benchmark = benchmark
        self.performance_metrics = {}
        self.optimization_history = []
        self.scaling_policies = []
        
        # Initialize scaling policies
        self._setup_scaling_policies()
    
    def _setup_scaling_policies(self):
        """Setup auto-scaling policies for different metrics."""
        from active_inference.scalability.auto_scaling import ScalingTrigger
        
        policies = [
            ScalingPolicy(
                name="cpu_policy",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=10
            ),
            ScalingPolicy(
                name="memory_policy", 
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=40.0,
                min_instances=1,
                max_instances=8
            ),
            ScalingPolicy(
                name="inference_latency_policy",
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=100.0,  # milliseconds
                scale_down_threshold=20.0,
                min_instances=1,
                max_instances=12
            )
        ]
        
        self.scaling_policies = policies
    
    def _evaluate_scaling_decision(self, metrics: ScalingMetrics) -> str:
        """Evaluate scaling decision based on policies."""
        for policy in self.scaling_policies:
            if policy.trigger.value == "cpu_utilization":
                value = metrics.cpu_percent
            elif policy.trigger.value == "memory_utilization":
                value = metrics.memory_percent
            elif policy.trigger.value == "response_time":
                value = metrics.avg_response_time
            elif policy.trigger.value == "throughput":
                value = metrics.requests_per_second
            else:
                continue
                
            if value > policy.scale_up_threshold:
                return f"scale_up_{policy.name}"
            elif value < policy.scale_down_threshold:
                return f"scale_down_{policy.name}"
        
        return "no_action"
    
    def create_optimized_agent(self, state_dim: int, optimization_level: str = "balanced") -> OptimizedActiveInferenceAgent:
        """Create performance-optimized agent."""
        
        # Adaptive optimization config based on problem size
        if state_dim <= 8:
            config = OptimizationConfig(
                use_gpu=False,
                enable_caching=True,
                parallel_belief_updates=False,
                batch_size=16,
                optimization_level="speed"
            )
        elif state_dim <= 32:
            config = OptimizationConfig(
                use_gpu=False,
                enable_caching=True,
                parallel_belief_updates=True,
                batch_size=32,
                optimization_level="balanced"
            )
        else:
            config = OptimizationConfig(
                use_gpu=True,  # Enable GPU for large problems
                enable_caching=True,
                parallel_belief_updates=True,
                vectorized_planning=True,
                batch_size=64,
                optimization_level="memory"
            )
        
        agent = OptimizedActiveInferenceAgent(
            state_dim=state_dim,
            obs_dim=state_dim * 2,
            action_dim=2,
            optimization_config=config,
            enable_logging=False  # Disable logging for performance
        )
        
        return agent
    
    def benchmark_single_agent_performance(self, state_dim: int, steps: int = 100) -> Dict:
        """Benchmark single agent performance."""
        
        agent = self.create_optimized_agent(state_dim)
        env = MockEnvironment(obs_dim=state_dim * 2, action_dim=2)
        
        # Warm up
        obs = env.reset()
        for _ in range(5):
            action = agent.act(obs)
            step_result = env.step(action)
            obs = step_result[0]  # Get observation from tuple
        
        # Actual benchmark
        start_time = time.time()
        inference_times = []
        
        obs = env.reset()
        for step in range(steps):
            inference_start = time.time()
            action = agent.act(obs)
            inference_time = (time.time() - inference_start) * 1000  # ms
            inference_times.append(inference_time)
            
            step_result = env.step(action)
            obs, reward = step_result[0], step_result[1]
            done = step_result[2] if len(step_result) > 2 else False
            if done:
                obs = env.reset()
        
        total_time = time.time() - start_time
        
        return {
            'state_dim': state_dim,
            'total_time': total_time,
            'steps_per_second': steps / total_time,
            'avg_inference_ms': np.mean(inference_times),
            'p95_inference_ms': np.percentile(inference_times, 95),
            'min_inference_ms': np.min(inference_times),
            'max_inference_ms': np.max(inference_times),
            'memory_usage_mb': agent.get_memory_usage() if hasattr(agent, 'get_memory_usage') else 0
        }
    
    def benchmark_concurrent_agents(self, num_agents: int, state_dim: int, steps: int = 100) -> Dict:
        """Benchmark concurrent agent performance."""
        
        def run_agent(agent_id: int) -> Dict:
            try:
                agent = self.create_optimized_agent(state_dim)
                env = MockEnvironment(obs_dim=state_dim * 2, action_dim=2)
                
                start_time = time.time()
                obs = env.reset()
                
                for step in range(steps):
                    action = agent.act(obs)
                    step_result = env.step(action)
                    obs, reward = step_result[0], step_result[1]
                    done = step_result[2] if len(step_result) > 2 else False
                    if done:
                        obs = env.reset()
                
                total_time = time.time() - start_time
                
                return {
                    'agent_id': agent_id,
                    'total_time': total_time,
                    'steps_per_second': steps / total_time,
                    'success': True
                }
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
                return {
                    'agent_id': agent_id,
                    'error': str(e),
                    'success': False
                }
        
        # Run agents concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(num_agents, mp.cpu_count())) as executor:
            futures = [executor.submit(run_agent, i) for i in range(num_agents)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        successful_agents = [r for r in results if r.get('success', False)]
        
        if not successful_agents:
            return {'error': 'All agents failed', 'results': results}
        
        total_steps = len(successful_agents) * steps
        aggregate_steps_per_second = sum(r['steps_per_second'] for r in successful_agents)
        
        return {
            'num_agents': num_agents,
            'state_dim': state_dim,
            'total_time': total_time,
            'successful_agents': len(successful_agents),
            'failed_agents': num_agents - len(successful_agents),
            'total_steps': total_steps,
            'aggregate_steps_per_second': aggregate_steps_per_second,
            'avg_steps_per_second_per_agent': aggregate_steps_per_second / len(successful_agents),
            'parallel_efficiency': (total_steps / total_time) / (steps / np.mean([r['total_time'] for r in successful_agents])),
            'results': results
        }
    
    def run_adaptive_scaling_benchmark(self) -> Dict:
        """Run comprehensive scaling benchmark with adaptive optimization."""
        
        logger.info("🚀 Starting Generation 3 Scalability Benchmark")
        
        benchmark_results = {
            'timestamp': time.time(),
            'single_agent_benchmarks': [],
            'concurrent_benchmarks': [],
            'scaling_decisions': [],
            'optimization_adaptations': []
        }
        
        # 1. Single agent performance across state dimensions
        logger.info("📊 Benchmarking single agent performance")
        for state_dim in self.benchmark.state_dims:
            logger.info(f"   Testing state dimension: {state_dim}")
            result = self.benchmark_single_agent_performance(state_dim, self.benchmark.steps_per_agent)
            benchmark_results['single_agent_benchmarks'].append(result)
            
            # Update auto-scaler with metrics
            metrics = ScalingMetrics(
                cpu_percent=min(result['avg_inference_ms'] / 10, 100),  # Mock CPU usage
                memory_percent=min(result.get('memory_usage_mb', 0) / 10, 100),
                memory_available_gb=8.0,
                active_instances=1,
                queue_depth=0,
                avg_response_time=result['avg_inference_ms'],
                requests_per_second=result['steps_per_second'],
                error_rate=0.0,
                timestamp=time.time()
            )
            
            # Evaluate scaling decisions based on policies
            scaling_decision = self._evaluate_scaling_decision(metrics)
            if scaling_decision != "no_action":
                benchmark_results['scaling_decisions'].append({
                    'state_dim': state_dim,
                    'decision': scaling_decision,
                    'metrics': metrics.__dict__
                })
        
        # 2. Concurrent agent performance
        logger.info("🔄 Benchmarking concurrent agent performance")
        for state_dim in [4, 16, 64]:  # Test subset for concurrent processing
            for num_agents in [2, 5, 10]:
                if num_agents <= self.benchmark.num_agents:
                    logger.info(f"   Testing {num_agents} agents with state dim {state_dim}")
                    result = self.benchmark_concurrent_agents(num_agents, state_dim, 50)  # Fewer steps for concurrent
                    benchmark_results['concurrent_benchmarks'].append(result)
        
        # 3. Generate performance report
        self._generate_performance_report(benchmark_results)
        
        return benchmark_results
    
    def _generate_performance_report(self, results: Dict):
        """Generate detailed performance report."""
        
        logger.info("\n" + "="*80)
        logger.info("🎯 GENERATION 3 SCALABILITY REPORT")
        logger.info("="*80)
        
        # Single agent performance
        if results['single_agent_benchmarks']:
            logger.info("\n📊 SINGLE AGENT PERFORMANCE:")
            for result in results['single_agent_benchmarks']:
                logger.info(f"   State Dim {result['state_dim']}: "
                          f"{result['steps_per_second']:.1f} steps/sec, "
                          f"avg {result['avg_inference_ms']:.2f}ms inference")
        
        # Find best performing configuration
        if results['single_agent_benchmarks']:
            best_throughput = max(results['single_agent_benchmarks'], key=lambda x: x['steps_per_second'])
            best_latency = min(results['single_agent_benchmarks'], key=lambda x: x['avg_inference_ms'])
            
            logger.info(f"\n🏆 BEST PERFORMANCE:")
            logger.info(f"   Highest Throughput: {best_throughput['steps_per_second']:.1f} steps/sec "
                       f"(state_dim={best_throughput['state_dim']})")
            logger.info(f"   Lowest Latency: {best_latency['avg_inference_ms']:.2f}ms "
                       f"(state_dim={best_latency['state_dim']})")
        
        # Concurrent performance
        if results['concurrent_benchmarks']:
            logger.info(f"\n🔄 CONCURRENT PROCESSING:")
            for result in results['concurrent_benchmarks']:
                if 'error' not in result:
                    efficiency = result.get('parallel_efficiency', 0) * 100
                    logger.info(f"   {result['num_agents']} agents (dim {result['state_dim']}): "
                              f"{result['aggregate_steps_per_second']:.1f} total steps/sec, "
                              f"{efficiency:.1f}% parallel efficiency")
        
        # Scaling decisions
        if results['scaling_decisions']:
            logger.info(f"\n⚖️  AUTO-SCALING DECISIONS:")
            for decision in results['scaling_decisions']:
                logger.info(f"   State dim {decision['state_dim']}: {decision['decision']}")
        
        logger.info("\n✅ Generation 3 Scalability Implementation Complete!")
        logger.info("="*80 + "\n")


def main():
    """Main scalability demonstration."""
    
    # Configure benchmark
    benchmark_config = ScalabilityBenchmark(
        num_agents=10,
        steps_per_agent=100,
        concurrent_processing=True,
        state_dims=[4, 8, 16, 32, 64, 128]  # Test up to 128 state dimensions
    )
    
    # Create scalability engine
    scalability_engine = AutonomousScalabilityEngine(benchmark_config)
    
    # Run comprehensive benchmark
    start_time = time.time()
    results = scalability_engine.run_adaptive_scaling_benchmark()
    total_time = time.time() - start_time
    
    # Save results
    import json
    timestamp = int(time.time())
    results_file = f"generation_3_scalability_results_{timestamp}.json"
    
    # Make results JSON serializable
    serializable_results = {}
    for key, value in results.items():
        if key == 'timestamp':
            serializable_results[key] = value
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"📁 Results saved to: {results_file}")
    logger.info(f"⏱️  Total benchmark time: {total_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    results = main()