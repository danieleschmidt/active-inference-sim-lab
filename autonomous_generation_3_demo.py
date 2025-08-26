#!/usr/bin/env python3
"""
Autonomous SDLC Generation 3 Demo: MAKE IT SCALE

This script demonstrates the scalability features implemented in Generation 3,
including performance optimization, caching, distributed processing, and
auto-scaling capabilities for the Active Inference framework.

Author: Terragon Labs
Generation: 3 (MAKE IT SCALE - Optimized)
"""

import sys
import os
import numpy as np
import time
import json
from typing import Dict, Any, List
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src', 'python'))

# Import Generation 3 components
from active_inference.optimization.performance_optimization import (
    PerformanceOptimizedActiveInference,
    ScalableActiveInferenceFramework,
    performance_benchmark
)
from active_inference.scalability.distributed_processing import (
    DistributedActiveInferenceCluster,
    distributed_benchmark
)

# Import telemetry for monitoring
from active_inference.monitoring.comprehensive_telemetry import TelemetryCollector


class Generation3ScalabilityDemo:
    """
    Comprehensive demonstration of Generation 3 scalability features.
    """
    
    def __init__(self):
        self.results = {
            'generation': 3,
            'demo_type': 'scalability_comprehensive',
            'scalability_features_tested': 0,
            'overall_performance_score': 0.0,
            'results': {},
            'system_scalability_metrics': {},
            'recommendations': []
        }
        
        self.telemetry = TelemetryCollector()
        self.telemetry.start()
        
        self.start_time = time.time()
        print("=== AUTONOMOUS SDLC GENERATION 3: MAKE IT SCALE ===")
        print("Testing advanced scalability and performance optimization features...\n")
    
    def test_performance_optimization(self) -> Dict[str, Any]:
        """Test performance optimization features."""
        print("🚀 Testing Performance Optimization...")
        
        try:
            # Create optimized agent
            agent = PerformanceOptimizedActiveInference(state_dim=8, action_dim=4)
            
            # Run performance benchmark
            benchmark_results = performance_benchmark(agent, num_iterations=500)
            
            # Test caching effectiveness
            cache_stats = agent.cache.stats()
            memory_pool_stats = agent.memory_pool.stats()
            
            # Test adaptive optimization
            agent.optimize_configuration()
            optimization_suggestions = agent.optimizer.suggest_optimizations()
            
            # Clean up
            agent.cleanup()
            
            result = {
                'benchmark_results': benchmark_results,
                'cache_performance': cache_stats,
                'memory_pool_performance': memory_pool_stats,
                'optimization_suggestions': optimization_suggestions,
                'success': True
            }
            
            print(f"✅ Performance optimization completed in {benchmark_results['total_time']:.2f}s")
            print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"   Memory pool hit rate: {memory_pool_stats['pool_hit_rate']:.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Performance optimization test failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def test_scalable_framework(self) -> Dict[str, Any]:
        """Test scalable framework with auto-scaling."""
        print("\n📈 Testing Scalable Framework with Auto-scaling...")
        
        try:
            # Create scalable framework
            framework = ScalableActiveInferenceFramework(initial_agents=2, max_agents=8)
            
            # Generate workload to trigger scaling
            observations = [np.random.rand(8) for _ in range(100)]
            
            start_time = time.time()
            
            # Process requests concurrently to simulate load
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(framework.process_request, obs) 
                    for obs in observations
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=5.0)
                        results.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")
            
            processing_time = time.time() - start_time
            
            # Get framework statistics
            framework_stats = framework.get_framework_stats()
            
            # Test manual scaling
            scale_up_success = framework.scale_up()
            scale_down_success = framework.scale_down()
            
            # Clean up
            framework.cleanup()
            
            result = {
                'processing_time': processing_time,
                'requests_processed': len(results),
                'throughput': len(results) / processing_time if processing_time > 0 else 0,
                'framework_stats': framework_stats,
                'auto_scaling': {
                    'scale_up_success': scale_up_success,
                    'scale_down_success': scale_down_success
                },
                'success': True
            }
            
            print(f"✅ Scalable framework test completed in {processing_time:.2f}s")
            print(f"   Throughput: {result['throughput']:.1f} requests/sec")
            print(f"   Active agents: {framework_stats['total_agents']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Scalable framework test failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def test_distributed_processing(self) -> Dict[str, Any]:
        """Test distributed processing capabilities."""
        print("\n🌐 Testing Distributed Processing...")
        
        try:
            # Create distributed cluster
            cluster = DistributedActiveInferenceCluster()
            cluster.start_cluster(num_workers=3)
            
            # Wait for cluster to initialize
            time.sleep(2)
            
            # Run distributed benchmark
            benchmark_results = distributed_benchmark(cluster, num_observations=150)
            
            # Get cluster performance metrics
            cluster_performance = cluster.get_cluster_performance()
            
            # Stop cluster
            cluster.stop_cluster()
            
            result = {
                'benchmark_results': benchmark_results,
                'cluster_performance': cluster_performance,
                'distributed_efficiency': (
                    benchmark_results['tasks_completed'] / benchmark_results['observations_processed']
                    if benchmark_results['observations_processed'] > 0 else 0
                ),
                'success': True
            }
            
            print(f"✅ Distributed processing completed in {benchmark_results['total_time']:.2f}s")
            print(f"   Distributed throughput: {benchmark_results['throughput']:.1f} obs/sec")
            print(f"   Workers used: {cluster_performance['coordinator_status']['active_workers']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Distributed processing test failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def test_concurrent_workload_management(self) -> Dict[str, Any]:
        """Test concurrent workload management under stress."""
        print("\n⚡ Testing Concurrent Workload Management...")
        
        try:
            # Create multiple agents for stress testing
            agents = [PerformanceOptimizedActiveInference() for _ in range(4)]
            
            # Generate large concurrent workload
            workload_size = 200
            observations = [np.random.rand(8) for _ in range(workload_size)]
            
            start_time = time.time()
            completed_tasks = 0
            failed_tasks = 0
            
            # Process workload with multiple concurrent threads
            def process_batch(agent, batch_obs):
                nonlocal completed_tasks, failed_tasks
                try:
                    results = agent.parallel_inference(batch_obs)
                    completed_tasks += len(results)
                    return results
                except Exception as e:
                    failed_tasks += len(batch_obs)
                    raise
            
            # Split workload among agents
            batch_size = workload_size // len(agents)
            
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                futures = []
                for i, agent in enumerate(agents):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size if i < len(agents) - 1 else workload_size
                    batch = observations[start_idx:end_idx]
                    
                    future = executor.submit(process_batch, agent, batch)
                    futures.append(future)
                
                # Collect results
                all_results = []
                for future in as_completed(futures):
                    try:
                        results = future.result(timeout=30.0)
                        all_results.extend(results)
                    except Exception as e:
                        print(f"Batch processing failed: {e}")
            
            processing_time = time.time() - start_time
            
            # Collect performance statistics
            agent_stats = []
            for i, agent in enumerate(agents):
                stats = agent.get_performance_stats()
                stats['agent_id'] = i
                agent_stats.append(stats)
                agent.cleanup()
            
            result = {
                'processing_time': processing_time,
                'workload_size': workload_size,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': completed_tasks / workload_size if workload_size > 0 else 0,
                'throughput': completed_tasks / processing_time if processing_time > 0 else 0,
                'agent_stats': agent_stats,
                'concurrent_agents': len(agents),
                'success': True
            }
            
            print(f"✅ Concurrent workload test completed in {processing_time:.2f}s")
            print(f"   Success rate: {result['success_rate']:.2%}")
            print(f"   Concurrent throughput: {result['throughput']:.1f} tasks/sec")
            
            return result
            
        except Exception as e:
            print(f"❌ Concurrent workload test failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def test_memory_and_resource_efficiency(self) -> Dict[str, Any]:
        """Test memory and resource efficiency optimizations."""
        print("\n💾 Testing Memory and Resource Efficiency...")
        
        try:
            agent = PerformanceOptimizedActiveInference()
            
            # Test memory pool efficiency
            initial_memory_stats = agent.memory_pool.stats()
            
            # Generate workload that exercises memory pool
            observations = [np.random.rand(8) for _ in range(100)]
            
            start_time = time.time()
            for obs in observations:
                beliefs = agent.update_beliefs(obs)
                action = agent.select_action(beliefs)
            
            processing_time = time.time() - start_time
            
            # Get final memory statistics
            final_memory_stats = agent.memory_pool.stats()
            cache_stats = agent.cache.stats()
            performance_stats = agent.get_performance_stats()
            
            # Test resource cleanup
            agent.cleanup()
            
            result = {
                'processing_time': processing_time,
                'observations_processed': len(observations),
                'initial_memory_stats': initial_memory_stats,
                'final_memory_stats': final_memory_stats,
                'cache_efficiency': cache_stats,
                'memory_pool_efficiency': final_memory_stats['pool_hit_rate'],
                'cache_hit_rate': cache_stats['hit_rate'],
                'performance_optimization': performance_stats['optimization_suggestions'],
                'success': True
            }
            
            print(f"✅ Memory efficiency test completed in {processing_time:.2f}s")
            print(f"   Memory pool efficiency: {final_memory_stats['pool_hit_rate']:.2%}")
            print(f"   Cache efficiency: {cache_stats['hit_rate']:.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Memory efficiency test failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def calculate_scalability_metrics(self) -> Dict[str, float]:
        """Calculate overall scalability metrics."""
        print("\n📊 Calculating Scalability Metrics...")
        
        # Performance optimization score
        perf_result = self.results['results'].get('performance_optimization', {})
        perf_score = 0
        if perf_result.get('success'):
            cache_hit_rate = perf_result.get('cache_performance', {}).get('hit_rate', 0)
            memory_hit_rate = perf_result.get('memory_pool_performance', {}).get('pool_hit_rate', 0)
            perf_score = (cache_hit_rate + memory_hit_rate) * 50  # Max 100
        
        # Scalability score
        scalability_result = self.results['results'].get('scalable_framework', {})
        scalability_score = 0
        if scalability_result.get('success'):
            throughput = scalability_result.get('throughput', 0)
            agents_used = scalability_result.get('framework_stats', {}).get('total_agents', 1)
            scalability_score = min(100, (throughput * agents_used) / 10)  # Normalize
        
        # Distributed processing score
        distributed_result = self.results['results'].get('distributed_processing', {})
        distributed_score = 0
        if distributed_result.get('success'):
            distributed_efficiency = distributed_result.get('distributed_efficiency', 0)
            distributed_score = distributed_efficiency * 100
        
        # Concurrency score
        concurrent_result = self.results['results'].get('concurrent_workload', {})
        concurrent_score = 0
        if concurrent_result.get('success'):
            success_rate = concurrent_result.get('success_rate', 0)
            throughput = concurrent_result.get('throughput', 0)
            concurrent_score = (success_rate + min(1.0, throughput / 50)) * 50  # Max 100
        
        # Resource efficiency score
        efficiency_result = self.results['results'].get('resource_efficiency', {})
        efficiency_score = 0
        if efficiency_result.get('success'):
            memory_efficiency = efficiency_result.get('memory_pool_efficiency', 0)
            cache_efficiency = efficiency_result.get('cache_hit_rate', 0)
            efficiency_score = (memory_efficiency + cache_efficiency) * 50  # Max 100
        
        return {
            'performance_optimization_score': perf_score,
            'scalability_score': scalability_score,
            'distributed_processing_score': distributed_score,
            'concurrency_score': concurrent_score,
            'resource_efficiency_score': efficiency_score
        }
    
    def generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        perf_result = self.results['results'].get('performance_optimization', {})
        if perf_result.get('success'):
            cache_hit_rate = perf_result.get('cache_performance', {}).get('hit_rate', 0)
            if cache_hit_rate < 0.5:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'issue': f'Low cache hit rate: {cache_hit_rate:.1%}',
                    'recommendation': 'Increase cache size or improve cache key strategy'
                })
        
        # Scalability recommendations
        scalability_result = self.results['results'].get('scalable_framework', {})
        if scalability_result.get('success'):
            throughput = scalability_result.get('throughput', 0)
            if throughput < 10:  # Low throughput
                recommendations.append({
                    'type': 'scalability',
                    'priority': 'medium',
                    'issue': f'Low throughput: {throughput:.1f} req/sec',
                    'recommendation': 'Consider increasing worker pool size or optimizing task processing'
                })
        
        # Resource efficiency recommendations
        efficiency_result = self.results['results'].get('resource_efficiency', {})
        if efficiency_result.get('success'):
            memory_efficiency = efficiency_result.get('memory_pool_efficiency', 0)
            if memory_efficiency < 0.3:
                recommendations.append({
                    'type': 'resource_efficiency',
                    'priority': 'medium',
                    'issue': f'Low memory pool efficiency: {memory_efficiency:.1%}',
                    'recommendation': 'Tune memory pool size or allocation patterns'
                })
        
        return recommendations
    
    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run the complete Generation 3 scalability demonstration."""
        print("Starting comprehensive scalability testing...\n")
        
        # Test 1: Performance Optimization
        self.results['results']['performance_optimization'] = self.test_performance_optimization()
        if self.results['results']['performance_optimization']['success']:
            self.results['scalability_features_tested'] += 1
        
        # Test 2: Scalable Framework
        self.results['results']['scalable_framework'] = self.test_scalable_framework()
        if self.results['results']['scalable_framework']['success']:
            self.results['scalability_features_tested'] += 1
        
        # Test 3: Distributed Processing
        self.results['results']['distributed_processing'] = self.test_distributed_processing()
        if self.results['results']['distributed_processing']['success']:
            self.results['scalability_features_tested'] += 1
        
        # Test 4: Concurrent Workload Management
        self.results['results']['concurrent_workload'] = self.test_concurrent_workload_management()
        if self.results['results']['concurrent_workload']['success']:
            self.results['scalability_features_tested'] += 1
        
        # Test 5: Resource Efficiency
        self.results['results']['resource_efficiency'] = self.test_memory_and_resource_efficiency()
        if self.results['results']['resource_efficiency']['success']:
            self.results['scalability_features_tested'] += 1
        
        # Calculate overall metrics
        self.results['system_scalability_metrics'] = self.calculate_scalability_metrics()
        
        # Calculate overall performance score
        metrics = self.results['system_scalability_metrics']
        self.results['overall_performance_score'] = sum(metrics.values()) / len(metrics)
        
        # Generate recommendations
        self.results['recommendations'] = self.generate_recommendations()
        
        # Finalize results
        self.results['total_time'] = time.time() - self.start_time
        self.results['timestamp'] = time.time()
        
        # Stop telemetry
        self.telemetry.stop()
        
        return self.results


def main():
    """Main execution function."""
    try:
        # Initialize and run Generation 3 demo
        demo = Generation3ScalabilityDemo()
        results = demo.run_comprehensive_demo()
        
        # Print final summary
        print("\n" + "="*70)
        print("🎯 GENERATION 3 SCALABILITY DEMONSTRATION COMPLETE")
        print("="*70)
        print(f"📊 Scalability features tested: {results['scalability_features_tested']}/5")
        print(f"⚡ Overall performance score: {results['overall_performance_score']:.1f}/100")
        print(f"⏱️  Total execution time: {results['total_time']:.2f}s")
        
        # Print scalability metrics
        print(f"\n🚀 SCALABILITY METRICS:")
        for metric, score in results['system_scalability_metrics'].items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.1f}/100")
        
        # Print recommendations
        if results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in results['recommendations']:
                print(f"   [{rec['priority'].upper()}] {rec['issue']}")
                print(f"   → {rec['recommendation']}")
        
        # Save results
        results_filename = f"generation_3_scalability_results_{int(time.time())}.json"
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_filename}")
        
        # Determine success
        success_rate = results['scalability_features_tested'] / 5
        if success_rate >= 0.8 and results['overall_performance_score'] >= 60:
            print("\n🎉 GENERATION 3 IMPLEMENTATION: SUCCESSFUL")
            print("   Scalability and performance optimization features working optimally!")
            return 0
        else:
            print("\n⚠️  GENERATION 3 IMPLEMENTATION: PARTIALLY SUCCESSFUL")
            print("   Some scalability features need attention.")
            return 1
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())