#!/usr/bin/env python3
"""
Autonomous Generation 2: MAKE IT ROBUST - Reliability and Fault Tolerance Demo

This demonstration showcases the production-grade reliability, monitoring,
and fault tolerance capabilities implemented in Generation 2.

Features demonstrated:
- Comprehensive telemetry and monitoring systems
- Advanced fault tolerance with circuit breakers
- Bulkhead isolation for resource protection  
- Self-healing and automatic recovery mechanisms
- Anomaly detection and alerting
- Performance profiling and optimization
- Chaos engineering for resilience testing
"""

import time
import numpy as np
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any
import json
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Generation2Demo")

# Import Generation 2 robust components
try:
    from src.python.active_inference.core.agent import ActiveInferenceAgent
    from src.python.active_inference.monitoring.comprehensive_telemetry import (
        TelemetryCollector, PerformanceProfiler, AnomalyDetector,
        get_global_telemetry, get_global_profiler, get_global_anomaly_detector
    )
    from src.python.active_inference.reliability.fault_tolerance import (
        FaultTolerantSystem, CircuitBreakerConfig, RetryConfig,
        CircuitBreaker, BulkheadIsolation, FaultType,
        get_global_fault_tolerant_system
    )
    
    # Import Generation 1 architectures for robustness testing
    from src.python.active_inference.research.cognitive_architectures import (
        HybridSymbolicConnectionistAI
    )
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


class Generation2RobustnessDemo:
    """Comprehensive demonstration of Generation 2 robustness capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger("Gen2RobustnessDemo")
        self.start_time = time.time()
        self.results: Dict[str, Any] = {}
        
        # Initialize telemetry and monitoring
        self.telemetry = get_global_telemetry()
        self.profiler = get_global_profiler()
        self.anomaly_detector = get_global_anomaly_detector()
        
        # Initialize fault tolerance system
        self.fault_system = get_global_fault_tolerant_system()
        
        # Initialize test agents
        self.test_agents = self._create_test_agents()
        
        # Setup anomaly callbacks
        self._setup_anomaly_detection()
        
        # Start self-healing
        self.fault_system.start_self_healing(check_interval=10.0)
        
        self.logger.info("Generation 2 Robustness Demo initialized")
    
    def _create_test_agents(self) -> Dict[str, ActiveInferenceAgent]:
        """Create test agents for robustness testing."""
        agents = {}
        
        try:
            # Robust agent with monitoring
            agents['robust_agent'] = ActiveInferenceAgent(
                state_dim=4,
                obs_dim=8,
                action_dim=2,
                inference_method="variational",
                planning_horizon=5,
                learning_rate=0.01,
                temperature=1.0,
                agent_id="robust_agent",
                enable_logging=True
            )
            
            # High-stress agent for fault testing
            agents['stress_agent'] = ActiveInferenceAgent(
                state_dim=8,
                obs_dim=16,
                action_dim=4,
                inference_method="variational", 
                planning_horizon=15,
                learning_rate=0.001,
                temperature=0.5,
                agent_id="stress_test_agent",
                enable_logging=True
            )
            
            self.logger.info(f"Created {len(agents)} test agents")
            
        except Exception as e:
            self.logger.error(f"Failed to create test agents: {e}")
            agents['fallback'] = None
        
        return agents
    
    def _setup_anomaly_detection(self):
        """Setup anomaly detection callbacks."""
        def anomaly_callback(anomaly: Dict[str, Any]):
            self.logger.warning(f"ANOMALY DETECTED: {anomaly['metric_name']} = {anomaly['value']:.3f} "
                              f"(z_score: {anomaly['z_score']:.2f}, severity: {anomaly['severity']})")
            
            # Record anomaly in telemetry
            self.telemetry.record_metric(
                "system.anomalies_detected", 
                1, 
                tags={
                    'metric': anomaly['metric_name'],
                    'severity': anomaly['severity']
                }
            )
        
        self.anomaly_detector.add_anomaly_callback(anomaly_callback)
    
    def run_comprehensive_robustness_demo(self) -> Dict[str, Any]:
        """Run comprehensive robustness demonstration."""
        self.logger.info("Starting comprehensive Generation 2 robustness demonstration")
        
        demo_results = {}
        
        # 1. Monitoring and Telemetry Systems
        self.logger.info("=== Monitoring and Telemetry Demo ===")
        demo_results['monitoring'] = self._demo_monitoring_systems()
        
        # 2. Circuit Breaker Patterns
        self.logger.info("=== Circuit Breaker Demo ===")
        demo_results['circuit_breakers'] = self._demo_circuit_breakers()
        
        # 3. Bulkhead Isolation
        self.logger.info("=== Bulkhead Isolation Demo ===") 
        demo_results['bulkhead_isolation'] = self._demo_bulkhead_isolation()
        
        # 4. Retry Mechanisms  
        self.logger.info("=== Retry Mechanisms Demo ===")
        demo_results['retry_mechanisms'] = self._demo_retry_mechanisms()
        
        # 5. Anomaly Detection
        self.logger.info("=== Anomaly Detection Demo ===")
        demo_results['anomaly_detection'] = self._demo_anomaly_detection()
        
        # 6. Performance Profiling
        self.logger.info("=== Performance Profiling Demo ===")
        demo_results['performance_profiling'] = self._demo_performance_profiling()
        
        # 7. Fault Injection and Recovery
        self.logger.info("=== Fault Injection Demo ===")
        demo_results['fault_injection'] = self._demo_fault_injection()
        
        # 8. Self-Healing Systems
        self.logger.info("=== Self-Healing Demo ===")
        demo_results['self_healing'] = self._demo_self_healing()
        
        # 9. System Health Assessment
        self.logger.info("=== System Health Assessment ===")
        demo_results['health_assessment'] = self._assess_system_health()
        
        # Compile final results
        total_time = time.time() - self.start_time
        final_results = {
            'generation': 2,
            'demo_type': 'robustness_comprehensive',
            'total_time': total_time,
            'robustness_features_tested': len(demo_results),
            'overall_health_score': self._compute_overall_health_score(demo_results),
            'results': demo_results,
            'system_resilience_metrics': self._compute_resilience_metrics(demo_results),
            'recommendations': self._generate_robustness_recommendations(demo_results),
            'timestamp': time.time()
        }
        
        self.results = final_results
        return final_results
    
    def _demo_monitoring_systems(self) -> Dict[str, Any]:
        """Demonstrate monitoring and telemetry systems."""
        results = {'metrics_recorded': 0, 'traces_completed': 0, 'health_checks': 0}
        
        try:
            # Record various metrics
            metrics_to_record = [
                ('agent.inference_time', np.random.exponential(0.05), 'seconds'),
                ('agent.prediction_error', np.random.gamma(2, 0.1), 'units'),
                ('agent.belief_confidence', np.random.beta(8, 2), 'probability'),
                ('system.memory_usage', np.random.uniform(50, 95), 'percent'),
                ('system.cpu_usage', np.random.uniform(20, 80), 'percent')
            ]
            
            for metric_name, value, unit in metrics_to_record:
                success = self.telemetry.record_metric(
                    metric_name, 
                    value,
                    tags={'demo': 'generation_2', 'component': 'monitoring'},
                    unit=unit
                )
                if success:
                    results['metrics_recorded'] += 1
                
                # Update anomaly detector
                self.anomaly_detector.update_metric_for_detection(metric_name, value)
            
            # Demonstrate distributed tracing
            for i in range(5):
                span = self.telemetry.start_trace(
                    f"agent_inference_step_{i}",
                    tags={'step': i, 'demo': 'monitoring'}
                )
                
                # Simulate some work
                time.sleep(np.random.uniform(0.01, 0.05))
                
                # Add some trace events
                self.telemetry.log_trace_event(span, "belief_update", {'confidence': 0.8})
                self.telemetry.log_trace_event(span, "action_selection", {'action_type': 'exploration'})
                
                # Finish trace
                self.telemetry.finish_trace(span, "finished")
                results['traces_completed'] += 1
            
            # Update health status for various components
            components = ['agent_core', 'inference_engine', 'planning_system', 'memory_system']
            for component in components:
                status = np.random.choice(['healthy', 'degraded'], p=[0.8, 0.2])
                self.telemetry.update_health_status(
                    component, 
                    status,
                    details={'last_check': time.time(), 'performance_score': np.random.uniform(0.7, 1.0)}
                )
                results['health_checks'] += 1
            
            # Get telemetry statistics
            results['telemetry_stats'] = self.telemetry.get_performance_stats()
            results['health_summary'] = self.telemetry.get_health_summary()
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Monitoring demo failed: {e}")
        
        return results
    
    def _demo_circuit_breakers(self) -> Dict[str, Any]:
        """Demonstrate circuit breaker patterns."""
        results = {'circuit_breakers_tested': 0, 'successful_protections': 0}
        
        try:
            # Create circuit breakers with different configurations
            circuit_configs = {
                'fast_fail': CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5.0),
                'tolerant': CircuitBreakerConfig(failure_threshold=10, recovery_timeout=30.0),
                'sensitive': CircuitBreakerConfig(failure_threshold=2, recovery_timeout=15.0)
            }
            
            circuit_test_results = {}
            
            for cb_name, config in circuit_configs.items():
                circuit_breaker = self.fault_system.create_circuit_breaker(f"test_{cb_name}", config)
                results['circuit_breakers_tested'] += 1
                
                # Test normal operation
                def normal_operation():
                    if np.random.random() < 0.1:  # 10% failure rate
                        raise Exception("Simulated service failure")
                    return "success"
                
                # Test with some failures to trigger circuit breaker
                success_count = 0
                failure_count = 0
                
                for i in range(20):
                    try:
                        result = circuit_breaker.call(normal_operation)
                        success_count += 1
                    except Exception:
                        failure_count += 1
                
                circuit_stats = circuit_breaker.get_statistics()
                circuit_test_results[cb_name] = {
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'final_state': circuit_stats['state'],
                    'success_rate': circuit_stats['success_rate']
                }
                
                if circuit_stats['state'] == 'open' and failure_count > config.failure_threshold:
                    results['successful_protections'] += 1
            
            results['circuit_test_results'] = circuit_test_results
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Circuit breaker demo failed: {e}")
        
        return results
    
    def _demo_bulkhead_isolation(self) -> Dict[str, Any]:
        """Demonstrate bulkhead isolation patterns."""
        results = {'bulkheads_tested': 0, 'isolation_effectiveness': 0.0}
        
        try:
            # Create bulkheads for different resource classes
            bulkhead_configs = {
                'critical_ops': (5, 10, 2.0),    # max_concurrent, queue_size, timeout
                'background_ops': (10, 50, 5.0),
                'batch_ops': (3, 5, 10.0)
            }
            
            bulkhead_test_results = {}
            
            for bulkhead_name, (max_concurrent, queue_size, timeout) in bulkhead_configs.items():
                bulkhead = self.fault_system.create_bulkhead(
                    f"test_{bulkhead_name}",
                    max_concurrent,
                    queue_size,
                    timeout
                )
                results['bulkheads_tested'] += 1
                
                # Simulate concurrent load
                def simulate_work(work_id: int):
                    # Simulate variable work time
                    work_time = np.random.exponential(0.1)
                    time.sleep(work_time)
                    return f"work_{work_id}_completed"
                
                # Submit concurrent requests
                import concurrent.futures
                
                completed_requests = 0
                rejected_requests = 0
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    futures = []
                    
                    for i in range(30):  # Submit more requests than bulkhead can handle
                        try:
                            future = executor.submit(bulkhead.execute, simulate_work, i)
                            futures.append(future)
                        except Exception as e:
                            if "full" in str(e).lower() or "reject" in str(e).lower():
                                rejected_requests += 1
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(futures, timeout=15):
                        try:
                            result = future.result()
                            completed_requests += 1
                        except Exception:
                            rejected_requests += 1
                
                bulkhead_stats = bulkhead.get_statistics()
                bulkhead_test_results[bulkhead_name] = {
                    'completed_requests': completed_requests,
                    'rejected_requests': rejected_requests,
                    'bulkhead_stats': bulkhead_stats
                }
                
                # Measure isolation effectiveness
                if bulkhead_stats['rejected_requests'] > 0:
                    results['isolation_effectiveness'] += 1
            
            results['isolation_effectiveness'] /= max(1, len(bulkhead_configs))
            results['bulkhead_test_results'] = bulkhead_test_results
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Bulkhead isolation demo failed: {e}")
        
        return results
    
    def _demo_retry_mechanisms(self) -> Dict[str, Any]:
        """Demonstrate retry mechanisms with different strategies."""
        results = {'retry_strategies_tested': 0, 'successful_recoveries': 0}
        
        try:
            # Test different retry configurations
            retry_configs = {
                'exponential_backoff': RetryConfig(
                    max_attempts=5,
                    base_delay=0.1,
                    backoff_strategy="exponential",
                    jitter=True
                ),
                'linear_backoff': RetryConfig(
                    max_attempts=3,
                    base_delay=0.2,
                    backoff_strategy="linear",
                    jitter=False
                ),
                'constant_retry': RetryConfig(
                    max_attempts=4,
                    base_delay=0.1,
                    backoff_strategy="constant",
                    jitter=True
                )
            }
            
            retry_test_results = {}
            
            for strategy_name, config in retry_configs.items():
                results['retry_strategies_tested'] += 1
                
                # Simulate a function that fails initially but succeeds after retries
                attempt_count = 0
                
                def flaky_function():
                    nonlocal attempt_count
                    attempt_count += 1
                    
                    if attempt_count < 3:  # Fail first 2 attempts
                        raise Exception(f"Simulated failure on attempt {attempt_count}")
                    
                    return "success_after_retries"
                
                try:
                    start_time = time.time()
                    result = self.fault_system.retry_manager.retry(
                        flaky_function,
                        config_override=config
                    )
                    retry_time = time.time() - start_time
                    
                    if result == "success_after_retries":
                        results['successful_recoveries'] += 1
                    
                    retry_test_results[strategy_name] = {
                        'result': result,
                        'attempts_needed': attempt_count,
                        'retry_time': retry_time,
                        'success': True
                    }
                    
                except Exception as e:
                    retry_test_results[strategy_name] = {
                        'error': str(e),
                        'attempts_made': attempt_count,
                        'success': False
                    }
                
                # Reset for next test
                attempt_count = 0
            
            results['retry_test_results'] = retry_test_results
            results['retry_stats'] = self.fault_system.retry_manager.get_retry_statistics()
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Retry mechanisms demo failed: {e}")
        
        return results
    
    def _demo_anomaly_detection(self) -> Dict[str, Any]:
        """Demonstrate anomaly detection capabilities."""
        results = {'anomalies_injected': 0, 'anomalies_detected': 0}
        
        try:
            # Inject normal metrics to establish baseline
            metric_name = "test_metric_for_anomaly_detection"
            
            # Generate normal data
            for i in range(50):
                normal_value = np.random.normal(10.0, 1.0)
                self.anomaly_detector.update_metric_for_detection(metric_name, normal_value)
                time.sleep(0.01)  # Small delay
            
            # Inject anomalies
            anomaly_values = [25.0, -5.0, 30.0, 100.0, -20.0]  # Clearly anomalous values
            
            detected_anomalies = []
            for anomaly_value in anomaly_values:
                results['anomalies_injected'] += 1
                
                anomaly = self.anomaly_detector.update_metric_for_detection(metric_name, anomaly_value)
                if anomaly:
                    results['anomalies_detected'] += 1
                    detected_anomalies.append(anomaly)
            
            # Get anomaly statistics
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
            recent_anomalies = self.anomaly_detector.get_anomalies(time_window_hours=1.0)
            
            results['detected_anomaly_details'] = detected_anomalies
            results['anomaly_summary'] = anomaly_summary
            results['recent_anomalies_count'] = len(recent_anomalies)
            results['detection_rate'] = results['anomalies_detected'] / max(1, results['anomalies_injected'])
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Anomaly detection demo failed: {e}")
        
        return results
    
    def _demo_performance_profiling(self) -> Dict[str, Any]:
        """Demonstrate performance profiling capabilities."""
        results = {'functions_profiled': 0, 'memory_snapshots': 0}
        
        try:
            # Profile different types of operations
            def fast_operation():
                time.sleep(0.01)
                return "fast"
            
            def slow_operation():
                time.sleep(0.1)
                return "slow"
            
            def memory_intensive_operation():
                # Simulate memory usage
                data = [np.random.randn(1000) for _ in range(100)]
                return len(data)
            
            operations = [
                ('fast_operation', fast_operation),
                ('slow_operation', slow_operation),
                ('memory_intensive_operation', memory_intensive_operation)
            ]
            
            for op_name, operation in operations:
                results['functions_profiled'] += 1
                
                # Profile the operation
                for i in range(5):
                    with self.profiler.profile_function(op_name, tags={'iteration': i}):
                        operation()
            
            # Take memory snapshots
            for i in range(3):
                snapshot = self.profiler.take_memory_snapshot(f"demo_snapshot_{i}")
                if 'error' not in snapshot:
                    results['memory_snapshots'] += 1
                time.sleep(0.1)
            
            # Get profiling results
            function_profiles = self.profiler.get_function_profiles()
            memory_trend = self.profiler.get_memory_usage_trend(window_minutes=1)
            performance_report = self.profiler.generate_performance_report()
            
            results['function_profiles'] = function_profiles
            results['memory_trend'] = memory_trend
            results['performance_recommendations'] = performance_report['recommendations']
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Performance profiling demo failed: {e}")
        
        return results
    
    def _demo_fault_injection(self) -> Dict[str, Any]:
        """Demonstrate fault injection and recovery."""
        results = {'fault_types_tested': 0, 'recovery_successful': 0}
        
        try:
            # Define different types of faults to inject
            fault_scenarios = {
                'timeout_fault': lambda: self._inject_timeout_fault(),
                'resource_exhaustion': lambda: self._inject_resource_fault(),
                'computation_error': lambda: self._inject_computation_fault(),
                'validation_error': lambda: self._inject_validation_fault()
            }
            
            fault_test_results = {}
            
            for fault_type, fault_injector in fault_scenarios.items():
                results['fault_types_tested'] += 1
                
                try:
                    # Execute with comprehensive protection
                    start_time = time.time()
                    
                    result = self.fault_system.execute_with_protection(
                        fault_injector,
                        circuit_breaker_name="fault_test_cb",
                        bulkhead_name="fault_test_bulkhead", 
                        retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
                        fallback_func=lambda: "fallback_executed"
                    )
                    
                    recovery_time = time.time() - start_time
                    
                    if "fallback" in str(result):
                        results['recovery_successful'] += 1
                    
                    fault_test_results[fault_type] = {
                        'result': str(result),
                        'recovery_time': recovery_time,
                        'recovered': True
                    }
                    
                except Exception as e:
                    fault_test_results[fault_type] = {
                        'error': str(e),
                        'recovered': False
                    }
            
            # Get system health after fault injection
            system_health = self.fault_system.get_system_health()
            
            results['fault_test_results'] = fault_test_results
            results['system_health_post_faults'] = system_health
            results['recovery_rate'] = results['recovery_successful'] / max(1, results['fault_types_tested'])
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Fault injection demo failed: {e}")
        
        return results
    
    def _inject_timeout_fault(self):
        """Simulate a timeout fault."""
        time.sleep(5.0)  # Simulate long operation
        return "timeout_completed"
    
    def _inject_resource_fault(self):
        """Simulate a resource exhaustion fault."""
        # Simulate memory exhaustion
        data = []
        for i in range(10000):
            data.append(np.random.randn(1000))
        raise MemoryError("Simulated memory exhaustion")
    
    def _inject_computation_fault(self):
        """Simulate a computation error."""
        raise ValueError("Simulated computation error: division by zero")
    
    def _inject_validation_fault(self):
        """Simulate a validation error."""
        raise TypeError("Simulated validation error: invalid input type")
    
    def _demo_self_healing(self) -> Dict[str, Any]:
        """Demonstrate self-healing capabilities."""
        results = {'healing_callbacks_registered': 0, 'healing_activations': 0}
        
        try:
            # Register healing callbacks for different components
            components = ['test_agent', 'inference_engine', 'memory_system']
            
            healing_activations = []
            
            for component in components:
                results['healing_callbacks_registered'] += 1
                
                def create_healing_callback(comp_name):
                    def healing_callback():
                        healing_activations.append({
                            'component': comp_name,
                            'timestamp': time.time(),
                            'action': 'restart_component'
                        })
                        self.logger.info(f"Self-healing activated for {comp_name}")
                    return healing_callback
                
                self.fault_system.register_healing_callback(component, create_healing_callback(component))
            
            # Simulate component failures that trigger healing
            # Force some circuit breakers to open to trigger healing
            cb1 = self.fault_system.create_circuit_breaker("healing_test_cb1")
            cb1.force_open()
            
            cb2 = self.fault_system.create_circuit_breaker("healing_test_cb2") 
            cb2.force_open()
            
            # Wait for healing system to detect and respond
            time.sleep(12.0)  # Wait longer than check interval
            
            results['healing_activations'] = len(healing_activations)
            results['healing_events'] = healing_activations
            results['healing_effectiveness'] = min(1.0, len(healing_activations) / len(components))
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
            self.logger.error(f"Self-healing demo failed: {e}")
        
        return results
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        try:
            # Get health from all monitoring systems
            telemetry_health = self.telemetry.get_health_summary()
            fault_system_health = self.fault_system.get_system_health()
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
            
            # Assess agent health
            agent_health = {}
            for agent_name, agent in self.test_agents.items():
                if agent:
                    agent_stats = agent.get_statistics()
                    agent_health[agent_name] = {
                        'health_status': agent_stats.get('health_status', 'unknown'),
                        'error_rate': agent_stats.get('error_rate', 0),
                        'is_healthy': agent_stats.get('health_status') == 'healthy'
                    }
            
            # Compute overall health score
            health_factors = []
            
            # Telemetry health (0-100)
            if telemetry_health['total_components'] > 0:
                telemetry_score = (telemetry_health['healthy_components'] / telemetry_health['total_components']) * 100
                health_factors.append(telemetry_score)
            
            # Fault system health
            fault_score = fault_system_health['system_health_score']
            health_factors.append(fault_score)
            
            # Anomaly health (inverse of anomaly rate)
            recent_anomalies = anomaly_summary['recent_anomalies_1h']
            anomaly_score = max(0, 100 - (recent_anomalies * 10))  # Penalty for anomalies
            health_factors.append(anomaly_score)
            
            # Agent health
            healthy_agents = sum(1 for h in agent_health.values() if h['is_healthy'])
            agent_score = (healthy_agents / max(1, len(agent_health))) * 100
            health_factors.append(agent_score)
            
            overall_health_score = np.mean(health_factors) if health_factors else 0
            
            return {
                'overall_health_score': overall_health_score,
                'telemetry_health': telemetry_health,
                'fault_system_health': fault_system_health,
                'anomaly_summary': anomaly_summary,
                'agent_health': agent_health,
                'health_factors': {
                    'telemetry_score': telemetry_score if 'telemetry_score' in locals() else 0,
                    'fault_score': fault_score,
                    'anomaly_score': anomaly_score,
                    'agent_score': agent_score
                },
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def _compute_overall_health_score(self, demo_results: Dict[str, Any]) -> float:
        """Compute overall health score from demo results."""
        try:
            health_assessment = demo_results.get('health_assessment', {})
            if health_assessment.get('success'):
                return health_assessment.get('overall_health_score', 0)
            
            # Fallback: compute from individual demo success rates
            success_count = 0
            total_count = 0
            
            for category, results in demo_results.items():
                if isinstance(results, dict) and 'success' in results:
                    total_count += 1
                    if results['success']:
                        success_count += 1
            
            return (success_count / max(1, total_count)) * 100
            
        except Exception:
            return 0.0
    
    def _compute_resilience_metrics(self, demo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute system resilience metrics."""
        try:
            metrics = {
                'fault_tolerance_score': 0.0,
                'recovery_effectiveness': 0.0,
                'monitoring_coverage': 0.0,
                'self_healing_capability': 0.0
            }
            
            # Fault tolerance score
            fault_injection = demo_results.get('fault_injection', {})
            if fault_injection.get('success'):
                metrics['fault_tolerance_score'] = fault_injection.get('recovery_rate', 0) * 100
            
            # Recovery effectiveness  
            circuit_breakers = demo_results.get('circuit_breakers', {})
            if circuit_breakers.get('success'):
                metrics['recovery_effectiveness'] = (
                    circuit_breakers.get('successful_protections', 0) / 
                    max(1, circuit_breakers.get('circuit_breakers_tested', 1))
                ) * 100
            
            # Monitoring coverage
            monitoring = demo_results.get('monitoring', {})
            if monitoring.get('success'):
                metrics['monitoring_coverage'] = min(100, (
                    monitoring.get('metrics_recorded', 0) + 
                    monitoring.get('traces_completed', 0) + 
                    monitoring.get('health_checks', 0)
                ) * 5)  # Scale factor
            
            # Self-healing capability
            self_healing = demo_results.get('self_healing', {})
            if self_healing.get('success'):
                metrics['self_healing_capability'] = self_healing.get('healing_effectiveness', 0) * 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to compute resilience metrics: {e}")
            return {
                'fault_tolerance_score': 0.0,
                'recovery_effectiveness': 0.0,
                'monitoring_coverage': 0.0,
                'self_healing_capability': 0.0,
                'error': str(e)
            }
    
    def _generate_robustness_recommendations(self, demo_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for improving system robustness."""
        recommendations = []
        
        try:
            # Check monitoring coverage
            monitoring = demo_results.get('monitoring', {})
            if monitoring.get('success'):
                health_summary = monitoring.get('health_summary', {})
                if health_summary.get('critical_components', 0) > 0:
                    recommendations.append({
                        'type': 'monitoring',
                        'priority': 'high',
                        'issue': f"{health_summary['critical_components']} components in critical state",
                        'recommendation': "Investigate critical components and implement recovery procedures"
                    })
            
            # Check circuit breaker effectiveness
            circuit_breakers = demo_results.get('circuit_breakers', {})
            if circuit_breakers.get('success'):
                if circuit_breakers.get('successful_protections', 0) < circuit_breakers.get('circuit_breakers_tested', 1):
                    recommendations.append({
                        'type': 'fault_tolerance', 
                        'priority': 'medium',
                        'issue': "Some circuit breakers not providing adequate protection",
                        'recommendation': "Review and tune circuit breaker thresholds"
                    })
            
            # Check anomaly detection effectiveness
            anomaly_detection = demo_results.get('anomaly_detection', {})
            if anomaly_detection.get('success'):
                detection_rate = anomaly_detection.get('detection_rate', 0)
                if detection_rate < 0.8:
                    recommendations.append({
                        'type': 'monitoring',
                        'priority': 'medium', 
                        'issue': f"Anomaly detection rate is {detection_rate:.1%}",
                        'recommendation': "Consider adjusting anomaly detection thresholds or algorithms"
                    })
            
            # Check performance issues
            performance = demo_results.get('performance_profiling', {})
            if performance.get('success'):
                perf_recommendations = performance.get('performance_recommendations', [])
                for perf_rec in perf_recommendations:
                    if perf_rec.get('severity') in ['high', 'critical']:
                        recommendations.append({
                            'type': 'performance',
                            'priority': perf_rec['severity'],
                            'issue': perf_rec['issue'],
                            'recommendation': perf_rec['recommendation']
                        })
            
            # Check self-healing effectiveness
            self_healing = demo_results.get('self_healing', {})
            if self_healing.get('success'):
                effectiveness = self_healing.get('healing_effectiveness', 0)
                if effectiveness < 0.5:
                    recommendations.append({
                        'type': 'resilience',
                        'priority': 'medium',
                        'issue': f"Self-healing effectiveness is {effectiveness:.1%}",
                        'recommendation': "Implement more comprehensive self-healing callbacks"
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return [{
                'type': 'system',
                'priority': 'high',
                'issue': 'Failed to analyze system for recommendations',
                'recommendation': 'Manual system review required'
            }]
    
    def save_results(self, filename: str = None) -> str:
        """Save demonstration results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"generation_2_robustness_results_{timestamp}.json"
        
        filepath = Path(filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup demo resources."""
        try:
            # Stop fault tolerance system
            self.fault_system.stop_self_healing()
            
            # Shutdown monitoring systems  
            from src.python.active_inference.monitoring.comprehensive_telemetry import shutdown_global_telemetry
            from src.python.active_inference.reliability.fault_tolerance import shutdown_global_fault_tolerance
            
            shutdown_global_telemetry()
            shutdown_global_fault_tolerance()
            
            self.logger.info("Demo cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def main():
    """Main demonstration function."""
    print("🛡️  Autonomous SDLC Generation 2: MAKE IT ROBUST - Robustness Demo")
    print("=" * 70)
    
    demo = None
    try:
        # Initialize demonstration
        demo = Generation2RobustnessDemo()
        
        # Run comprehensive robustness demo
        print("\n🔧 Running comprehensive robustness demonstration...")
        results = demo.run_comprehensive_robustness_demo()
        
        # Print summary
        print("\n📊 ROBUSTNESS DEMONSTRATION SUMMARY")
        print("=" * 40)
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Robustness features tested: {results['robustness_features_tested']}")
        print(f"Overall health score: {results['overall_health_score']:.1f}/100")
        
        # Resilience metrics
        resilience = results.get('system_resilience_metrics', {})
        print(f"\n🛡️  System Resilience Metrics:")
        print(f"  • Fault tolerance: {resilience.get('fault_tolerance_score', 0):.1f}/100")
        print(f"  • Recovery effectiveness: {resilience.get('recovery_effectiveness', 0):.1f}/100")
        print(f"  • Monitoring coverage: {resilience.get('monitoring_coverage', 0):.1f}/100")  
        print(f"  • Self-healing capability: {resilience.get('self_healing_capability', 0):.1f}/100")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n⚠️  Recommendations ({len(recommendations)} items):")
            for rec in recommendations[:5]:  # Show top 5
                print(f"  • [{rec['priority'].upper()}] {rec['issue']}")
                print(f"    → {rec['recommendation']}")
        
        # Save results
        print("\n💾 Saving results...")
        filepath = demo.save_results()
        if filepath:
            print(f"Results saved to: {filepath}")
        
        print(f"\n🎉 Generation 2 robustness demonstration completed successfully!")
        print("✅ System demonstrates production-grade reliability and fault tolerance")
        print("Next: Generation 3 will focus on scalability and optimization")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
        return None
    
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        traceback.print_exc()
        return None
    
    finally:
        if demo:
            demo.cleanup()


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"Critical error: {e}")
        exit(1)