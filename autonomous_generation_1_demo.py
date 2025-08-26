#!/usr/bin/env python3
"""
Autonomous Generation 1: MAKE IT WORK - Novel Research Demo

This demo showcases the advanced cognitive architectures and continual learning
capabilities implemented in Generation 1 of the Autonomous SDLC enhancement.

Features demonstrated:
- Hybrid Symbolic-Connectionist Active Inference (HSCAI)
- Compositional Active Inference for structured reasoning
- Causal Active Inference with interventional planning
- Elastic Weight Consolidation for Active Inference (EWC-AI)
- Memory-Augmented Active Inference with episodic replay
- Progressive Neural Networks for hierarchical skill acquisition
"""

import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Generation1Demo")

# Import our enhanced Active Inference components
try:
    from src.python.active_inference.core.agent import ActiveInferenceAgent
    from src.python.active_inference.research.cognitive_architectures import (
        HybridSymbolicConnectionistAI,
        CompositionalActiveInference,
        CausalActiveInference
    )
    from src.python.active_inference.research.continual_learning import (
        ElasticWeightConsolidationAI,
        MemoryAugmentedActiveInference,
        ProgressiveNeuralNetworks
    )
    from src.python.active_inference.research.advanced_algorithms import (
        HierarchicalTemporalActiveInference,
        MetaActiveInference,
        QuantumInspiredVariationalInference,
        MultiModalActiveInference
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


class Generation1ResearchDemo:
    """Comprehensive demonstration of Generation 1 research capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger("Gen1Demo")
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()
        
        # Initialize base agents for different experiments
        self.base_agents = self._create_base_agents()
        
        # Initialize advanced architectures
        self.architectures = self._initialize_architectures()
        
        self.logger.info("Generation 1 Research Demo initialized")
    
    def _create_base_agents(self) -> Dict[str, ActiveInferenceAgent]:
        """Create base agents for different experiments."""
        agents = {}
        
        try:
            # Standard agent for basic experiments
            agents['standard'] = ActiveInferenceAgent(
                state_dim=4,
                obs_dim=8,
                action_dim=2,
                inference_method="variational",
                planning_horizon=5,
                learning_rate=0.01,
                temperature=1.0,
                agent_id="standard_agent"
            )
            
            # High-dimensional agent for complex experiments
            agents['complex'] = ActiveInferenceAgent(
                state_dim=8,
                obs_dim=16,
                action_dim=4,
                inference_method="variational",
                planning_horizon=10,
                learning_rate=0.005,
                temperature=0.8,
                agent_id="complex_agent"
            )
            
            # Fast agent for real-time experiments
            agents['fast'] = ActiveInferenceAgent(
                state_dim=2,
                obs_dim=4,
                action_dim=2,
                inference_method="variational",
                planning_horizon=3,
                learning_rate=0.02,
                temperature=1.5,
                agent_id="fast_agent"
            )
            
            self.logger.info(f"Created {len(agents)} base agents")
            
        except Exception as e:
            self.logger.error(f"Failed to create base agents: {e}")
            agents['fallback'] = None
        
        return agents
    
    def _initialize_architectures(self) -> Dict[str, Any]:
        """Initialize advanced cognitive architectures."""
        architectures = {}
        
        if 'standard' in self.base_agents and self.base_agents['standard']:
            try:
                # Hybrid Symbolic-Connectionist AI
                architectures['hscai'] = HybridSymbolicConnectionistAI(
                    base_agent=self.base_agents['standard'],
                    max_symbolic_rules=50,
                    rule_learning_rate=0.1
                )
                
                # Compositional Active Inference
                architectures['compositional'] = CompositionalActiveInference(
                    base_agent=self.base_agents['standard']
                )
                
                # Causal Active Inference
                architectures['causal'] = CausalActiveInference(
                    base_agent=self.base_agents['standard']
                )
                
                # Continual Learning Components
                architectures['ewc'] = ElasticWeightConsolidationAI(
                    base_agent=self.base_agents['standard'],
                    ewc_lambda=1000.0
                )
                
                architectures['memory'] = MemoryAugmentedActiveInference(
                    base_agent=self.base_agents['standard'],
                    memory_size=1000,
                    replay_batch_size=16
                )
                
                architectures['progressive'] = ProgressiveNeuralNetworks(
                    base_agent=self.base_agents['standard']
                )
                
                # Advanced Algorithms
                architectures['hierarchical'] = HierarchicalTemporalActiveInference(
                    n_levels=3,
                    temporal_scales=[1, 5, 15]
                )
                
                architectures['meta'] = MetaActiveInference(
                    base_agent=self.base_agents['standard']
                )
                
                architectures['quantum'] = QuantumInspiredVariationalInference(
                    n_qubits=6
                )
                
                if 'complex' in self.base_agents and self.base_agents['complex']:
                    architectures['multimodal'] = MultiModalActiveInference(
                        modalities=['visual', 'auditory', 'proprioceptive']
                    )
                
                self.logger.info(f"Initialized {len(architectures)} advanced architectures")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize architectures: {e}")
        
        return architectures
    
    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of all Generation 1 capabilities."""
        self.logger.info("Starting comprehensive Generation 1 demonstration")
        
        demo_results = {}
        
        # 1. Hybrid Symbolic-Connectionist Reasoning
        self.logger.info("=== Hybrid Symbolic-Connectionist AI Demo ===")
        demo_results['hscai'] = self._demo_hybrid_symbolic_connectionist()
        
        # 2. Compositional Reasoning
        self.logger.info("=== Compositional Active Inference Demo ===")
        demo_results['compositional'] = self._demo_compositional_reasoning()
        
        # 3. Causal Reasoning and Planning
        self.logger.info("=== Causal Active Inference Demo ===")
        demo_results['causal'] = self._demo_causal_reasoning()
        
        # 4. Continual Learning
        self.logger.info("=== Continual Learning Demo ===")
        demo_results['continual'] = self._demo_continual_learning()
        
        # 5. Advanced Algorithms
        self.logger.info("=== Advanced Algorithms Demo ===")
        demo_results['advanced'] = self._demo_advanced_algorithms()
        
        # 6. Performance Benchmarking
        self.logger.info("=== Performance Benchmarking ===")
        demo_results['benchmarks'] = self._run_performance_benchmarks()
        
        # Compile final results
        total_time = time.time() - self.start_time
        final_results = {
            'generation': 1,
            'demo_type': 'comprehensive',
            'total_time': total_time,
            'architectures_tested': len(self.architectures),
            'success_rate': self._compute_overall_success_rate(demo_results),
            'results': demo_results,
            'summary': self._generate_results_summary(demo_results),
            'timestamp': time.time()
        }
        
        self.results = final_results
        return final_results
    
    def _demo_hybrid_symbolic_connectionist(self) -> Dict[str, Any]:
        """Demonstrate hybrid symbolic-connectionist reasoning."""
        if 'hscai' not in self.architectures:
            return {'error': 'HSCAI not available'}
        
        hscai = self.architectures['hscai']
        results = {'test_cases': []}
        
        try:
            # Test Case 1: High uncertainty situation
            observation1 = np.random.randn(8) * 2.0  # High variance observation
            context1 = {'high_uncertainty': True, 'exploration_needed': True}
            
            result1 = hscai.hybrid_decision_making(observation1, context1)
            results['test_cases'].append({
                'case': 'high_uncertainty',
                'symbolic_rules_activated': result1.get('symbolic_rules_activated', 0),
                'neural_confidence': result1.get('neural_confidence', 0),
                'symbolic_confidence': result1.get('symbolic_confidence', 0),
                'processing_time': result1.get('processing_time', 0)
            })
            
            # Test Case 2: Goal achievement situation
            observation2 = np.ones(8) * 0.1  # Low variance, stable
            context2 = {'goal_achieved': True, 'low_uncertainty': True}
            
            result2 = hscai.hybrid_decision_making(observation2, context2)
            results['test_cases'].append({
                'case': 'goal_achieved',
                'symbolic_rules_activated': result2.get('symbolic_rules_activated', 0),
                'neural_confidence': result2.get('neural_confidence', 0),
                'symbolic_confidence': result2.get('symbolic_confidence', 0),
                'processing_time': result2.get('processing_time', 0)
            })
            
            # Test Case 3: Learning scenario with feedback
            observation3 = np.random.randn(8) * 0.5
            context3 = {'performance_feedback': 0.8}  # Positive feedback
            
            result3 = hscai.hybrid_decision_making(observation3, context3)
            results['test_cases'].append({
                'case': 'positive_feedback',
                'symbolic_rules_activated': result3.get('symbolic_rules_activated', 0),
                'neural_confidence': result3.get('neural_confidence', 0),
                'symbolic_confidence': result3.get('symbolic_confidence', 0),
                'processing_time': result3.get('processing_time', 0)
            })
            
            # Get comprehensive statistics
            results['statistics'] = hscai.get_hybrid_statistics()
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _demo_compositional_reasoning(self) -> Dict[str, Any]:
        """Demonstrate compositional reasoning capabilities."""
        if 'compositional' not in self.architectures:
            return {'error': 'Compositional AI not available'}
        
        compositional = self.architectures['compositional']
        results = {'goal_achievements': []}
        
        try:
            # Goal 1: Navigation task
            observation1 = np.array([1.0, 0.5, -0.2, 0.8, 0.1, -0.3, 0.6, 0.4])
            goal1 = "move forward and then turn right to explore the area"
            
            result1 = compositional.compositional_reasoning(observation1, goal1, max_composition_depth=4)
            results['goal_achievements'].append({
                'goal': goal1,
                'composition_complexity': result1.get('composition_complexity', 0),
                'reasoning_time': result1.get('reasoning_time', 0),
                'required_skills': result1.get('required_skills', []),
                'execution_success': result1.get('execution_result', {}).get('success_rate', 0)
            })
            
            # Goal 2: Exploration and observation task
            observation2 = np.array([0.2, -0.1, 0.9, -0.5, 0.3, 0.7, -0.4, 0.1])
            goal2 = "explore the environment and observe interesting patterns"
            
            result2 = compositional.compositional_reasoning(observation2, goal2, max_composition_depth=3)
            results['goal_achievements'].append({
                'goal': goal2,
                'composition_complexity': result2.get('composition_complexity', 0),
                'reasoning_time': result2.get('reasoning_time', 0),
                'required_skills': result2.get('required_skills', []),
                'execution_success': result2.get('execution_result', {}).get('success_rate', 0)
            })
            
            # Goal 3: Approach and wait task
            observation3 = np.array([-0.3, 0.6, 0.1, 0.4, -0.7, 0.2, 0.5, -0.1])
            goal3 = "approach the target and wait for the right moment"
            
            result3 = compositional.compositional_reasoning(observation3, goal3, max_composition_depth=2)
            results['goal_achievements'].append({
                'goal': goal3,
                'composition_complexity': result3.get('composition_complexity', 0),
                'reasoning_time': result3.get('reasoning_time', 0),
                'required_skills': result3.get('required_skills', []),
                'execution_success': result3.get('execution_result', {}).get('success_rate', 0)
            })
            
            # Get comprehensive statistics
            results['statistics'] = compositional.get_compositional_statistics()
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _demo_causal_reasoning(self) -> Dict[str, Any]:
        """Demonstrate causal reasoning and interventional planning."""
        if 'causal' not in self.architectures:
            return {'error': 'Causal AI not available'}
        
        causal = self.architectures['causal']
        results = {'causal_scenarios': []}
        
        try:
            # Scenario 1: Goal-directed intervention
            observation1 = np.array([0.5, -0.2, 0.8, 0.1, -0.6, 0.3, 0.7, -0.1])
            goal_state1 = {
                'prediction_error': 0.2,
                'uncertainty': 0.3,
                'goal_progress': 0.9
            }
            
            result1 = causal.causal_reasoning_and_planning(
                observation1, goal_state1, intervention_budget=2
            )
            results['causal_scenarios'].append({
                'scenario': 'goal_directed_intervention',
                'causal_relations_discovered': len(result1.get('causal_model_update', {}).get('new_relations', [])),
                'interventions_planned': len(result1.get('intervention_plan', {}).get('selected_interventions', [])),
                'reasoning_time': result1.get('reasoning_time', 0),
                'counterfactual_analysis': result1.get('counterfactual_analysis', {})
            })
            
            # Scenario 2: Uncertainty reduction
            observation2 = np.array([0.1, 0.9, -0.3, 0.6, 0.2, -0.8, 0.4, 0.5])
            goal_state2 = {
                'uncertainty': 0.1,
                'prediction_error': 0.1,
                'exploration_tendency': 0.3
            }
            
            result2 = causal.causal_reasoning_and_planning(
                observation2, goal_state2, intervention_budget=3
            )
            results['causal_scenarios'].append({
                'scenario': 'uncertainty_reduction',
                'causal_relations_discovered': len(result2.get('causal_model_update', {}).get('new_relations', [])),
                'interventions_planned': len(result2.get('intervention_plan', {}).get('selected_interventions', [])),
                'reasoning_time': result2.get('reasoning_time', 0),
                'counterfactual_analysis': result2.get('counterfactual_analysis', {})
            })
            
            # Get comprehensive statistics
            results['statistics'] = causal.get_causal_statistics()
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _demo_continual_learning(self) -> Dict[str, Any]:
        """Demonstrate continual learning capabilities."""
        results = {'ewc': {}, 'memory': {}, 'progressive': {}}
        
        # EWC Demonstration
        if 'ewc' in self.architectures:
            try:
                ewc = self.architectures['ewc']
                
                # Simulate multiple tasks
                task_results = []
                
                for task_id in ['navigation', 'exploration', 'manipulation']:
                    # Begin new task
                    task_start = ewc.begin_new_task(task_id, f"Learning {task_id} skills")
                    
                    # Simulate learning episodes
                    for episode in range(5):
                        obs = np.random.randn(8)
                        action = np.random.randn(2)
                        reward = np.random.uniform(-1, 1)
                        
                        update_result = ewc.ewc_regularized_update(obs, action, reward)
                    
                    # Evaluate transfer learning
                    transfer_result = ewc.evaluate_transfer_learning()
                    
                    task_results.append({
                        'task_id': task_id,
                        'initialization': task_start,
                        'transfer_learning': transfer_result
                    })
                
                results['ewc'] = {
                    'tasks_completed': len(task_results),
                    'task_results': task_results,
                    'statistics': ewc.get_ewc_statistics(),
                    'success': True
                }
                
            except Exception as e:
                results['ewc'] = {'error': str(e), 'success': False}
        
        # Memory-Augmented Demonstration
        if 'memory' in self.architectures:
            try:
                memory = self.architectures['memory']
                
                # Store diverse experiences
                storage_results = []
                for i in range(20):
                    obs = np.random.randn(8)
                    action = np.random.randn(2)
                    reward = np.random.uniform(-1, 1)
                    context = {'episode': i, 'phase': 'learning'}
                    
                    storage_result = memory.store_experience(obs, action, reward, context)
                    storage_results.append(storage_result)
                
                # Perform experience replay
                replay_results = []
                for i in range(5):
                    obs = np.random.randn(8)
                    replay_result = memory.adaptive_replay(obs)
                    replay_results.append(replay_result)
                
                results['memory'] = {
                    'experiences_stored': len(storage_results),
                    'replay_sessions': len(replay_results),
                    'statistics': memory.get_memory_statistics(),
                    'success': True
                }
                
            except Exception as e:
                results['memory'] = {'error': str(e), 'success': False}
        
        # Progressive Networks Demonstration
        if 'progressive' in self.architectures:
            try:
                progressive = self.architectures['progressive']
                
                # Add multiple task columns
                task_columns = []
                for task_id in ['task_a', 'task_b', 'task_c']:
                    column_result = progressive.add_new_task_column(task_id)
                    task_columns.append(column_result)
                
                # Demonstrate progressive forward passes
                forward_results = []
                for task_id in ['task_a', 'task_b', 'task_c']:
                    obs = np.random.randn(8)
                    forward_result = progressive.progressive_forward_pass(obs, task_id)
                    forward_results.append(forward_result)
                
                # Freeze previous columns
                freeze_result = progressive.freeze_previous_columns(except_task='task_c')
                
                results['progressive'] = {
                    'task_columns_added': len(task_columns),
                    'forward_passes': len(forward_results),
                    'freeze_result': freeze_result,
                    'statistics': progressive.get_progressive_statistics(),
                    'success': True
                }
                
            except Exception as e:
                results['progressive'] = {'error': str(e), 'success': False}
        
        return results
    
    def _demo_advanced_algorithms(self) -> Dict[str, Any]:
        """Demonstrate advanced algorithms."""
        results = {'hierarchical': {}, 'meta': {}, 'quantum': {}, 'multimodal': {}}
        
        # Hierarchical Temporal Active Inference
        if 'hierarchical' in self.architectures:
            try:
                hierarchical = self.architectures['hierarchical']
                
                # Process observations through hierarchy
                hierarchy_results = []
                for i in range(5):
                    obs = np.random.randn(8)
                    action = np.random.randn(2)
                    
                    hierarchy_result = hierarchical.process_observation(obs, action)
                    hierarchy_results.append(hierarchy_result)
                
                # Plan hierarchical actions
                planning_result = hierarchical.plan_hierarchical_action(horizon=10)
                
                results['hierarchical'] = {
                    'hierarchy_processing': len(hierarchy_results),
                    'planning_result': planning_result,
                    'statistics': hierarchical.get_hierarchy_statistics(),
                    'success': True
                }
                
            except Exception as e:
                results['hierarchical'] = {'error': str(e), 'success': False}
        
        # Meta-Active Inference
        if 'meta' in self.architectures:
            try:
                meta = self.architectures['meta']
                
                # Simulate adaptation to new tasks
                adaptation_results = []
                for task_id in ['meta_task_1', 'meta_task_2']:
                    initial_obs = [np.random.randn(8) for _ in range(3)]
                    adaptation_result = meta.adapt_to_new_task(task_id, initial_obs, max_adaptation_steps=5)
                    adaptation_results.append(adaptation_result)
                
                results['meta'] = {
                    'adaptations_completed': len(adaptation_results),
                    'adaptation_results': adaptation_results,
                    'statistics': meta.get_meta_statistics(),
                    'success': True
                }
                
            except Exception as e:
                results['meta'] = {'error': str(e), 'success': False}
        
        # Quantum-Inspired Variational Inference
        if 'quantum' in self.architectures:
            try:
                quantum = self.architectures['quantum']
                
                # Simulate quantum belief updates
                if 'standard' in self.base_agents:
                    quantum_results = []
                    for i in range(3):
                        obs = np.random.randn(8)
                        beliefs = self.base_agents['standard'].beliefs
                        
                        updated_beliefs = quantum.quantum_belief_update(obs, beliefs)
                        quantum_results.append({'success': True})
                
                    results['quantum'] = {
                        'quantum_updates': len(quantum_results),
                        'statistics': quantum.get_quantum_statistics(),
                        'success': True
                    }
                else:
                    results['quantum'] = {'error': 'No base agent available', 'success': False}
                
            except Exception as e:
                results['quantum'] = {'error': str(e), 'success': False}
        
        # Multi-Modal Active Inference
        if 'multimodal' in self.architectures:
            try:
                multimodal = self.architectures['multimodal']
                
                # Process multi-modal observations
                multimodal_results = []
                for i in range(3):
                    observations = {
                        'visual': np.random.randn(64),
                        'auditory': np.random.randn(16),
                        'proprioceptive': np.random.randn(12)
                    }
                    
                    actions = {
                        'visual': np.random.randn(2),
                        'auditory': np.random.randn(2),
                        'proprioceptive': np.random.randn(2)
                    }
                    
                    multimodal_result = multimodal.process_multimodal_observation(observations, actions)
                    multimodal_results.append(multimodal_result)
                
                results['multimodal'] = {
                    'multimodal_processing': len(multimodal_results),
                    'statistics': multimodal.get_multimodal_statistics(),
                    'success': True
                }
                
            except Exception as e:
                results['multimodal'] = {'error': str(e), 'success': False}
        
        return results
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks across all systems."""
        benchmarks = {
            'processing_speeds': {},
            'memory_usage': {},
            'accuracy_metrics': {},
            'scalability_tests': {}
        }
        
        try:
            # Processing speed benchmarks
            for arch_name, architecture in self.architectures.items():
                if hasattr(architecture, 'get_statistics') or hasattr(architecture, f'get_{arch_name}_statistics'):
                    try:
                        start_time = time.time()
                        
                        # Run a standard operation
                        if hasattr(architecture, 'hybrid_decision_making'):
                            obs = np.random.randn(8)
                            architecture.hybrid_decision_making(obs)
                        elif hasattr(architecture, 'compositional_reasoning'):
                            obs = np.random.randn(8)
                            architecture.compositional_reasoning(obs, "test goal")
                        elif hasattr(architecture, 'process_observation'):
                            obs = np.random.randn(8)
                            action = np.random.randn(2)
                            architecture.process_observation(obs, action)
                        
                        processing_time = time.time() - start_time
                        benchmarks['processing_speeds'][arch_name] = processing_time
                        
                    except Exception as e:
                        benchmarks['processing_speeds'][arch_name] = f"Error: {str(e)}"
            
            # Memory usage estimation (simplified)
            import sys
            for arch_name, architecture in self.architectures.items():
                try:
                    memory_size = sys.getsizeof(architecture)
                    benchmarks['memory_usage'][arch_name] = memory_size
                except Exception as e:
                    benchmarks['memory_usage'][arch_name] = f"Error: {str(e)}"
            
            # Scalability tests
            scalability_results = {}
            for n_operations in [10, 50, 100]:
                operation_times = []
                
                for i in range(n_operations):
                    start_time = time.time()
                    
                    # Simple operation: agent action
                    if 'standard' in self.base_agents and self.base_agents['standard']:
                        obs = np.random.randn(8)
                        self.base_agents['standard'].act(obs)
                    
                    operation_time = time.time() - start_time
                    operation_times.append(operation_time)
                
                scalability_results[f'{n_operations}_operations'] = {
                    'avg_time': np.mean(operation_times),
                    'total_time': sum(operation_times),
                    'throughput': n_operations / sum(operation_times)
                }
            
            benchmarks['scalability_tests'] = scalability_results
            benchmarks['success'] = True
            
        except Exception as e:
            benchmarks['error'] = str(e)
            benchmarks['success'] = False
        
        return benchmarks
    
    def _compute_overall_success_rate(self, demo_results: Dict[str, Any]) -> float:
        """Compute overall success rate across all demonstrations."""
        total_tests = 0
        successful_tests = 0
        
        for category, results in demo_results.items():
            if isinstance(results, dict):
                if 'success' in results:
                    total_tests += 1
                    if results['success']:
                        successful_tests += 1
                
                # Check nested results
                for key, value in results.items():
                    if isinstance(value, dict) and 'success' in value:
                        total_tests += 1
                        if value['success']:
                            successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0
    
    def _generate_results_summary(self, demo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of results."""
        summary = {
            'total_architectures_tested': len(self.architectures),
            'successful_architectures': 0,
            'failed_architectures': 0,
            'key_achievements': [],
            'performance_highlights': {},
            'areas_for_improvement': []
        }
        
        # Count successful vs failed architectures
        for category, results in demo_results.items():
            if isinstance(results, dict):
                if results.get('success', False):
                    summary['successful_architectures'] += 1
                else:
                    summary['failed_architectures'] += 1
        
        # Extract key achievements
        if demo_results.get('hscai', {}).get('success'):
            summary['key_achievements'].append("Hybrid Symbolic-Connectionist reasoning operational")
        
        if demo_results.get('compositional', {}).get('success'):
            summary['key_achievements'].append("Compositional skill learning and execution")
        
        if demo_results.get('causal', {}).get('success'):
            summary['key_achievements'].append("Causal reasoning and interventional planning")
        
        if demo_results.get('continual', {}).get('ewc', {}).get('success'):
            summary['key_achievements'].append("Elastic Weight Consolidation for continual learning")
        
        if demo_results.get('continual', {}).get('memory', {}).get('success'):
            summary['key_achievements'].append("Memory-augmented learning with episodic replay")
        
        # Performance highlights
        benchmarks = demo_results.get('benchmarks', {})
        if benchmarks.get('success'):
            summary['performance_highlights'] = {
                'fastest_architecture': min(benchmarks.get('processing_speeds', {}), 
                                          key=lambda k: benchmarks['processing_speeds'][k] 
                                          if isinstance(benchmarks['processing_speeds'][k], (int, float)) else float('inf')),
                'scalability_achieved': len(benchmarks.get('scalability_tests', {})) > 0,
                'memory_efficiency': 'tested' if benchmarks.get('memory_usage') else 'not_tested'
            }
        
        return summary
    
    def save_results(self, filename: str = None) -> str:
        """Save demonstration results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"generation_1_demo_results_{timestamp}.json"
        
        filepath = Path(filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return ""


def main():
    """Main demonstration function."""
    print("🚀 Autonomous SDLC Generation 1: MAKE IT WORK - Research Demo")
    print("=" * 60)
    
    # Initialize demonstration
    demo = Generation1ResearchDemo()
    
    # Run comprehensive demo
    print("\n📊 Running comprehensive demonstration...")
    results = demo.run_comprehensive_demo()
    
    # Print summary
    print("\n📈 DEMONSTRATION SUMMARY")
    print("=" * 30)
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Architectures tested: {results['architectures_tested']}")
    print(f"Overall success rate: {results['success_rate']:.2%}")
    
    summary = results.get('summary', {})
    print(f"\nSuccessful architectures: {summary.get('successful_architectures', 0)}")
    print(f"Failed architectures: {summary.get('failed_architectures', 0)}")
    
    print("\n🎯 Key Achievements:")
    for achievement in summary.get('key_achievements', []):
        print(f"  ✓ {achievement}")
    
    print("\n⚡ Performance Highlights:")
    perf = summary.get('performance_highlights', {})
    for key, value in perf.items():
        print(f"  • {key}: {value}")
    
    # Save results
    print("\n💾 Saving results...")
    filepath = demo.save_results()
    if filepath:
        print(f"Results saved to: {filepath}")
    
    print("\n🎉 Generation 1 demonstration completed successfully!")
    print("Next: Generation 2 will focus on robustness and reliability")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()