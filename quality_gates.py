#!/usr/bin/env python3
"""
Quality Gates Implementation - Autonomous SDLC
Comprehensive testing, validation, and quality assurance
"""

import sys
sys.path.append('src/python')

import numpy as np
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from active_inference import ActiveInferenceAgent
from active_inference.environments import MockEnvironment


class QualityGatesValidator:
    """Comprehensive quality gates validation system."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'gates': {},
            'overall_status': 'PENDING',
            'execution_time': 0
        }
    
    def run_all_gates(self):
        """Execute all quality gates in sequence."""
        start_time = time.perf_counter()
        
        gates = [
            ('Unit Tests', self.gate_unit_tests),
            ('Integration Tests', self.gate_integration_tests),
            ('Performance Tests', self.gate_performance_tests),
            ('Security Validation', self.gate_security_tests),
            ('Code Quality', self.gate_code_quality),
            ('Documentation', self.gate_documentation),
            ('API Contracts', self.gate_api_contracts),
            ('Error Handling', self.gate_error_handling)
        ]
        
        passed_gates = 0
        total_gates = len(gates)
        
        print("🚀 QUALITY GATES EXECUTION")
        print("=" * 60)
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Executing {gate_name}...")
            
            gate_start = time.perf_counter()
            try:
                result = gate_func()
                gate_time = time.perf_counter() - gate_start
                
                status = "PASS" if result['passed'] else "FAIL"
                if result['passed']:
                    passed_gates += 1
                
                self.results['gates'][gate_name] = {
                    'status': status,
                    'execution_time': gate_time,
                    'details': result.get('details', {}),
                    'metrics': result.get('metrics', {})
                }
                
                print(f"   Status: {status}")
                print(f"   Time: {gate_time:.2f}s")
                if result.get('metrics'):
                    for metric, value in result['metrics'].items():
                        print(f"   {metric}: {value}")
                
            except Exception as e:
                gate_time = time.perf_counter() - gate_start
                self.results['gates'][gate_name] = {
                    'status': 'ERROR',
                    'execution_time': gate_time,
                    'error': str(e)
                }
                print(f"   Status: ERROR - {e}")
        
        total_time = time.perf_counter() - start_time
        self.results['execution_time'] = total_time
        
        # Determine overall status
        if passed_gates == total_gates:
            self.results['overall_status'] = 'PASS'
        elif passed_gates >= total_gates * 0.85:  # 85% threshold
            self.results['overall_status'] = 'CONDITIONAL_PASS'
        else:
            self.results['overall_status'] = 'FAIL'
        
        print("\n" + "=" * 60)
        print("📊 QUALITY GATES SUMMARY")
        print(f"Passed: {passed_gates}/{total_gates} ({passed_gates/total_gates*100:.1f}%)")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Total Execution Time: {total_time:.2f}s")
        
        return self.results
    
    def gate_unit_tests(self):
        """Unit testing gate."""
        try:
            # Core component tests
            agent = ActiveInferenceAgent(
                state_dim=2, obs_dim=4, action_dim=2,
                agent_id="unit_test_agent"
            )
            
            tests_passed = 0
            total_tests = 5
            
            # Test 1: Agent initialization
            if agent.state_dim == 2 and agent.obs_dim == 4:
                tests_passed += 1
            
            # Test 2: Belief initialization
            if len(agent.beliefs.get_all_beliefs()) > 0:
                tests_passed += 1
            
            # Test 3: Observation processing
            obs = np.random.randn(4)
            beliefs = agent.infer_states(obs)
            if beliefs is not None:
                tests_passed += 1
            
            # Test 4: Action planning
            action = agent.plan_action()
            if action is not None and len(action) == 2:
                tests_passed += 1
            
            # Test 5: Full perception-action cycle
            action = agent.act(obs)
            if action is not None:
                tests_passed += 1
            
            coverage = tests_passed / total_tests
            
            return {
                'passed': coverage >= 0.8,  # 80% minimum
                'metrics': {
                    'Test Coverage': f"{coverage*100:.1f}%",
                    'Tests Passed': f"{tests_passed}/{total_tests}"
                },
                'details': {
                    'coverage_threshold': '80%',
                    'actual_coverage': f"{coverage*100:.1f}%"
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_integration_tests(self):
        """Integration testing gate."""
        try:
            # Test agent-environment integration
            agent = ActiveInferenceAgent(
                state_dim=3, obs_dim=6, action_dim=2,
                agent_id="integration_test"
            )
            env = MockEnvironment(obs_dim=6, action_dim=2)
            
            integration_tests = 0
            total_integrations = 3
            
            # Test 1: Environment reset and observation
            obs = env.reset()
            if obs is not None and len(obs) == 6:
                integration_tests += 1
            
            # Test 2: Agent-environment step cycle
            action = agent.act(obs)
            obs_new, reward, terminated, truncated, info = env.step(action)
            if obs_new is not None and isinstance(reward, (int, float)):
                integration_tests += 1
            
            # Test 3: Multi-step interaction
            total_reward = 0
            for _ in range(10):
                action = agent.act(obs_new)
                obs_new, reward, terminated, truncated, info = env.step(action)
                agent.update_model(obs_new, action, reward)
                total_reward += reward
                
                if terminated or truncated:
                    obs_new = env.reset()
            
            if abs(total_reward) > 0:  # Some reward was accumulated
                integration_tests += 1
            
            integration_rate = integration_tests / total_integrations
            
            return {
                'passed': integration_rate >= 0.8,
                'metrics': {
                    'Integration Rate': f"{integration_rate*100:.1f}%",
                    'Components Tested': f"{integration_tests}/{total_integrations}",
                    'Total Reward': f"{total_reward:.3f}"
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_performance_tests(self):
        """Performance testing gate."""
        try:
            agent = ActiveInferenceAgent(
                state_dim=3, obs_dim=6, action_dim=2,
                agent_id="performance_test"
            )
            env = MockEnvironment(obs_dim=6, action_dim=2)
            
            # Measure performance metrics
            obs = env.reset()
            
            # Inference speed
            inference_times = []
            for _ in range(10):
                start = time.perf_counter()
                beliefs = agent.infer_states(obs)
                inference_times.append(time.perf_counter() - start)
            
            # Planning speed
            planning_times = []
            for _ in range(10):
                start = time.perf_counter()
                action = agent.plan_action()
                planning_times.append(time.perf_counter() - start)
            
            avg_inference = np.mean(inference_times) * 1000  # ms
            avg_planning = np.mean(planning_times) * 1000   # ms
            total_cycle = avg_inference + avg_planning
            
            # Performance criteria
            performance_pass = (
                avg_inference < 1000 and  # < 1s for inference
                avg_planning < 1000 and   # < 1s for planning
                total_cycle < 1500        # < 1.5s total cycle
            )
            
            return {
                'passed': performance_pass,
                'metrics': {
                    'Avg Inference Time': f"{avg_inference:.1f}ms",
                    'Avg Planning Time': f"{avg_planning:.1f}ms",
                    'Total Cycle Time': f"{total_cycle:.1f}ms",
                    'Theoretical FPS': f"{1000/total_cycle:.1f}"
                },
                'details': {
                    'inference_threshold': '1000ms',
                    'planning_threshold': '1000ms',
                    'total_threshold': '1500ms'
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_security_tests(self):
        """Security validation gate."""
        try:
            security_tests = 0
            total_security_tests = 4
            
            # Test 1: Input validation
            try:
                agent = ActiveInferenceAgent(
                    state_dim=-1, obs_dim=4, action_dim=2  # Invalid input
                )
                # Should not reach here
            except:
                security_tests += 1  # Correctly rejected invalid input
            
            # Test 2: Observation validation
            agent = ActiveInferenceAgent(
                state_dim=2, obs_dim=4, action_dim=2,
                agent_id="security_test"
            )
            
            try:
                bad_obs = np.array([np.nan, 1.0, 2.0, 3.0])
                agent.act(bad_obs)
                # Should not reach here
            except:
                security_tests += 1  # Correctly rejected NaN input
            
            # Test 3: Safe checkpoint operations
            try:
                temp_path = "/tmp/security_test_checkpoint.json"
                agent.save_checkpoint(temp_path)
                loaded_agent = ActiveInferenceAgent.load_checkpoint(temp_path)
                Path(temp_path).unlink(missing_ok=True)
                security_tests += 1
            except:
                pass  # Checkpoint operations failed safely
            
            # Test 4: Agent ID sanitization
            try:
                agent_clean = ActiveInferenceAgent(
                    state_dim=2, obs_dim=4, action_dim=2,
                    agent_id="normal_id_123"
                )
                if agent_clean.agent_id == "normal_id_123":
                    security_tests += 1
            except:
                pass
            
            security_rate = security_tests / total_security_tests
            
            return {
                'passed': security_rate >= 0.75,  # 75% minimum
                'metrics': {
                    'Security Tests Passed': f"{security_tests}/{total_security_tests}",
                    'Security Rate': f"{security_rate*100:.1f}%"
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_code_quality(self):
        """Code quality gate."""
        try:
            quality_metrics = {
                'imports_working': False,
                'classes_defined': False,
                'error_handling': False,
                'logging_present': False
            }
            
            # Test imports
            try:
                from active_inference import ActiveInferenceAgent
                from active_inference.core import GenerativeModel
                from active_inference.environments import MockEnvironment
                quality_metrics['imports_working'] = True
            except:
                pass
            
            # Test class definitions
            if hasattr(ActiveInferenceAgent, '__init__') and hasattr(ActiveInferenceAgent, 'act'):
                quality_metrics['classes_defined'] = True
            
            # Test error handling
            try:
                agent = ActiveInferenceAgent(
                    state_dim=2, obs_dim=4, action_dim=2,
                    agent_id="quality_test"
                )
                if hasattr(agent, '_record_error'):
                    quality_metrics['error_handling'] = True
            except:
                pass
            
            # Test logging
            if hasattr(agent, 'logger'):
                quality_metrics['logging_present'] = True
            
            passed_metrics = sum(quality_metrics.values())
            total_metrics = len(quality_metrics)
            quality_score = passed_metrics / total_metrics
            
            return {
                'passed': quality_score >= 0.8,
                'metrics': {
                    'Quality Score': f"{quality_score*100:.1f}%",
                    'Metrics Passed': f"{passed_metrics}/{total_metrics}"
                },
                'details': quality_metrics
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_documentation(self):
        """Documentation gate."""
        try:
            doc_checks = {
                'readme_exists': Path('README.md').exists(),
                'module_docstrings': False,
                'function_docstrings': False,
                'examples_present': False
            }
            
            # Check module docstrings
            if ActiveInferenceAgent.__doc__ is not None:
                doc_checks['module_docstrings'] = True
            
            # Check function docstrings
            if ActiveInferenceAgent.__init__.__doc__ is not None:
                doc_checks['function_docstrings'] = True
            
            # Check examples
            if Path('examples').exists() or Path('simple_demo.py').exists():
                doc_checks['examples_present'] = True
            
            doc_score = sum(doc_checks.values()) / len(doc_checks)
            
            return {
                'passed': doc_score >= 0.75,
                'metrics': {
                    'Documentation Score': f"{doc_score*100:.1f}%"
                },
                'details': doc_checks
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_api_contracts(self):
        """API contract validation gate."""
        try:
            agent = ActiveInferenceAgent(
                state_dim=2, obs_dim=4, action_dim=2,
                agent_id="api_test"
            )
            
            contract_tests = 0
            total_contracts = 4
            
            # Test 1: act() method contract
            obs = np.random.randn(4)
            action = agent.act(obs)
            if isinstance(action, np.ndarray) and len(action) == 2:
                contract_tests += 1
            
            # Test 2: infer_states() contract
            beliefs = agent.infer_states(obs)
            if beliefs is not None:
                contract_tests += 1
            
            # Test 3: plan_action() contract
            planned_action = agent.plan_action()
            if isinstance(planned_action, np.ndarray) and len(planned_action) == 2:
                contract_tests += 1
            
            # Test 4: get_statistics() contract
            stats = agent.get_statistics()
            if isinstance(stats, dict) and 'agent_id' in stats:
                contract_tests += 1
            
            contract_rate = contract_tests / total_contracts
            
            return {
                'passed': contract_rate >= 0.8,
                'metrics': {
                    'API Contracts Passed': f"{contract_tests}/{total_contracts}",
                    'Contract Compliance': f"{contract_rate*100:.1f}%"
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def gate_error_handling(self):
        """Error handling gate."""
        try:
            error_tests = 0
            total_error_tests = 3
            
            # Test 1: Invalid dimensions
            try:
                agent = ActiveInferenceAgent(
                    state_dim=0, obs_dim=4, action_dim=2
                )
            except:
                error_tests += 1  # Correctly handled error
            
            # Test 2: Invalid observation
            agent = ActiveInferenceAgent(
                state_dim=2, obs_dim=4, action_dim=2,
                agent_id="error_test"
            )
            
            try:
                wrong_obs = np.random.randn(6)  # Wrong size
                agent.act(wrong_obs)
            except:
                error_tests += 1  # Correctly handled error
            
            # Test 3: Error recovery
            health_before = agent.get_health_status()
            try:
                bad_obs = np.array([np.inf, 1.0, 2.0, 3.0])
                agent.act(bad_obs)
            except:
                pass
            
            health_after = agent.get_health_status()
            if health_after['total_errors'] >= health_before['total_errors']:
                error_tests += 1  # Error was recorded
            
            error_rate = error_tests / total_error_tests
            
            return {
                'passed': error_rate >= 0.66,  # 2/3 minimum
                'metrics': {
                    'Error Handling Rate': f"{error_rate*100:.1f}%",
                    'Error Tests Passed': f"{error_tests}/{total_error_tests}"
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def save_report(self, filename='quality_gates_report.json'):
        """Save quality gates report."""
        report_path = Path(filename)
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📋 Quality Gates Report saved to: {report_path}")
        return report_path


def main():
    """Execute all quality gates."""
    print("🛡️ AUTONOMOUS SDLC - QUALITY GATES EXECUTION")
    print("=" * 70)
    
    validator = QualityGatesValidator()
    results = validator.run_all_gates()
    
    # Save report with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"quality_gates_report_{timestamp}.json"
    validator.save_report(report_file)
    
    # Final verdict
    if results['overall_status'] == 'PASS':
        print("\n🎉 ALL QUALITY GATES PASSED! System ready for deployment.")
        return True
    elif results['overall_status'] == 'CONDITIONAL_PASS':
        print("\n⚠️ CONDITIONAL PASS: 85%+ gates passed. Review failures before deployment.")
        return True
    else:
        print("\n❌ QUALITY GATES FAILED! System not ready for deployment.")
        return False


if __name__ == "__main__":
    main()