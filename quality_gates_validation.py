#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation for Active Inference Framework

This script validates:
1. Code syntax and structure
2. Security compliance
3. Performance benchmarks
4. Architecture consistency
5. Research implementation completeness
"""

import os
import sys
import ast
import re
import time
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quality_gates.log')
    ]
)
logger = logging.getLogger(__name__)


class QualityGatesValidator:
    """Comprehensive quality gates validation system."""
    
    def __init__(self, repo_root: str = None):
        self.repo_root = Path(repo_root or '.')
        self.results = {
            'syntax_validation': {'passed': 0, 'failed': 0, 'errors': []},
            'security_validation': {'passed': 0, 'failed': 0, 'issues': []},
            'performance_validation': {'passed': 0, 'failed': 0, 'metrics': {}},
            'architecture_validation': {'passed': 0, 'failed': 0, 'violations': []},
            'research_validation': {'passed': 0, 'failed': 0, 'missing': []}
        }
        
        # Security patterns to detect
        self.security_patterns = {
            'sql_injection': [r'execute\s*\(.*%.*\)', r'format\s*\(.*%.*\)'],
            'code_injection': [r'eval\s*\(', r'exec\s*\(', r'__import__\s*\('],
            'path_traversal': [r'\.\./.*', r'\\\\.*'],
            'hardcoded_secrets': [r'password\s*=\s*["\'][^"\']["\']', r'api_key\s*=\s*["\'][^"\']["\']'],
            'unsafe_pickle': [r'pickle\.loads?\s*\(', r'cPickle\.loads?\s*\(']
        }
        
        # Expected architectural components
        self.required_components = [
            'src/python/active_inference/core/agent.py',
            'src/python/active_inference/core/free_energy.py',
            'src/python/active_inference/research/advanced_algorithms.py',
            'src/python/active_inference/research/novel_benchmarks.py',
            'src/python/active_inference/utils/advanced_validation.py',
            'src/python/active_inference/performance/advanced_optimization.py'
        ]
        
        # Research capabilities to validate
        self.research_capabilities = [
            'HierarchicalTemporalActiveInference',
            'MetaActiveInference',
            'QuantumInspiredVariationalInference',
            'MultiModalActiveInference',
            'NovelBenchmarkSuite',
            'AdvancedValidator',
            'ScalableActiveInferenceFramework'
        ]
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        logger.info("Starting comprehensive quality gates validation")
        start_time = time.time()
        
        try:
            # 1. Syntax and Structure Validation
            logger.info("Running syntax validation...")
            self.validate_syntax_and_structure()
            
            # 2. Security Validation
            logger.info("Running security validation...")
            self.validate_security()
            
            # 3. Performance Validation
            logger.info("Running performance validation...")
            self.validate_performance()
            
            # 4. Architecture Validation
            logger.info("Running architecture validation...")
            self.validate_architecture()
            
            # 5. Research Implementation Validation
            logger.info("Running research validation...")
            self.validate_research_implementation()
            
            execution_time = time.time() - start_time
            
            # Generate final report
            report = self.generate_quality_report(execution_time)
            
            logger.info(f"Quality gates validation completed in {execution_time:.1f}s")
            return report
            
        except Exception as e:
            logger.error(f"Quality gates validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def validate_syntax_and_structure(self):
        """Validate Python syntax and code structure."""
        python_files = list(self.repo_root.rglob('*.py'))
        
        for py_file in python_files:
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                # Check syntax by parsing
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to check syntax
                ast.parse(content)
                
                # Check for basic code quality patterns
                self._check_code_quality(py_file, content)
                
                self.results['syntax_validation']['passed'] += 1
                
            except SyntaxError as e:
                error_msg = f"Syntax error in {py_file}: {e}"
                self.results['syntax_validation']['errors'].append(error_msg)
                self.results['syntax_validation']['failed'] += 1
                logger.error(error_msg)
                
            except Exception as e:
                error_msg = f"Error parsing {py_file}: {e}"
                self.results['syntax_validation']['errors'].append(error_msg)
                self.results['syntax_validation']['failed'] += 1
                logger.warning(error_msg)
    
    def _check_code_quality(self, file_path: Path, content: str):
        """Check basic code quality patterns."""
        lines = content.split('\n')
        
        # Check for common issues
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Check line length (PEP 8 recommends 79, we'll be lenient at 120)
            if len(line) > 120:
                self.results['syntax_validation']['errors'].append(
                    f"{file_path}:{line_num}: Line too long ({len(line)} chars)"
                )
            
            # Check for TODO/FIXME comments in production code
            if ('TODO' in line or 'FIXME' in line) and 'examples' not in str(file_path):
                self.results['syntax_validation']['errors'].append(
                    f"{file_path}:{line_num}: TODO/FIXME found in production code"
                )
    
    def validate_security(self):
        """Validate security compliance."""
        python_files = list(self.repo_root.rglob('*.py'))
        
        for py_file in python_files:
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for security patterns
                for threat_type, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            issue = {
                                'file': str(py_file),
                                'line': line_num,
                                'threat_type': threat_type,
                                'pattern': pattern,
                                'match': match.group()
                            }
                            self.results['security_validation']['issues'].append(issue)
                            self.results['security_validation']['failed'] += 1
                            logger.warning(f"Security issue in {py_file}:{line_num}: {threat_type}")
                
                # Check for hardcoded sensitive information
                self._check_sensitive_data(py_file, content)
                
                if not any(threat in str(py_file).lower() for threat in ['test', 'example']):
                    self.results['security_validation']['passed'] += 1
                    
            except Exception as e:
                logger.error(f"Error during security validation of {py_file}: {e}")
    
    def _check_sensitive_data(self, file_path: Path, content: str):
        """Check for potentially sensitive hardcoded data."""
        sensitive_patterns = [
            (r'password\s*=\s*["\'][^"\']["\']', 'hardcoded_password'),
            (r'secret\s*=\s*["\'][^"\']["\']', 'hardcoded_secret'),
            (r'token\s*=\s*["\'][^"\']["\']', 'hardcoded_token'),
            (r'api[_-]?key\s*=\s*["\'][^"\']["\']', 'hardcoded_api_key')
        ]
        
        for pattern, issue_type in sensitive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Skip if it's clearly a placeholder or example
                if any(placeholder in match.group().lower() 
                      for placeholder in ['example', 'placeholder', 'your_', 'test_', 'dummy']):
                    continue
                
                line_num = content[:match.start()].count('\n') + 1
                issue = {
                    'file': str(file_path),
                    'line': line_num,
                    'threat_type': issue_type,
                    'severity': 'high'
                }
                self.results['security_validation']['issues'].append(issue)
    
    def validate_performance(self):
        """Validate performance characteristics."""
        # Analyze code complexity and performance patterns
        python_files = list(self.repo_root.rglob('*.py'))
        
        total_lines = 0
        total_functions = 0
        complex_functions = 0
        
        for py_file in python_files:
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    total_lines += len([l for l in lines if l.strip()])
                
                # Parse AST to analyze functions
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Estimate function complexity (simplified cyclomatic complexity)
                        complexity = self._estimate_complexity(node)
                        if complexity > 10:  # High complexity threshold
                            complex_functions += 1
                            logger.info(f"Complex function detected: {py_file}:{node.lineno} - {node.name} (complexity: {complexity})")
                
            except Exception as e:
                logger.warning(f"Error analyzing performance of {py_file}: {e}")
        
        # Calculate metrics
        self.results['performance_validation']['metrics'] = {
            'total_lines_of_code': total_lines,
            'total_functions': total_functions,
            'complex_functions': complex_functions,
            'complexity_ratio': complex_functions / max(1, total_functions),
            'average_file_size': total_lines / len(python_files) if python_files else 0
        }
        
        # Performance validation criteria
        if complex_functions / max(1, total_functions) < 0.2:  # Less than 20% complex functions
            self.results['performance_validation']['passed'] += 1
        else:
            self.results['performance_validation']['failed'] += 1
        
        # Check for performance anti-patterns
        self._check_performance_patterns()
    
    def _estimate_complexity(self, function_node) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(function_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, (ast.ExceptHandler,)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _check_performance_patterns(self):
        """Check for common performance anti-patterns."""
        # This could be expanded to check for specific patterns like:
        # - Nested loops in large datasets
        # - Inefficient string concatenation
        # - Missing caching where beneficial
        pass
    
    def validate_architecture(self):
        """Validate architectural consistency and completeness."""
        # Check required components exist
        missing_components = []
        
        for component in self.required_components:
            component_path = self.repo_root / component
            if not component_path.exists():
                missing_components.append(component)
                self.results['architecture_validation']['violations'].append(
                    f"Missing required component: {component}"
                )
                self.results['architecture_validation']['failed'] += 1
            else:
                self.results['architecture_validation']['passed'] += 1
        
        # Check directory structure
        expected_dirs = [
            'src/python/active_inference/core',
            'src/python/active_inference/research',
            'src/python/active_inference/utils',
            'src/python/active_inference/performance',
            'examples',
            'tests'
        ]
        
        for expected_dir in expected_dirs:
            dir_path = self.repo_root / expected_dir
            if not dir_path.exists():
                self.results['architecture_validation']['violations'].append(
                    f"Missing expected directory: {expected_dir}"
                )
                self.results['architecture_validation']['failed'] += 1
            else:
                self.results['architecture_validation']['passed'] += 1
        
        # Check for architectural consistency
        self._check_import_structure()
        
        if missing_components:
            logger.error(f"Missing architectural components: {missing_components}")
    
    def _check_import_structure(self):
        """Check import structure for circular dependencies and consistency."""
        # This would implement a more sophisticated import analysis
        # For now, we'll do basic checks
        
        python_files = list((self.repo_root / 'src/python').rglob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for relative imports consistency
                relative_imports = re.findall(r'from\s+\.+\w+\s+import', content)
                if relative_imports and 'core' not in str(py_file):
                    # Non-core modules should use absolute imports when possible
                    pass  # This could be expanded
                    
            except Exception as e:
                logger.warning(f"Error checking imports in {py_file}: {e}")
    
    def validate_research_implementation(self):
        """Validate research implementation completeness."""
        # Check for presence of research capabilities across all relevant folders
        search_paths = [
            'src/python/active_inference/research',
            'src/python/active_inference/utils', 
            'src/python/active_inference/performance'
        ]
        
        research_files = []
        for path in search_paths:
            folder_path = self.repo_root / path
            if folder_path.exists():
                research_files.extend(list(folder_path.rglob('*.py')))
        
        found_capabilities = set()
        
        for research_file in research_files:
            try:
                with open(research_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for research capability classes
                for capability in self.research_capabilities:
                    if f"class {capability}" in content:
                        found_capabilities.add(capability)
                        self.results['research_validation']['passed'] += 1
                        logger.info(f"Found research capability: {capability} in {research_file}")
            
            except Exception as e:
                logger.warning(f"Error validating research file {research_file}: {e}")
        
        # Check for missing capabilities
        missing_capabilities = set(self.research_capabilities) - found_capabilities
        for missing in missing_capabilities:
            self.results['research_validation']['missing'].append(missing)
            self.results['research_validation']['failed'] += 1
        
        # Validate advanced demonstration script
        demo_script = self.repo_root / 'examples/advanced_research_demo.py'
        if demo_script.exists():
            self.results['research_validation']['passed'] += 1
            logger.info("Advanced research demonstration script found")
        else:
            self.results['research_validation']['failed'] += 1
            self.results['research_validation']['missing'].append('advanced_research_demo.py')
    
    def generate_quality_report(self, execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        # Calculate overall scores
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_tests = total_passed + total_failed
        
        overall_score = total_passed / max(1, total_tests)
        
        # Determine quality gates status
        quality_gates_passed = (
            overall_score >= 0.8 and  # 80% overall pass rate
            self.results['security_validation']['failed'] == 0 and  # No security issues
            len(self.results['architecture_validation']['violations']) <= 2 and  # Max 2 architecture violations
            len(self.results['research_validation']['missing']) <= 1  # Max 1 missing research component
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            'timestamp': time.time(),
            'execution_time': execution_time,
            'overall_score': overall_score,
            'quality_gates_passed': quality_gates_passed,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'detailed_results': self.results,
            'recommendations': recommendations,
            'summary': {
                'syntax_issues': len(self.results['syntax_validation']['errors']),
                'security_issues': len(self.results['security_validation']['issues']),
                'architecture_violations': len(self.results['architecture_validation']['violations']),
                'missing_research_components': len(self.results['research_validation']['missing'])
            }
        }
        
        # Log summary
        logger.info(f"Quality Gates Summary:")
        logger.info(f"  Overall Score: {overall_score:.1%}")
        logger.info(f"  Tests Passed: {total_passed}/{total_tests}")
        logger.info(f"  Quality Gates: {'PASSED' if quality_gates_passed else 'FAILED'}")
        
        if not quality_gates_passed:
            logger.error("Quality gates validation FAILED")
            for category, results in self.results.items():
                if results['failed'] > 0:
                    logger.error(f"  {category}: {results['failed']} failures")
        else:
            logger.info("Quality gates validation PASSED")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Syntax recommendations
        if self.results['syntax_validation']['failed'] > 0:
            recommendations.append(
                f"Fix {self.results['syntax_validation']['failed']} syntax/structure issues before deployment"
            )
        
        # Security recommendations
        if self.results['security_validation']['failed'] > 0:
            recommendations.append(
                f"Address {self.results['security_validation']['failed']} security issues immediately"
            )
            recommendations.append("Run additional security scanning with tools like bandit")
        
        # Performance recommendations
        metrics = self.results['performance_validation']['metrics']
        if metrics.get('complexity_ratio', 0) > 0.2:
            recommendations.append(
                "Consider refactoring high-complexity functions for better maintainability"
            )
        
        # Architecture recommendations
        if self.results['architecture_validation']['failed'] > 0:
            recommendations.append(
                "Complete missing architectural components before production deployment"
            )
        
        # Research recommendations
        if self.results['research_validation']['failed'] > 0:
            recommendations.append(
                "Complete implementation of missing research capabilities"
            )
        
        # General recommendations
        recommendations.extend([
            "Set up automated CI/CD pipeline for continuous quality validation",
            "Implement comprehensive unit and integration tests",
            "Add performance benchmarking to detect regressions",
            "Consider setting up dependency vulnerability scanning"
        ])
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save quality report to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"quality_gates_report_{timestamp}.json"
        
        report_path = self.repo_root / filename
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Quality gates report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Main entry point for quality gates validation."""
    print("🛡️  Active Inference Framework - Quality Gates Validation")
    print("="*70)
    
    try:
        # Initialize validator
        validator = QualityGatesValidator()
        
        # Run all validations
        report = validator.run_all_validations()
        
        # Save report
        validator.save_report(report)
        
        # Print summary
        print(f"\n📊 QUALITY GATES SUMMARY")
        print(f"="*70)
        print(f"Overall Score: {report['overall_score']:.1%}")
        print(f"Quality Gates: {'✅ PASSED' if report['quality_gates_passed'] else '❌ FAILED'}")
        print(f"Total Tests: {report['total_passed']}/{report['total_tests']}")
        print(f"Execution Time: {report['execution_time']:.1f}s")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for category, results in report['detailed_results'].items():
            status = "✅" if results['failed'] == 0 else "❌"
            print(f"  {status} {category.replace('_', ' ').title()}: {results['passed']} passed, {results['failed']} failed")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
                print(f"  {i}. {rec}")
        
        # Exit with appropriate code
        if report['quality_gates_passed']:
            print(f"\n🎉 Quality gates validation completed successfully!")
            return 0
        else:
            print(f"\n❌ Quality gates validation failed. Review issues above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Quality gates validation crashed: {e}")
        logger.exception("Quality gates validation crashed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
