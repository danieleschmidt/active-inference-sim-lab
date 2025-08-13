"""
Statistical analysis and performance evaluation tools for Active Inference research.

This module provides comprehensive analysis tools for understanding
agent behavior, performance characteristics, and research insights.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
import json
from pathlib import Path

from ..core.agent import ActiveInferenceAgent
from ..core.beliefs import BeliefState
from .benchmarks import BenchmarkResult
from .experiments import ExperimentResult


@dataclass
class AnalysisResult:
    """Result from statistical analysis."""
    analysis_type: str
    metrics: Dict[str, float]
    insights: List[str]
    visualizations: Dict[str, Any]
    statistical_tests: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for Active Inference experiments.
    
    Provides hypothesis testing, effect size calculation, and
    scientific rigor for research validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("StatisticalAnalyzer")
    
    def analyze_learning_curves(self, 
                               experiment_results: List[ExperimentResult],
                               significance_level: float = 0.05) -> AnalysisResult:
        """
        Analyze learning curves for statistical significance and patterns.
        
        Args:
            experiment_results: List of experiment results to analyze
            significance_level: Alpha level for statistical tests
            
        Returns:
            Statistical analysis of learning performance
        """
        
        # Extract learning curves from all experiments
        all_curves = []
        experiment_names = []
        
        for result in experiment_results:
            for run_result in result.run_results:
                all_curves.append(run_result['learning_curve'])
                experiment_names.append(result.config.name)
        
        if not all_curves:
            return AnalysisResult(
                analysis_type="learning_curves",
                metrics={},
                insights=["No learning curve data available"],
                visualizations={},
                statistical_tests={},
                metadata={}
            )
        
        # Normalize curve lengths
        min_length = min(len(curve) for curve in all_curves)
        normalized_curves = [curve[:min_length] for curve in all_curves]
        
        # Statistical tests
        statistical_tests = {}
        
        # 1. Test for learning (performance improvement over time)
        learning_slopes = []
        for curve in normalized_curves:
            if len(curve) > 1:
                slope, _, _, p_value, _ = stats.linregress(range(len(curve)), curve)
                learning_slopes.append(slope)
        
        if learning_slopes:
            # Test if slopes are significantly positive (learning)
            t_stat, p_value = stats.ttest_1samp(learning_slopes, 0)
            statistical_tests['learning_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'effect_size': np.mean(learning_slopes) / (np.std(learning_slopes) + 1e-8)
            }
        
        # 2. Convergence analysis
        convergence_metrics = self._analyze_convergence(normalized_curves)
        statistical_tests['convergence'] = convergence_metrics
        
        # 3. Performance variance analysis
        final_performances = [curve[-1] for curve in normalized_curves]
        variance_metrics = {
            'final_performance_mean': np.mean(final_performances),
            'final_performance_std': np.std(final_performances),
            'coefficient_of_variation': np.std(final_performances) / (abs(np.mean(final_performances)) + 1e-8)
        }
        statistical_tests['variance'] = variance_metrics
        
        # Generate insights
        insights = self._generate_learning_insights(statistical_tests, normalized_curves)
        
        # Compute summary metrics
        metrics = {
            'mean_learning_slope': np.mean(learning_slopes) if learning_slopes else 0,
            'learning_consistency': statistical_tests['learning_test']['significant'] if 'learning_test' in statistical_tests else False,
            'convergence_rate': convergence_metrics.get('convergence_rate', 0),
            'performance_stability': 1.0 - variance_metrics['coefficient_of_variation']
        }
        
        return AnalysisResult(
            analysis_type="learning_curves",
            metrics=metrics,
            insights=insights,
            visualizations={'learning_curves': normalized_curves},
            statistical_tests=statistical_tests,
            metadata={
                'n_curves': len(normalized_curves),
                'curve_length': min_length,
                'experiments': experiment_names
            }
        )
    
    def _analyze_convergence(self, curves: List[List[float]]) -> Dict[str, float]:
        """Analyze convergence properties of learning curves."""
        
        convergence_episodes = []
        
        for curve in curves:
            if len(curve) < 10:
                continue
            
            # Find convergence point (when variance stabilizes)
            window_size = max(5, len(curve) // 10)
            
            for i in range(window_size, len(curve) - window_size):
                early_window = curve[i-window_size:i]
                late_window = curve[i:i+window_size]
                
                early_var = np.var(early_window)
                late_var = np.var(late_window)
                
                # Convergence if variance decreases significantly
                if late_var < early_var * 0.5 and np.mean(late_window) > np.mean(early_window):
                    convergence_episodes.append(i)
                    break
        
        convergence_rate = len(convergence_episodes) / len(curves)
        mean_convergence = np.mean(convergence_episodes) if convergence_episodes else float('inf')
        
        return {
            'convergence_rate': convergence_rate,
            'mean_convergence_episode': mean_convergence,
            'convergence_consistency': 1.0 - (np.std(convergence_episodes) / (mean_convergence + 1e-8)) if convergence_episodes else 0
        }
    
    def _generate_learning_insights(self, 
                                  statistical_tests: Dict, 
                                  curves: List[List[float]]) -> List[str]:
        """Generate insights from learning curve analysis."""
        
        insights = []
        
        # Learning effectiveness
        if 'learning_test' in statistical_tests:
            learning_test = statistical_tests['learning_test']
            if learning_test['significant'] and learning_test['t_statistic'] > 0:
                insights.append("Agents demonstrate statistically significant learning")
                
                effect_size = abs(learning_test['effect_size'])
                if effect_size > 0.8:
                    insights.append("Learning effect size is large (Cohen's d > 0.8)")
                elif effect_size > 0.5:
                    insights.append("Learning effect size is medium (Cohen's d > 0.5)")
                else:
                    insights.append("Learning effect size is small (Cohen's d < 0.5)")
            else:
                insights.append("No statistically significant learning detected")
        
        # Convergence insights
        if 'convergence' in statistical_tests:
            conv = statistical_tests['convergence']
            if conv['convergence_rate'] > 0.7:
                insights.append(f"High convergence rate ({conv['convergence_rate']:.1%} of runs)")
            elif conv['convergence_rate'] > 0.3:
                insights.append(f"Moderate convergence rate ({conv['convergence_rate']:.1%} of runs)")
            else:
                insights.append(f"Low convergence rate ({conv['convergence_rate']:.1%} of runs)")
        
        # Performance consistency
        if 'variance' in statistical_tests:
            var = statistical_tests['variance']
            cv = var['coefficient_of_variation']
            if cv < 0.2:
                insights.append("Highly consistent performance across runs")
            elif cv < 0.5:
                insights.append("Moderately consistent performance")
            else:
                insights.append("High performance variability across runs")
        
        return insights
    
    def compare_agents(self, 
                      results1: ExperimentResult,
                      results2: ExperimentResult,
                      comparison_name: str = "Agent Comparison") -> AnalysisResult:
        """
        Statistical comparison between two agents or conditions.
        
        Args:
            results1: First agent/condition results
            results2: Second agent/condition results
            comparison_name: Name for this comparison
            
        Returns:
            Statistical comparison analysis
        """
        
        # Extract performance data
        perf1 = [r['final_performance'] for r in results1.run_results]
        perf2 = [r['final_performance'] for r in results2.run_results]
        
        # Statistical tests
        statistical_tests = {}
        
        # 1. T-test for mean difference
        t_stat, p_value = stats.ttest_ind(perf1, perf2)
        statistical_tests['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'direction': 'agent1_better' if t_stat > 0 else 'agent2_better'
        }
        
        # 2. Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(perf1) + np.var(perf2)) / 2)
        cohens_d = (np.mean(perf1) - np.mean(perf2)) / (pooled_std + 1e-8)
        statistical_tests['effect_size'] = {
            'cohens_d': cohens_d,
            'magnitude': self._interpret_effect_size(abs(cohens_d))
        }
        
        # 3. Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(perf1, perf2, alternative='two-sided')
        statistical_tests['mann_whitney'] = {
            'u_statistic': u_stat,
            'p_value': u_p_value,
            'significant': u_p_value < 0.05
        }
        
        # 4. Variance equality test
        f_stat, f_p_value = stats.levene(perf1, perf2)
        statistical_tests['variance_test'] = {
            'f_statistic': f_stat,
            'p_value': f_p_value,
            'equal_variances': f_p_value > 0.05
        }
        
        # Generate insights
        insights = self._generate_comparison_insights(statistical_tests, perf1, perf2, 
                                                     results1.config.name, results2.config.name)
        
        # Summary metrics
        metrics = {
            'agent1_mean': np.mean(perf1),
            'agent2_mean': np.mean(perf2),
            'mean_difference': np.mean(perf1) - np.mean(perf2),
            'relative_improvement': (np.mean(perf1) - np.mean(perf2)) / (abs(np.mean(perf2)) + 1e-8),
            'significance_level': statistical_tests['t_test']['p_value']
        }
        
        return AnalysisResult(
            analysis_type="agent_comparison",
            metrics=metrics,
            insights=insights,
            visualizations={
                'performance_distributions': {'agent1': perf1, 'agent2': perf2}
            },
            statistical_tests=statistical_tests,
            metadata={
                'comparison_name': comparison_name,
                'n_runs_agent1': len(perf1),
                'n_runs_agent2': len(perf2)
            }
        )
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_comparison_insights(self, 
                                    statistical_tests: Dict,
                                    perf1: List[float],
                                    perf2: List[float],
                                    name1: str,
                                    name2: str) -> List[str]:
        """Generate insights from agent comparison."""
        
        insights = []
        
        # Statistical significance
        if statistical_tests['t_test']['significant']:
            direction = statistical_tests['t_test']['direction']
            better_agent = name1 if direction == 'agent1_better' else name2
            p_val = statistical_tests['t_test']['p_value']
            insights.append(f"{better_agent} performs significantly better (p={p_val:.3f})")
        else:
            insights.append("No statistically significant difference between agents")
        
        # Effect size interpretation
        effect_mag = statistical_tests['effect_size']['magnitude']
        cohens_d = statistical_tests['effect_size']['cohens_d']
        insights.append(f"Effect size is {effect_mag} (Cohen's d = {cohens_d:.3f})")
        
        # Practical significance
        mean_diff = np.mean(perf1) - np.mean(perf2)
        relative_diff = abs(mean_diff) / (abs(np.mean(perf2)) + 1e-8) * 100
        if relative_diff > 20:
            insights.append(f"Large practical difference ({relative_diff:.1f}% improvement)")
        elif relative_diff > 5:
            insights.append(f"Moderate practical difference ({relative_diff:.1f}% improvement)")
        else:
            insights.append(f"Small practical difference ({relative_diff:.1f}% improvement)")
        
        # Consistency comparison
        cv1 = np.std(perf1) / (abs(np.mean(perf1)) + 1e-8)
        cv2 = np.std(perf2) / (abs(np.mean(perf2)) + 1e-8)
        
        if abs(cv1 - cv2) > 0.2:
            more_consistent = name1 if cv1 < cv2 else name2
            insights.append(f"{more_consistent} shows more consistent performance")
        
        return insights


class PerformanceAnalyzer:
    """Analyzer for detailed performance characteristics."""
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceAnalyzer")
    
    def analyze_sample_efficiency(self, 
                                 experiment_results: List[ExperimentResult]) -> AnalysisResult:
        """Analyze sample efficiency characteristics."""
        
        convergence_data = []
        efficiency_metrics = {}
        
        for result in experiment_results:
            for run_result in result.run_results:
                if run_result['convergence_episode'] is not None:
                    convergence_data.append({
                        'experiment': result.config.name,
                        'convergence_episode': run_result['convergence_episode'],
                        'final_performance': run_result['final_performance']
                    })
        
        if not convergence_data:
            return AnalysisResult(
                analysis_type="sample_efficiency",
                metrics={'efficiency_score': 0.0},
                insights=["No convergence data available"],
                visualizations={},
                statistical_tests={},
                metadata={}
            )
        
        # Compute efficiency metrics
        convergence_episodes = [d['convergence_episode'] for d in convergence_data]
        final_performances = [d['final_performance'] for d in convergence_data]
        
        efficiency_metrics = {
            'mean_convergence_episode': np.mean(convergence_episodes),
            'median_convergence_episode': np.median(convergence_episodes),
            'convergence_efficiency': 1.0 / (np.mean(convergence_episodes) + 1),  # Inverse relationship
            'performance_efficiency': np.mean(final_performances) / (np.mean(convergence_episodes) + 1)
        }
        
        # Statistical tests
        statistical_tests = {
            'convergence_distribution': {
                'mean': np.mean(convergence_episodes),
                'std': np.std(convergence_episodes),
                'skewness': stats.skew(convergence_episodes),
                'kurtosis': stats.kurtosis(convergence_episodes)
            }
        }
        
        # Generate insights
        insights = []
        mean_conv = efficiency_metrics['mean_convergence_episode']
        if mean_conv < 50:
            insights.append(f"Very fast convergence (avg {mean_conv:.1f} episodes)")
        elif mean_conv < 100:
            insights.append(f"Fast convergence (avg {mean_conv:.1f} episodes)")
        elif mean_conv < 200:
            insights.append(f"Moderate convergence (avg {mean_conv:.1f} episodes)")
        else:
            insights.append(f"Slow convergence (avg {mean_conv:.1f} episodes)")
        
        return AnalysisResult(
            analysis_type="sample_efficiency",
            metrics=efficiency_metrics,
            insights=insights,
            visualizations={'convergence_data': convergence_data},
            statistical_tests=statistical_tests,
            metadata={'n_convergent_runs': len(convergence_data)}
        )


class BehaviorAnalyzer:
    """Analyzer for emergent behavior patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger("BehaviorAnalyzer")
    
    def analyze_exploration_patterns(self, 
                                   agent: ActiveInferenceAgent,
                                   trajectory_data: List[Dict]) -> AnalysisResult:
        """Analyze exploration vs exploitation patterns."""
        
        if not trajectory_data:
            return AnalysisResult(
                analysis_type="exploration_patterns",
                metrics={},
                insights=["No trajectory data available"],
                visualizations={},
                statistical_tests={},
                metadata={}
            )
        
        # Extract action patterns
        all_actions = []
        action_entropies = []
        
        for episode_trajectory in trajectory_data:
            episode_actions = [step['action'] for step in episode_trajectory]
            all_actions.extend(episode_actions)
            
            # Compute action entropy for this episode
            if episode_actions:
                action_magnitudes = [np.linalg.norm(action) for action in episode_actions]
                if len(set(action_magnitudes)) > 1:
                    entropy = stats.entropy(np.histogram(action_magnitudes, bins=10)[0] + 1e-8)
                    action_entropies.append(entropy)
        
        # Behavioral metrics
        behavior_metrics = {}
        
        if all_actions:
            action_array = np.array(all_actions)
            behavior_metrics = {
                'action_diversity': np.std(action_array.flatten()),
                'mean_action_magnitude': np.mean([np.linalg.norm(a) for a in all_actions]),
                'action_entropy': np.mean(action_entropies) if action_entropies else 0,
                'exploration_consistency': 1.0 - (np.std(action_entropies) / (np.mean(action_entropies) + 1e-8)) if action_entropies else 0
            }
        
        # Generate insights
        insights = []
        if 'action_diversity' in behavior_metrics:
            diversity = behavior_metrics['action_diversity']
            if diversity > 1.0:
                insights.append("High action diversity - extensive exploration")
            elif diversity > 0.5:
                insights.append("Moderate action diversity - balanced exploration")
            else:
                insights.append("Low action diversity - focused behavior")
        
        return AnalysisResult(
            analysis_type="exploration_patterns",
            metrics=behavior_metrics,
            insights=insights,
            visualizations={'action_patterns': all_actions[:1000]},  # Limit for storage
            statistical_tests={},
            metadata={'n_actions': len(all_actions), 'n_episodes': len(trajectory_data)}
        )


class NoveltyDetector:
    """Detector for novel or anomalous behaviors."""
    
    def __init__(self):
        self.logger = logging.getLogger("NoveltyDetector")
    
    def detect_anomalous_performance(self, 
                                   experiment_results: List[ExperimentResult],
                                   threshold_std: float = 2.0) -> AnalysisResult:
        """Detect anomalous performance patterns."""
        
        all_performances = []
        run_metadata = []
        
        for result in experiment_results:
            for run_result in result.run_results:
                all_performances.append(run_result['final_performance'])
                run_metadata.append({
                    'experiment': result.config.name,
                    'run_id': run_result['run_id'],
                    'performance': run_result['final_performance']
                })
        
        if len(all_performances) < 3:
            return AnalysisResult(
                analysis_type="anomaly_detection",
                metrics={},
                insights=["Insufficient data for anomaly detection"],
                visualizations={},
                statistical_tests={},
                metadata={}
            )
        
        # Statistical outlier detection
        mean_perf = np.mean(all_performances)
        std_perf = np.std(all_performances)
        
        outliers = []
        for i, (perf, metadata) in enumerate(zip(all_performances, run_metadata)):
            z_score = abs(perf - mean_perf) / (std_perf + 1e-8)
            if z_score > threshold_std:
                outliers.append({
                    'index': i,
                    'performance': perf,
                    'z_score': z_score,
                    'metadata': metadata
                })
        
        # Novelty metrics
        novelty_metrics = {
            'outlier_rate': len(outliers) / len(all_performances),
            'performance_variability': std_perf / (abs(mean_perf) + 1e-8),
            'max_z_score': max([o['z_score'] for o in outliers]) if outliers else 0
        }
        
        # Generate insights
        insights = []
        if novelty_metrics['outlier_rate'] > 0.1:
            insights.append(f"High outlier rate ({novelty_metrics['outlier_rate']:.1%})")
        elif novelty_metrics['outlier_rate'] > 0.05:
            insights.append(f"Moderate outlier rate ({novelty_metrics['outlier_rate']:.1%})")
        else:
            insights.append("Low outlier rate - consistent performance")
        
        if outliers:
            best_outlier = max(outliers, key=lambda x: x['performance'])
            insights.append(f"Best outlier performance: {best_outlier['performance']:.3f}")
        
        return AnalysisResult(
            analysis_type="anomaly_detection",
            metrics=novelty_metrics,
            insights=insights,
            visualizations={'outliers': outliers},
            statistical_tests={
                'outlier_analysis': {
                    'mean_performance': mean_perf,
                    'std_performance': std_perf,
                    'threshold_std': threshold_std,
                    'n_outliers': len(outliers)
                }
            },
            metadata={'total_runs': len(all_performances)}
        )