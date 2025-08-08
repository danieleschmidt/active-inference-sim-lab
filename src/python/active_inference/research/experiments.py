"""
Experimental framework for systematic Active Inference research.

This module provides tools for conducting controlled experiments,
ablation studies, and parameter sweeps to validate research hypotheses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import itertools
import logging
import json
import time
from pathlib import Path
from abc import ABC, abstractmethod

from ..core.agent import ActiveInferenceAgent
from .benchmarks import BenchmarkResult
from .validation import ValidationResult


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    parameters: Dict[str, Any]
    environment_config: Dict[str, Any]
    agent_config: Dict[str, Any]
    n_runs: int = 5
    n_episodes: int = 100
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 10
    save_trajectories: bool = False
    save_beliefs: bool = False


@dataclass
class ExperimentResult:
    """Result from an experiment."""
    config: ExperimentConfig
    run_results: List[Dict[str, Any]]
    aggregate_stats: Dict[str, float]
    statistical_significance: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentFramework:
    """
    Framework for conducting systematic Active Inference experiments.
    
    Supports controlled experiments, parameter sweeps, and ablation studies
    with proper statistical analysis and reproducibility controls.
    """
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("ExperimentFramework")
        
        # Experiment tracking
        self.experiments: List[ExperimentResult] = []
        self.current_experiment: Optional[ExperimentResult] = None
    
    def run_experiment(self,
                      config: ExperimentConfig,
                      agent_factory: Callable[[Dict], ActiveInferenceAgent],
                      environment_factory: Callable[[Dict], Any]) -> ExperimentResult:
        """
        Run a single experiment with multiple runs for statistical validity.
        
        Args:
            config: Experiment configuration
            agent_factory: Function to create agent instances
            environment_factory: Function to create environment instances
            
        Returns:
            Experiment results with statistical analysis
        """
        start_time = time.time()
        
        self.logger.info(f"Starting experiment: {config.name}")
        self.logger.info(f"Configuration: {config.parameters}")
        
        run_results = []
        
        try:
            for run in range(config.n_runs):
                self.logger.info(f"Run {run + 1}/{config.n_runs}")
                
                # Create fresh instances for each run
                agent = agent_factory(config.agent_config)
                environment = environment_factory(config.environment_config)
                
                # Run single experimental run
                run_result = self._run_single_experiment(
                    agent, environment, config, run_id=run
                )
                run_results.append(run_result)
                
                # Log progress
                if 'final_performance' in run_result:
                    self.logger.info(f"Run {run + 1} final performance: "
                                   f"{run_result['final_performance']:.3f}")
            
            # Aggregate statistics
            aggregate_stats = self._compute_aggregate_statistics(run_results)
            
            # Statistical significance tests
            significance = self._compute_statistical_significance(run_results, config)
            
            execution_time = time.time() - start_time
            
            # Create experiment result
            result = ExperimentResult(
                config=config,
                run_results=run_results,
                aggregate_stats=aggregate_stats,
                statistical_significance=significance,
                execution_time=execution_time,
                metadata={
                    'completed_runs': len(run_results),
                    'timestamp': time.time()
                }
            )
            
            self.experiments.append(result)
            self.current_experiment = result
            
            # Save results
            self._save_experiment_result(result)
            
            self.logger.info(f"Experiment {config.name} completed in {execution_time:.1f}s")
            self.logger.info(f"Average performance: {aggregate_stats.get('mean_final_performance', 0):.3f} "
                           f"± {aggregate_stats.get('std_final_performance', 0):.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment {config.name} failed: {e}")
            raise
    
    def _run_single_experiment(self,
                              agent: ActiveInferenceAgent,
                              environment: Any,
                              config: ExperimentConfig,
                              run_id: int) -> Dict[str, Any]:
        """Run a single experimental run."""
        
        learning_curve = []
        belief_history = [] if config.save_beliefs else None
        trajectory_history = [] if config.save_trajectories else None
        
        evaluation_scores = []
        convergence_episode = None
        
        for episode in range(config.n_episodes):
            obs = environment.reset()
            agent.reset(obs)
            
            episode_reward = 0
            episode_steps = 0
            episode_trajectory = [] if config.save_trajectories else None
            
            while episode_steps < config.max_steps_per_episode:
                action = agent.act(obs)
                next_obs, reward, done = environment.step(action)
                agent.update_model(next_obs, action, reward)
                
                episode_reward += reward
                episode_steps += 1
                
                # Save trajectory if requested
                if config.save_trajectories:
                    episode_trajectory.append({
                        'obs': obs.copy(),
                        'action': action.copy(),
                        'reward': reward,
                        'next_obs': next_obs.copy()
                    })
                
                obs = next_obs
                
                if done:
                    break
            
            learning_curve.append(episode_reward)
            
            # Save beliefs if requested
            if config.save_beliefs:
                belief_snapshot = {
                    'episode': episode,
                    'beliefs': agent.beliefs.get_all_beliefs(),
                    'entropy': agent.beliefs.total_entropy(),
                    'confidence': agent.beliefs.average_confidence()
                }
                belief_history.append(belief_snapshot)
            
            # Save trajectory if requested
            if config.save_trajectories:
                trajectory_history.append(episode_trajectory)
            
            # Periodic evaluation
            if episode % config.evaluation_frequency == 0:
                eval_score = self._evaluate_agent(agent, environment, n_episodes=3)
                evaluation_scores.append({'episode': episode, 'score': eval_score})
            
            # Check convergence
            if len(learning_curve) >= 10 and convergence_episode is None:
                recent_avg = np.mean(learning_curve[-10:])
                if len(learning_curve) >= 20:
                    early_avg = np.mean(learning_curve[:10])
                    if recent_avg > early_avg * 1.5:  # 50% improvement
                        convergence_episode = episode
        
        # Compute run statistics
        final_performance = np.mean(learning_curve[-10:]) if len(learning_curve) >= 10 else np.mean(learning_curve)
        
        return {
            'run_id': run_id,
            'learning_curve': learning_curve,
            'final_performance': final_performance,
            'convergence_episode': convergence_episode,
            'evaluation_scores': evaluation_scores,
            'belief_history': belief_history,
            'trajectory_history': trajectory_history,
            'agent_stats': agent.get_statistics()
        }
    
    def _evaluate_agent(self, agent: ActiveInferenceAgent, environment: Any, n_episodes: int = 5) -> float:
        """Evaluate agent performance."""
        eval_rewards = []
        
        for _ in range(n_episodes):
            obs = environment.reset()
            episode_reward = 0
            
            while True:
                action = agent.act(obs)
                obs, reward, done = environment.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def _compute_aggregate_statistics(self, run_results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate statistics across runs."""
        
        # Extract key metrics
        final_performances = [r['final_performance'] for r in run_results]
        convergence_episodes = [r['convergence_episode'] for r in run_results if r['convergence_episode'] is not None]
        
        stats = {
            'mean_final_performance': np.mean(final_performances),
            'std_final_performance': np.std(final_performances),
            'median_final_performance': np.median(final_performances),
            'min_final_performance': np.min(final_performances),
            'max_final_performance': np.max(final_performances),
        }
        
        if convergence_episodes:
            stats.update({
                'mean_convergence_episode': np.mean(convergence_episodes),
                'std_convergence_episode': np.std(convergence_episodes),
                'convergence_rate': len(convergence_episodes) / len(run_results)
            })
        else:
            stats.update({
                'mean_convergence_episode': float('inf'),
                'std_convergence_episode': 0.0,
                'convergence_rate': 0.0
            })
        
        return stats
    
    def _compute_statistical_significance(self, run_results: List[Dict], config: ExperimentConfig) -> Dict[str, float]:
        """Compute statistical significance tests."""
        
        final_performances = [r['final_performance'] for r in run_results]
        
        # Test against zero (baseline)
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(final_performances, 0)
        
        # Effect size (Cohen's d)
        mean_perf = np.mean(final_performances)
        std_perf = np.std(final_performances)
        cohens_d = mean_perf / (std_perf + 1e-8)
        
        # Confidence interval
        ci_95 = stats.t.interval(0.95, len(final_performances)-1, 
                               loc=mean_perf, 
                               scale=stats.sem(final_performances))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'confidence_interval_lower': ci_95[0],
            'confidence_interval_upper': ci_95[1],
            'sample_size': len(final_performances)
        }
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to disk."""
        
        # Create experiment directory
        exp_dir = self.output_dir / f"experiment_{result.config.name}_{int(time.time())}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_data = {
            'name': result.config.name,
            'description': result.config.description,
            'parameters': result.config.parameters,
            'environment_config': result.config.environment_config,
            'agent_config': result.config.agent_config,
            'n_runs': result.config.n_runs,
            'n_episodes': result.config.n_episodes
        }
        
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        # Save aggregate results
        results_data = {
            'aggregate_stats': result.aggregate_stats,
            'statistical_significance': result.statistical_significance,
            'execution_time': result.execution_time,
            'metadata': result.metadata
        }
        
        with open(exp_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save individual run results (without trajectories to save space)
        run_data = []
        for run_result in result.run_results:
            run_summary = {
                'run_id': run_result['run_id'],
                'final_performance': run_result['final_performance'],
                'convergence_episode': run_result['convergence_episode'],
                'learning_curve': run_result['learning_curve'],
                'evaluation_scores': run_result['evaluation_scores'],
                'agent_stats': run_result['agent_stats']
            }
            run_data.append(run_summary)
        
        with open(exp_dir / "runs.json", 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        
        self.logger.info(f"Experiment results saved to {exp_dir}")


class ControlledExperiment:
    """Framework for controlled experiments comparing conditions."""
    
    def __init__(self, framework: ExperimentFramework):
        self.framework = framework
        self.logger = logging.getLogger("ControlledExperiment")
    
    def compare_conditions(self,
                          base_config: ExperimentConfig,
                          conditions: Dict[str, Dict[str, Any]],
                          agent_factory: Callable,
                          environment_factory: Callable) -> Dict[str, ExperimentResult]:
        """
        Compare multiple experimental conditions.
        
        Args:
            base_config: Base experimental configuration
            conditions: Dictionary of condition_name -> parameter_overrides
            agent_factory: Function to create agents
            environment_factory: Function to create environments
            
        Returns:
            Dictionary of condition_name -> experiment_result
        """
        
        results = {}
        
        for condition_name, parameter_overrides in conditions.items():
            self.logger.info(f"Running condition: {condition_name}")
            
            # Create modified config for this condition
            condition_config = ExperimentConfig(
                name=f"{base_config.name}_{condition_name}",
                description=f"{base_config.description} - {condition_name}",
                parameters={**base_config.parameters, **parameter_overrides},
                environment_config=base_config.environment_config,
                agent_config={**base_config.agent_config, **parameter_overrides},
                n_runs=base_config.n_runs,
                n_episodes=base_config.n_episodes,
                max_steps_per_episode=base_config.max_steps_per_episode,
                evaluation_frequency=base_config.evaluation_frequency
            )
            
            # Run experiment for this condition
            result = self.framework.run_experiment(
                condition_config, agent_factory, environment_factory
            )
            results[condition_name] = result
        
        # Compare results
        self._compare_results(results)
        
        return results
    
    def _compare_results(self, results: Dict[str, ExperimentResult]):
        """Compare results across conditions."""
        
        self.logger.info("=== EXPERIMENTAL COMPARISON ===")
        
        for condition_name, result in results.items():
            mean_perf = result.aggregate_stats['mean_final_performance']
            std_perf = result.aggregate_stats['std_final_performance']
            p_value = result.statistical_significance['p_value']
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            self.logger.info(f"{condition_name}: {mean_perf:.3f} ± {std_perf:.3f} {significance}")
        
        # Statistical comparison between conditions
        condition_names = list(results.keys())
        if len(condition_names) >= 2:
            self._statistical_comparison(results, condition_names)
    
    def _statistical_comparison(self, results: Dict[str, ExperimentResult], condition_names: List[str]):
        """Perform statistical comparison between conditions."""
        
        from scipy import stats
        
        # Pairwise comparisons
        for i, cond1 in enumerate(condition_names):
            for cond2 in condition_names[i+1:]:
                
                perf1 = [r['final_performance'] for r in results[cond1].run_results]
                perf2 = [r['final_performance'] for r in results[cond2].run_results]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(perf1, perf2)
                
                # Effect size
                pooled_std = np.sqrt((np.var(perf1) + np.var(perf2)) / 2)
                cohens_d = (np.mean(perf1) - np.mean(perf2)) / (pooled_std + 1e-8)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                self.logger.info(f"{cond1} vs {cond2}: d={cohens_d:.3f}, p={p_value:.3f} {significance}")


class AblationStudy:
    """Framework for ablation studies."""
    
    def __init__(self, framework: ExperimentFramework):
        self.framework = framework
        self.logger = logging.getLogger("AblationStudy")
    
    def run_ablation(self,
                    base_config: ExperimentConfig,
                    components_to_ablate: Dict[str, Any],
                    agent_factory: Callable,
                    environment_factory: Callable) -> Dict[str, ExperimentResult]:
        """
        Run ablation study by systematically removing components.
        
        Args:
            base_config: Full model configuration
            components_to_ablate: Component_name -> disabled_value mapping
            agent_factory: Function to create agents
            environment_factory: Function to create environments
            
        Returns:
            Dictionary of ablation_condition -> experiment_result
        """
        
        results = {}
        
        # Baseline (full model)
        baseline_result = self.framework.run_experiment(
            base_config, agent_factory, environment_factory
        )
        results['baseline'] = baseline_result
        
        # Individual component ablations
        for component_name, disabled_value in components_to_ablate.items():
            self.logger.info(f"Ablating component: {component_name}")
            
            ablation_config = ExperimentConfig(
                name=f"{base_config.name}_ablate_{component_name}",
                description=f"Ablation study - {component_name} disabled",
                parameters=base_config.parameters,
                environment_config=base_config.environment_config,
                agent_config={**base_config.agent_config, component_name: disabled_value},
                n_runs=base_config.n_runs,
                n_episodes=base_config.n_episodes,
                max_steps_per_episode=base_config.max_steps_per_episode,
                evaluation_frequency=base_config.evaluation_frequency
            )
            
            result = self.framework.run_experiment(
                ablation_config, agent_factory, environment_factory
            )
            results[f"ablate_{component_name}"] = result
        
        # Analyze ablation effects
        self._analyze_ablation_effects(results)
        
        return results
    
    def _analyze_ablation_effects(self, results: Dict[str, ExperimentResult]):
        """Analyze the effect of each ablation."""
        
        baseline_perf = results['baseline'].aggregate_stats['mean_final_performance']
        
        self.logger.info("=== ABLATION ANALYSIS ===")
        self.logger.info(f"Baseline performance: {baseline_perf:.3f}")
        
        for condition_name, result in results.items():
            if condition_name == 'baseline':
                continue
            
            ablated_perf = result.aggregate_stats['mean_final_performance']
            performance_drop = baseline_perf - ablated_perf
            relative_drop = performance_drop / (abs(baseline_perf) + 1e-8) * 100
            
            component_name = condition_name.replace('ablate_', '')
            
            self.logger.info(f"{component_name}: {ablated_perf:.3f} "
                           f"(drop: {performance_drop:.3f}, {relative_drop:.1f}%)")


class ParameterSweep:
    """Framework for parameter sweeps and hyperparameter optimization."""
    
    def __init__(self, framework: ExperimentFramework):
        self.framework = framework
        self.logger = logging.getLogger("ParameterSweep")
    
    def grid_search(self,
                   base_config: ExperimentConfig,
                   parameter_grids: Dict[str, List[Any]],
                   agent_factory: Callable,
                   environment_factory: Callable,
                   max_combinations: int = 50) -> Dict[str, ExperimentResult]:
        """
        Perform grid search over parameter space.
        
        Args:
            base_config: Base configuration
            parameter_grids: Dict of parameter_name -> [values_to_try]
            agent_factory: Function to create agents
            environment_factory: Function to create environments
            max_combinations: Maximum number of combinations to try
            
        Returns:
            Dictionary of parameter_combination -> experiment_result
        """
        
        # Generate all parameter combinations
        param_names = list(parameter_grids.keys())
        param_values = list(parameter_grids.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            self.logger.warning(f"Too many combinations ({len(all_combinations)}), "
                              f"randomly sampling {max_combinations}")
            np.random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_combinations]
        
        results = {}
        best_performance = float('-inf')
        best_params = None
        
        for i, param_combination in enumerate(all_combinations):
            # Create parameter dict
            params = dict(zip(param_names, param_combination))
            param_str = "_".join([f"{k}={v}" for k, v in params.items()])
            
            self.logger.info(f"Testing combination {i+1}/{len(all_combinations)}: {params}")
            
            # Create config for this combination
            sweep_config = ExperimentConfig(
                name=f"{base_config.name}_sweep_{param_str}",
                description=f"Parameter sweep: {params}",
                parameters={**base_config.parameters, **params},
                environment_config=base_config.environment_config,
                agent_config={**base_config.agent_config, **params},
                n_runs=max(1, base_config.n_runs // 2),  # Fewer runs for efficiency
                n_episodes=base_config.n_episodes,
                max_steps_per_episode=base_config.max_steps_per_episode,
                evaluation_frequency=base_config.evaluation_frequency
            )
            
            try:
                result = self.framework.run_experiment(
                    sweep_config, agent_factory, environment_factory
                )
                results[param_str] = result
                
                # Track best performance
                performance = result.aggregate_stats['mean_final_performance']
                if performance > best_performance:
                    best_performance = performance
                    best_params = params
                
            except Exception as e:
                self.logger.error(f"Parameter combination {params} failed: {e}")
                continue
        
        # Report results
        self._report_sweep_results(results, best_params, best_performance)
        
        return results
    
    def _report_sweep_results(self, results: Dict[str, ExperimentResult], 
                            best_params: Dict, best_performance: float):
        """Report parameter sweep results."""
        
        self.logger.info("=== PARAMETER SWEEP RESULTS ===")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best performance: {best_performance:.3f}")
        
        # Show top 5 performers
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].aggregate_stats['mean_final_performance'], 
                              reverse=True)
        
        self.logger.info("Top 5 parameter combinations:")
        for i, (param_str, result) in enumerate(sorted_results[:5]):
            perf = result.aggregate_stats['mean_final_performance']
            self.logger.info(f"{i+1}. {param_str}: {perf:.3f}")