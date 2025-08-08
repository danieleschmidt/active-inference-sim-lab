"""
Benchmarking framework for Active Inference implementations.

This module provides comprehensive benchmarks to evaluate Active Inference
agents against published results and baseline methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import time
import logging
from abc import ABC, abstractmethod
import json
from pathlib import Path

from ..core.agent import ActiveInferenceAgent
from .validation import ValidationResult


@dataclass
class BenchmarkResult:
    """Result from a benchmark test."""
    benchmark_name: str
    agent_id: str
    score: float
    baseline_score: Optional[float]
    relative_performance: Optional[float]  # agent_score / baseline_score
    execution_time: float
    sample_efficiency: Optional[float]
    convergence_steps: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AXIOMBenchmark:
    """
    Benchmark reproducing AXIOM results from the published paper.
    
    Tests rapid learning on Atari-style games with minimal data,
    aiming to reproduce the 3-minute Pong mastery claim.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AXIOMBenchmark")
        
        # AXIOM target performance from paper
        self.axiom_targets = {
            'pong': {
                'final_score': 21.0,
                'training_minutes': 3.0,
                'sample_efficiency': 10.0  # Episodes to reach competence
            },
            'breakout': {
                'final_score': 400.0,
                'training_minutes': 5.0,
                'sample_efficiency': 15.0
            }
        }
    
    def benchmark_rapid_learning(self,
                                agent: ActiveInferenceAgent,
                                environment: Any,
                                game_name: str = 'pong',
                                max_episodes: int = 20,
                                target_time_minutes: float = 3.0) -> BenchmarkResult:
        """
        Benchmark rapid learning capability.
        
        Tests if agent can reach expert performance in minimal training time,
        reproducing AXIOM's rapid learning claims.
        """
        start_time = time.time()
        
        if game_name not in self.axiom_targets:
            raise ValueError(f"Unknown game: {game_name}")
        
        target = self.axiom_targets[game_name]
        
        scores = []
        episode_rewards = []
        convergence_episode = None
        
        try:
            for episode in range(max_episodes):
                episode_start = time.time()
                
                # Check time limit
                elapsed_minutes = (episode_start - start_time) / 60.0
                if elapsed_minutes > target_time_minutes:
                    break
                
                # Run episode
                obs = environment.reset()
                agent.reset(obs)
                
                episode_reward = 0
                steps = 0
                
                while steps < 1000:  # Max steps per episode
                    action = agent.act(obs)
                    obs, reward, done = environment.step(action)
                    agent.update_model(obs, action, reward)
                    
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                # Check for convergence to target performance
                if len(episode_rewards) >= 3:
                    recent_avg = np.mean(episode_rewards[-3:])
                    if recent_avg >= target['final_score'] * 0.8 and convergence_episode is None:
                        convergence_episode = episode
                
                self.logger.info(f"Episode {episode}: reward={episode_reward:.1f}, "
                               f"time={elapsed_minutes:.1f}min")
            
            # Compute final metrics
            final_score = np.mean(episode_rewards[-3:]) if len(episode_rewards) >= 3 else np.mean(episode_rewards)
            execution_time = time.time() - start_time
            
            # Sample efficiency: episodes to reach 80% of target
            sample_efficiency = convergence_episode if convergence_episode is not None else max_episodes
            
            # Performance relative to AXIOM target
            relative_performance = final_score / target['final_score']
            
            # Overall score (weighted combination)
            score = 0.0
            
            # Performance component (40%)
            perf_score = min(1.0, relative_performance) * 0.4
            score += perf_score
            
            # Time efficiency component (30%)
            time_ratio = (execution_time / 60.0) / target_time_minutes
            time_score = max(0.0, 1.0 - time_ratio) * 0.3
            score += time_score
            
            # Sample efficiency component (30%)
            sample_ratio = sample_efficiency / target['sample_efficiency']
            sample_score = max(0.0, 2.0 - sample_ratio) * 0.3  # Better than target gets bonus
            score += sample_score
            
            return BenchmarkResult(
                benchmark_name=f"axiom_{game_name}",
                agent_id=agent.agent_id,
                score=score,
                baseline_score=target['final_score'],
                relative_performance=relative_performance,
                execution_time=execution_time,
                sample_efficiency=sample_efficiency,
                convergence_steps=convergence_episode,
                metadata={
                    'final_score': final_score,
                    'target_score': target['final_score'],
                    'episode_rewards': episode_rewards,
                    'target_time_minutes': target_time_minutes,
                    'actual_time_minutes': execution_time / 60.0,
                    'episodes_completed': len(episode_rewards)
                }
            )
            
        except Exception as e:
            self.logger.error(f"AXIOM benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name=f"axiom_{game_name}",
                agent_id=agent.agent_id,
                score=0.0,
                baseline_score=target['final_score'],
                relative_performance=0.0,
                execution_time=time.time() - start_time,
                sample_efficiency=max_episodes,
                convergence_steps=None,
                metadata={'error': str(e)}
            )


class ComparativeBenchmark:
    """
    Comparative benchmark against standard RL baselines.
    
    Compares Active Inference agent performance against PPO, DQN,
    and other standard algorithms on common environments.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ComparativeBenchmark")
        
        # Known baseline performances (from literature)
        self.baselines = {
            'cartpole': {
                'ppo': {'episodes_to_solve': 200, 'final_score': 500},
                'dqn': {'episodes_to_solve': 500, 'final_score': 500},
                'random': {'episodes_to_solve': float('inf'), 'final_score': 50}
            },
            'mountain_car': {
                'ppo': {'episodes_to_solve': 1000, 'final_score': -110},
                'dqn': {'episodes_to_solve': 2000, 'final_score': -120},
                'random': {'episodes_to_solve': float('inf'), 'final_score': -200}
            }
        }
    
    def compare_sample_efficiency(self,
                                agent: ActiveInferenceAgent,
                                environment: Any,
                                env_name: str,
                                max_episodes: int = 1000,
                                target_score: float = None) -> BenchmarkResult:
        """
        Compare sample efficiency against baselines.
        
        Measures how many episodes the agent needs to reach target performance
        compared to standard RL algorithms.
        """
        start_time = time.time()
        
        if env_name not in self.baselines:
            self.logger.warning(f"No baselines available for {env_name}")
            baselines = {'ppo': {'episodes_to_solve': max_episodes}}
        else:
            baselines = self.baselines[env_name]
        
        scores = []
        convergence_episode = None
        
        # Determine target score
        if target_score is None:
            if env_name in self.baselines and 'ppo' in self.baselines[env_name]:
                target_score = self.baselines[env_name]['ppo']['final_score']
            else:
                target_score = 0  # Will be updated based on performance
        
        try:
            for episode in range(max_episodes):
                obs = environment.reset()
                agent.reset(obs)
                
                episode_reward = 0
                steps = 0
                
                while steps < 1000:
                    action = agent.act(obs)
                    obs, reward, done = environment.step(action)
                    agent.update_model(obs, action, reward)
                    
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                scores.append(episode_reward)
                
                # Check convergence
                if len(scores) >= 10:
                    recent_avg = np.mean(scores[-10:])
                    if target_score == 0:
                        # Adaptive target based on progress
                        target_score = np.mean(scores) + 2 * np.std(scores)
                    
                    if recent_avg >= target_score * 0.9 and convergence_episode is None:
                        convergence_episode = episode
                        break
                
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                    self.logger.info(f"Episode {episode}: avg_score={avg_score:.1f}")
            
            # Compute metrics
            final_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            execution_time = time.time() - start_time
            
            # Sample efficiency compared to PPO baseline
            ppo_episodes = baselines.get('ppo', {}).get('episodes_to_solve', max_episodes)
            agent_episodes = convergence_episode if convergence_episode is not None else max_episodes
            
            sample_efficiency = ppo_episodes / agent_episodes if agent_episodes > 0 else 0
            
            # Relative performance compared to PPO
            ppo_score = baselines.get('ppo', {}).get('final_score', target_score)
            relative_performance = final_score / ppo_score if ppo_score != 0 else 1.0
            
            # Overall score
            score = min(1.0, sample_efficiency * 0.6 + min(1.0, relative_performance) * 0.4)
            
            return BenchmarkResult(
                benchmark_name=f"comparative_{env_name}",
                agent_id=agent.agent_id,
                score=score,
                baseline_score=ppo_score,
                relative_performance=relative_performance,
                execution_time=execution_time,
                sample_efficiency=sample_efficiency,
                convergence_steps=convergence_episode,
                metadata={
                    'final_score': final_score,
                    'target_score': target_score,
                    'agent_episodes': agent_episodes,
                    'ppo_episodes': ppo_episodes,
                    'all_scores': scores[-50:],  # Last 50 for storage efficiency
                    'baselines': baselines
                }
            )
            
        except Exception as e:
            self.logger.error(f"Comparative benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name=f"comparative_{env_name}",
                agent_id=agent.agent_id,
                score=0.0,
                baseline_score=None,
                relative_performance=0.0,
                execution_time=time.time() - start_time,
                sample_efficiency=0.0,
                convergence_steps=None,
                metadata={'error': str(e)}
            )


class SampleEfficiencyBenchmark:
    """Dedicated benchmark for measuring sample efficiency."""
    
    def __init__(self):
        self.logger = logging.getLogger("SampleEfficiencyBenchmark")
    
    def measure_learning_curve(self,
                              agent: ActiveInferenceAgent,
                              environment: Any,
                              n_episodes: int = 500,
                              eval_frequency: int = 10) -> BenchmarkResult:
        """
        Measure detailed learning curve and sample efficiency.
        
        Tracks performance over training to characterize learning dynamics.
        """
        start_time = time.time()
        
        learning_curve = []
        eval_scores = []
        
        try:
            for episode in range(n_episodes):
                # Training episode
                obs = environment.reset()
                agent.reset(obs)
                
                episode_reward = 0
                while True:
                    action = agent.act(obs)
                    obs, reward, done = environment.step(action)
                    agent.update_model(obs, action, reward)
                    episode_reward += reward
                    
                    if done:
                        break
                
                learning_curve.append(episode_reward)
                
                # Periodic evaluation
                if episode % eval_frequency == 0:
                    eval_score = self._evaluate_agent(agent, environment, n_eval_episodes=3)
                    eval_scores.append((episode, eval_score))
                    
                    self.logger.info(f"Episode {episode}: "
                                   f"train_reward={episode_reward:.1f}, "
                                   f"eval_score={eval_score:.1f}")
            
            # Analyze learning curve
            execution_time = time.time() - start_time
            
            # Find when agent reaches 80% of final performance
            final_performance = np.mean(learning_curve[-10:])
            target_performance = final_performance * 0.8
            
            convergence_episode = None
            for i, score in enumerate(learning_curve):
                if i >= 10:  # Need some samples for averaging
                    recent_avg = np.mean(learning_curve[max(0, i-10):i])
                    if recent_avg >= target_performance:
                        convergence_episode = i
                        break
            
            # Sample efficiency score
            if convergence_episode is not None:
                sample_efficiency = 1.0 - (convergence_episode / n_episodes)
            else:
                sample_efficiency = 0.0
            
            # Learning stability (lower variance = more stable)
            if len(learning_curve) >= 20:
                late_variance = np.var(learning_curve[-20:])
                early_variance = np.var(learning_curve[:20])
                stability = max(0.0, 1.0 - late_variance / (early_variance + 1e-6))
            else:
                stability = 0.5
            
            # Overall score combines efficiency and stability
            score = sample_efficiency * 0.7 + stability * 0.3
            
            return BenchmarkResult(
                benchmark_name="sample_efficiency",
                agent_id=agent.agent_id,
                score=score,
                baseline_score=None,
                relative_performance=None,
                execution_time=execution_time,
                sample_efficiency=sample_efficiency,
                convergence_steps=convergence_episode,
                metadata={
                    'learning_curve': learning_curve,
                    'eval_scores': eval_scores,
                    'final_performance': final_performance,
                    'convergence_episode': convergence_episode,
                    'stability_score': stability,
                    'late_variance': late_variance if 'late_variance' in locals() else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Sample efficiency benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name="sample_efficiency",
                agent_id=agent.agent_id,
                score=0.0,
                baseline_score=None,
                relative_performance=None,
                execution_time=time.time() - start_time,
                sample_efficiency=0.0,
                convergence_steps=None,
                metadata={'error': str(e)}
            )
    
    def _evaluate_agent(self, agent: ActiveInferenceAgent, environment: Any, n_eval_episodes: int = 5) -> float:
        """Evaluate agent performance over multiple episodes."""
        eval_rewards = []
        
        for _ in range(n_eval_episodes):
            obs = environment.reset()
            agent.reset(obs)
            
            episode_reward = 0
            while True:
                action = agent.act(obs)
                obs, reward, done = environment.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)


class ReproducibilityBenchmark:
    """Benchmark for testing reproducibility and statistical significance."""
    
    def __init__(self):
        self.logger = logging.getLogger("ReproducibilityBenchmark")
    
    def test_reproducibility(self,
                           agent_factory: Callable[[], ActiveInferenceAgent],
                           environment_factory: Callable[[], Any],
                           n_runs: int = 10,
                           n_episodes_per_run: int = 100) -> BenchmarkResult:
        """
        Test reproducibility across multiple independent runs.
        
        Evaluates statistical significance and consistency of results.
        """
        start_time = time.time()
        
        run_results = []
        
        try:
            for run in range(n_runs):
                self.logger.info(f"Starting reproducibility run {run + 1}/{n_runs}")
                
                # Create fresh agent and environment
                agent = agent_factory()
                environment = environment_factory()
                
                # Run training
                episode_rewards = []
                for episode in range(n_episodes_per_run):
                    obs = environment.reset()
                    agent.reset(obs)
                    
                    episode_reward = 0
                    while True:
                        action = agent.act(obs)
                        obs, reward, done = environment.step(action)
                        agent.update_model(obs, action, reward)
                        episode_reward += reward
                        
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                
                # Record final performance for this run
                final_performance = np.mean(episode_rewards[-10:])
                run_results.append(final_performance)
                
                self.logger.info(f"Run {run + 1} final performance: {final_performance:.2f}")
            
            # Statistical analysis
            mean_performance = np.mean(run_results)
            std_performance = np.std(run_results)
            
            # Coefficient of variation (lower = more reproducible)
            cv = std_performance / (abs(mean_performance) + 1e-6)
            
            # Statistical significance test (compared to random performance)
            # Simple test: is mean significantly different from 0?
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(run_results, 0)
            
            # Reproducibility score (lower CV = higher score)
            reproducibility_score = max(0.0, 1.0 - cv)
            
            # Significance score (lower p-value = higher score, if mean > 0)
            if mean_performance > 0:
                significance_score = max(0.0, 1.0 - p_value)
            else:
                significance_score = 0.0
            
            # Overall score
            score = reproducibility_score * 0.6 + significance_score * 0.4
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name="reproducibility",
                agent_id="multi_agent_test",
                score=score,
                baseline_score=None,
                relative_performance=None,
                execution_time=execution_time,
                sample_efficiency=None,
                convergence_steps=None,
                metadata={
                    'n_runs': n_runs,
                    'mean_performance': mean_performance,
                    'std_performance': std_performance,
                    'coefficient_of_variation': cv,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'run_results': run_results,
                    'reproducibility_score': reproducibility_score,
                    'significance_score': significance_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Reproducibility benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_name="reproducibility",
                agent_id="multi_agent_test",
                score=0.0,
                baseline_score=None,
                relative_performance=None,
                execution_time=time.time() - start_time,
                sample_efficiency=None,
                convergence_steps=None,
                metadata={'error': str(e)}
            )


class BenchmarkSuite:
    """Comprehensive benchmark suite for Active Inference agents."""
    
    def __init__(self):
        self.logger = logging.getLogger("BenchmarkSuite")
        self.results: List[BenchmarkResult] = []
    
    def run_full_benchmark(self,
                          agent: ActiveInferenceAgent,
                          environment: Any,
                          env_name: str = "test_env") -> Dict[str, BenchmarkResult]:
        """Run complete benchmark suite."""
        
        benchmarks = {
            'axiom': AXIOMBenchmark(),
            'comparative': ComparativeBenchmark(),
            'sample_efficiency': SampleEfficiencyBenchmark(),
        }
        
        results = {}
        
        for name, benchmark in benchmarks.items():
            try:
                self.logger.info(f"Running {name} benchmark...")
                
                if name == 'axiom':
                    result = benchmark.benchmark_rapid_learning(agent, environment, env_name)
                elif name == 'comparative':
                    result = benchmark.compare_sample_efficiency(agent, environment, env_name)
                elif name == 'sample_efficiency':
                    result = benchmark.measure_learning_curve(agent, environment)
                
                results[name] = result
                self.results.append(result)
                
                self.logger.info(f"{name} benchmark completed: score={result.score:.3f}")
                
            except Exception as e:
                self.logger.error(f"{name} benchmark failed: {e}")
                results[name] = BenchmarkResult(
                    benchmark_name=name,
                    agent_id=agent.agent_id,
                    score=0.0,
                    baseline_score=None,
                    relative_performance=None,
                    execution_time=0.0,
                    sample_efficiency=None,
                    convergence_steps=None,
                    metadata={'error': str(e)}
                )
        
        return results
    
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        results_data = []
        
        for result in self.results:
            results_data.append({
                'benchmark_name': result.benchmark_name,
                'agent_id': result.agent_id,
                'score': result.score,
                'baseline_score': result.baseline_score,
                'relative_performance': result.relative_performance,
                'execution_time': result.execution_time,
                'sample_efficiency': result.sample_efficiency,
                'convergence_steps': result.convergence_steps,
                'metadata': result.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to {filepath}")