"""
Advanced telemetry system for Active Inference agents.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging
import json
from datetime import datetime


@dataclass
class TelemetryEvent:
    """Individual telemetry event."""
    timestamp: float
    agent_id: str
    event_type: str
    data: Dict[str, Any]
    session_id: Optional[str] = None
    episode_id: Optional[str] = None


class AgentTelemetry:
    """
    Comprehensive telemetry system for Active Inference agents.
    
    Tracks performance, behavior patterns, learning progress, and system metrics
    across multiple agents and environments.
    """
    
    def __init__(self,
                 buffer_size: int = 10000,
                 aggregation_interval: float = 10.0,
                 enable_real_time: bool = True):
        """
        Initialize agent telemetry system.
        
        Args:
            buffer_size: Maximum number of events to buffer
            aggregation_interval: Seconds between metric aggregations
            enable_real_time: Enable real-time metric computation
        """
        self.buffer_size = buffer_size
        self.aggregation_interval = aggregation_interval
        self.enable_real_time = enable_real_time
        
        # Event storage
        self._events: deque = deque(maxlen=buffer_size)
        self._aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._agent_sessions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Real-time tracking
        self._real_time_metrics: Dict[str, Any] = {}
        self._performance_baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Threading for aggregation
        self._aggregation_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger("AgentTelemetry")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Start aggregation if real-time enabled
        if self.enable_real_time:
            self.start_real_time_processing()
        
        self.logger.info("AgentTelemetry initialized")
    
    def record_event(self,
                    agent_id: str,
                    event_type: str,
                    data: Dict[str, Any],
                    session_id: Optional[str] = None,
                    episode_id: Optional[str] = None) -> None:
        """Record a telemetry event."""
        event = TelemetryEvent(
            timestamp=time.time(),
            agent_id=agent_id,
            event_type=event_type,
            data=data.copy(),
            session_id=session_id,
            episode_id=episode_id
        )
        
        with self._lock:
            self._events.append(event)
        
        # Update real-time metrics immediately for critical events
        if event_type in ['action_taken', 'belief_updated', 'reward_received']:
            self._update_real_time_metrics(event)
    
    def record_agent_step(self,
                         agent_id: str,
                         observation: np.ndarray,
                         action: np.ndarray,
                         belief_state: Dict[str, Any],
                         free_energy: float,
                         reward: Optional[float] = None,
                         episode_id: Optional[str] = None) -> None:
        """Record a complete agent step with all relevant data."""
        step_data = {
            'observation_norm': float(np.linalg.norm(observation)),
            'observation_mean': float(np.mean(observation)),
            'observation_std': float(np.std(observation)),
            'action_norm': float(np.linalg.norm(action)),
            'action_mean': float(np.mean(action)),
            'free_energy': float(free_energy),
            'belief_entropy': self._compute_belief_entropy(belief_state),
            'belief_confidence': self._compute_belief_confidence(belief_state)
        }
        
        if reward is not None:
            step_data['reward'] = float(reward)
        
        self.record_event(
            agent_id=agent_id,
            event_type='agent_step',
            data=step_data,
            episode_id=episode_id
        )
    
    def record_learning_event(self,
                             agent_id: str,
                             learning_type: str,
                             metrics: Dict[str, float],
                             episode_id: Optional[str] = None) -> None:
        """Record learning-related events."""
        self.record_event(
            agent_id=agent_id,
            event_type='learning_event',
            data={
                'learning_type': learning_type,
                'metrics': metrics,
                'timestamp': time.time()
            },
            episode_id=episode_id
        )
    
    def record_performance_benchmark(self,
                                   agent_id: str,
                                   benchmark_name: str,
                                   score: float,
                                   additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Record performance benchmark results."""
        data = {
            'benchmark_name': benchmark_name,
            'score': score,
            'timestamp': time.time()
        }
        
        if additional_metrics:
            data['additional_metrics'] = additional_metrics
        
        self.record_event(
            agent_id=agent_id,
            event_type='performance_benchmark',
            data=data
        )
        
        # Update baseline for comparison
        with self._lock:
            if benchmark_name not in self._performance_baselines[agent_id]:
                self._performance_baselines[agent_id][benchmark_name] = score
            else:
                # Use exponential moving average
                alpha = 0.1
                current = self._performance_baselines[agent_id][benchmark_name]
                self._performance_baselines[agent_id][benchmark_name] = alpha * score + (1 - alpha) * current
    
    def get_agent_summary(self, agent_id: str, time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Get comprehensive summary for a specific agent."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        with self._lock:
            # Filter events for this agent and time window
            agent_events = [
                event for event in self._events
                if event.agent_id == agent_id and event.timestamp > cutoff_time
            ]
        
        if not agent_events:
            return {'agent_id': agent_id, 'message': 'No recent events'}
        
        # Compute summary statistics
        summary = {
            'agent_id': agent_id,
            'time_window_hours': time_window_hours,
            'total_events': len(agent_events),
            'event_types': defaultdict(int),
            'performance_metrics': {},
            'learning_progress': {},
            'behavioral_patterns': {}
        }
        
        # Count event types
        for event in agent_events:
            summary['event_types'][event.event_type] += 1
        
        # Compute performance metrics
        step_events = [e for e in agent_events if e.event_type == 'agent_step']
        if step_events:
            recent_fe = [e.data['free_energy'] for e in step_events[-100:]]
            recent_entropy = [e.data.get('belief_entropy', 0) for e in step_events[-100:]]
            recent_confidence = [e.data.get('belief_confidence', 0) for e in step_events[-100:]]
            
            summary['performance_metrics'] = {
                'avg_free_energy': float(np.mean(recent_fe)),
                'std_free_energy': float(np.std(recent_fe)),
                'avg_belief_entropy': float(np.mean(recent_entropy)),
                'avg_belief_confidence': float(np.mean(recent_confidence)),
                'total_steps': len(step_events)
            }
            
            # Trend analysis
            if len(recent_fe) > 10:
                early_fe = np.mean(recent_fe[:len(recent_fe)//2])
                late_fe = np.mean(recent_fe[len(recent_fe)//2:])
                summary['performance_metrics']['free_energy_trend'] = float(late_fe - early_fe)
        
        # Learning progress
        learning_events = [e for e in agent_events if e.event_type == 'learning_event']
        if learning_events:
            learning_types = defaultdict(list)
            for event in learning_events:
                learning_type = event.data.get('learning_type', 'unknown')
                metrics = event.data.get('metrics', {})
                learning_types[learning_type].append(metrics)
            
            summary['learning_progress'] = {
                ltype: {
                    'event_count': len(metrics_list),
                    'latest_metrics': metrics_list[-1] if metrics_list else {}
                }
                for ltype, metrics_list in learning_types.items()
            }
        
        # Performance benchmarks
        benchmark_events = [e for e in agent_events if e.event_type == 'performance_benchmark']
        if benchmark_events:
            benchmarks = defaultdict(list)
            for event in benchmark_events:
                benchmark_name = event.data.get('benchmark_name', 'unknown')
                score = event.data.get('score', 0)
                benchmarks[benchmark_name].append(score)
            
            summary['benchmarks'] = {
                name: {
                    'latest_score': scores[-1],
                    'avg_score': float(np.mean(scores)),
                    'best_score': float(np.max(scores)),
                    'improvement': float(scores[-1] - scores[0]) if len(scores) > 1 else 0.0
                }
                for name, scores in benchmarks.items()
            }
        
        return summary
    
    def get_multi_agent_comparison(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple agents."""
        comparison = {
            'agents': agent_ids,
            'timestamp': time.time(),
            'metrics': {}
        }
        
        # Get summaries for all agents
        summaries = {}
        for agent_id in agent_ids:
            summaries[agent_id] = self.get_agent_summary(agent_id, time_window_hours=1.0)
        
        # Compare key metrics
        metrics_to_compare = ['avg_free_energy', 'avg_belief_entropy', 'total_steps']
        
        for metric in metrics_to_compare:
            values = []
            valid_agents = []
            
            for agent_id in agent_ids:
                summary = summaries[agent_id]
                perf_metrics = summary.get('performance_metrics', {})
                if metric in perf_metrics:
                    values.append(perf_metrics[metric])
                    valid_agents.append(agent_id)
            
            if values:
                comparison['metrics'][metric] = {
                    'values': dict(zip(valid_agents, values)),
                    'best_agent': valid_agents[np.argmin(values) if 'free_energy' in metric else np.argmax(values)],
                    'worst_agent': valid_agents[np.argmax(values) if 'free_energy' in metric else np.argmin(values)],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        return comparison
    
    def detect_performance_anomalies(self, agent_id: str) -> List[Dict[str, Any]]:
        """Detect performance anomalies for an agent."""
        summary = self.get_agent_summary(agent_id, time_window_hours=1.0)
        anomalies = []
        
        perf_metrics = summary.get('performance_metrics', {})
        if not perf_metrics:
            return anomalies
        
        # Check for anomalies based on baselines and statistical thresholds
        current_fe = perf_metrics.get('avg_free_energy', 0)
        baseline_fe = self._performance_baselines[agent_id].get('free_energy', current_fe)
        
        # Free energy anomaly (significant increase)
        if current_fe > baseline_fe * 1.5:
            anomalies.append({
                'type': 'high_free_energy',
                'severity': 'warning' if current_fe < baseline_fe * 2.0 else 'critical',
                'current_value': current_fe,
                'baseline_value': baseline_fe,
                'description': f"Free energy elevated: {current_fe:.3f} vs baseline {baseline_fe:.3f}"
            })
        
        # Belief confidence anomaly (very low confidence)
        confidence = perf_metrics.get('avg_belief_confidence', 1.0)
        if confidence < 0.3:
            anomalies.append({
                'type': 'low_confidence',
                'severity': 'warning' if confidence > 0.1 else 'critical',
                'current_value': confidence,
                'description': f"Low belief confidence: {confidence:.3f}"
            })
        
        # Performance trend anomaly (degrading performance)
        trend = perf_metrics.get('free_energy_trend', 0)
        if trend > 0.1:
            anomalies.append({
                'type': 'performance_degradation',
                'severity': 'warning',
                'trend_value': trend,
                'description': f"Performance degrading: free energy increasing by {trend:.3f}"
            })
        
        return anomalies
    
    def export_telemetry_data(self,
                             filepath: str,
                             agent_ids: Optional[List[str]] = None,
                             time_window_hours: Optional[float] = None) -> None:
        """Export telemetry data to file for analysis."""
        cutoff_time = time.time() - (time_window_hours * 3600) if time_window_hours else 0
        
        with self._lock:
            # Filter events
            filtered_events = []
            for event in self._events:
                if event.timestamp > cutoff_time:
                    if agent_ids is None or event.agent_id in agent_ids:
                        filtered_events.append({
                            'timestamp': event.timestamp,
                            'agent_id': event.agent_id,
                            'event_type': event.event_type,
                            'data': event.data,
                            'session_id': event.session_id,
                            'episode_id': event.episode_id
                        })
        
        export_data = {
            'export_timestamp': time.time(),
            'time_window_hours': time_window_hours,
            'filtered_agent_ids': agent_ids,
            'total_events': len(filtered_events),
            'events': filtered_events,
            'performance_baselines': dict(self._performance_baselines)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Telemetry data exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export telemetry data: {e}")
            raise
    
    def start_real_time_processing(self) -> None:
        """Start real-time metric aggregation."""
        if self._aggregation_thread and self._aggregation_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True,
            name="TelemetryAggregation"
        )
        self._aggregation_thread.start()
        self.logger.info("Real-time telemetry processing started")
    
    def stop_real_time_processing(self) -> None:
        """Stop real-time metric aggregation."""
        if self._aggregation_thread and self._aggregation_thread.is_alive():
            self._stop_event.set()
            self._aggregation_thread.join(timeout=5.0)
        
        self.logger.info("Real-time telemetry processing stopped")
    
    def _aggregation_loop(self) -> None:
        """Main aggregation loop."""
        while not self._stop_event.is_set():
            try:
                self._compute_aggregated_metrics()
                self._stop_event.wait(self.aggregation_interval)
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                self._stop_event.wait(1.0)
    
    def _compute_aggregated_metrics(self) -> None:
        """Compute aggregated metrics from recent events."""
        current_time = time.time()
        window_start = current_time - self.aggregation_interval
        
        with self._lock:
            # Get recent events
            recent_events = [
                event for event in self._events
                if event.timestamp > window_start
            ]
        
        # Group by agent
        agent_events = defaultdict(list)
        for event in recent_events:
            agent_events[event.agent_id].append(event)
        
        # Compute metrics per agent
        for agent_id, events in agent_events.items():
            metrics = self._compute_agent_window_metrics(events)
            self._aggregated_metrics[agent_id][current_time] = metrics
        
        # Cleanup old aggregated metrics
        self._cleanup_old_metrics(current_time)
    
    def _compute_agent_window_metrics(self, events: List[TelemetryEvent]) -> Dict[str, Any]:
        """Compute metrics for an agent within a time window."""
        if not events:
            return {}
        
        metrics = {
            'event_count': len(events),
            'event_types': defaultdict(int)
        }
        
        # Count event types
        for event in events:
            metrics['event_types'][event.event_type] += 1
        
        # Compute step-based metrics
        step_events = [e for e in events if e.event_type == 'agent_step']
        if step_events:
            free_energies = [e.data['free_energy'] for e in step_events]
            entropies = [e.data.get('belief_entropy', 0) for e in step_events]
            
            metrics.update({
                'avg_free_energy': float(np.mean(free_energies)),
                'min_free_energy': float(np.min(free_energies)),
                'max_free_energy': float(np.max(free_energies)),
                'avg_belief_entropy': float(np.mean(entropies)) if entropies else 0.0
            })
        
        return metrics
    
    def _update_real_time_metrics(self, event: TelemetryEvent) -> None:
        """Update real-time metrics immediately."""
        agent_id = event.agent_id
        
        with self._lock:
            if agent_id not in self._real_time_metrics:
                self._real_time_metrics[agent_id] = {
                    'last_update': event.timestamp,
                    'step_count': 0,
                    'recent_free_energy': deque(maxlen=100),
                    'recent_rewards': deque(maxlen=100)
                }
            
            metrics = self._real_time_metrics[agent_id]
            metrics['last_update'] = event.timestamp
            
            if event.event_type == 'agent_step':
                metrics['step_count'] += 1
                if 'free_energy' in event.data:
                    metrics['recent_free_energy'].append(event.data['free_energy'])
                if 'reward' in event.data:
                    metrics['recent_rewards'].append(event.data['reward'])
    
    def _cleanup_old_metrics(self, current_time: float, max_age_hours: float = 24.0) -> None:
        """Remove old aggregated metrics to prevent memory bloat."""
        cutoff_time = current_time - (max_age_hours * 3600)
        
        for agent_id in list(self._aggregated_metrics.keys()):
            agent_metrics = self._aggregated_metrics[agent_id]
            self._aggregated_metrics[agent_id] = {
                timestamp: metrics
                for timestamp, metrics in agent_metrics.items()
                if timestamp > cutoff_time
            }
    
    def _compute_belief_entropy(self, belief_state: Dict[str, Any]) -> float:
        """Compute entropy of belief state."""
        try:
            entropy = 0.0
            for belief_name, belief_data in belief_state.items():
                if isinstance(belief_data, dict) and 'variance' in belief_data:
                    # Assume Gaussian belief: entropy = 0.5 * log(2πeσ²)
                    variance = belief_data['variance']
                    if isinstance(variance, (list, np.ndarray)):
                        variance = np.mean(variance)
                    if variance > 0:
                        entropy += 0.5 * np.log(2 * np.pi * np.e * variance)
            return float(entropy)
        except Exception:
            return 0.0
    
    def _compute_belief_confidence(self, belief_state: Dict[str, Any]) -> float:
        """Compute average confidence of belief state."""
        try:
            confidences = []
            for belief_name, belief_data in belief_state.items():
                if isinstance(belief_data, dict) and 'variance' in belief_data:
                    variance = belief_data['variance']
                    if isinstance(variance, (list, np.ndarray)):
                        variance = np.mean(variance)
                    # Confidence inversely related to variance
                    confidence = 1.0 / (1.0 + variance)
                    confidences.append(confidence)
            
            return float(np.mean(confidences)) if confidences else 0.0
        except Exception:
            return 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"AgentTelemetry(events={len(self._events)}, "
                f"agents={len(set(e.agent_id for e in self._events))}, "
                f"real_time={self.enable_real_time})")
    
    def __enter__(self):
        """Context manager entry."""
        if self.enable_real_time:
            self.start_real_time_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enable_real_time:
            self.stop_real_time_processing()