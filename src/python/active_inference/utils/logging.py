"""
Logging and monitoring utilities for active inference agents.

This module provides structured logging, performance monitoring,
and telemetry collection for active inference systems.
"""

import logging
import time
import json
import os
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from functools import wraps
from datetime import datetime
import numpy as np


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup structured logging for active inference.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = (
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s "
            "(%(filename)s:%(lineno)d)"
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("active_inference")
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    logger.info("Active Inference logging initialized")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f"active_inference.{name}")


class PerformanceMonitor:
    """
    Monitor performance metrics for active inference components.
    """
    
    def __init__(self, component_name: str):
        """
        Initialize performance monitor.
        
        Args:
            component_name: Name of the component being monitored
        """
        self.component_name = component_name
        self.metrics = {
            'call_count': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'last_call_time': None
        }
        self.history = []
        self.logger = get_logger(f"perf_monitor.{component_name}")
    
    @contextmanager
    def measure(self, operation_name: str = "default"):
        """
        Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
        """
        start_time = time.perf_counter()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.metrics['error_count'] += 1
            self.logger.error(f"Error in {operation_name}: {str(e)}")
            raise
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            self._update_metrics(elapsed, error_occurred)
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': operation_name,
                'duration': elapsed,
                'error': error_occurred
            })
            
            # Keep last 1000 entries
            if len(self.history) > 1000:
                self.history.pop(0)
    
    def _update_metrics(self, elapsed: float, error_occurred: bool) -> None:
        """Update performance metrics."""
        self.metrics['call_count'] += 1
        self.metrics['total_time'] += elapsed
        self.metrics['average_time'] = self.metrics['total_time'] / self.metrics['call_count']
        self.metrics['min_time'] = min(self.metrics['min_time'], elapsed)
        self.metrics['max_time'] = max(self.metrics['max_time'], elapsed)
        self.metrics['last_call_time'] = elapsed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def get_history(self, n_recent: Optional[int] = None) -> List[Dict]:
        """
        Get performance history.
        
        Args:
            n_recent: Number of recent entries to return (None for all)
            
        Returns:
            List of performance records
        """
        if n_recent is None:
            return self.history.copy()
        else:
            return self.history[-n_recent:]
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {
            'call_count': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'last_call_time': None
        }
        self.history.clear()
        self.logger.info("Performance metrics reset")


def monitor_performance(component_name: str = None, operation_name: str = "default"):
    """
    Decorator for monitoring function performance.
    
    Args:
        component_name: Name of component (uses function name if None)
        operation_name: Name of operation being monitored
        
    Returns:
        Decorator function
    """
    def decorator(func):
        nonlocal component_name
        if component_name is None:
            component_name = func.__name__
        
        monitor = PerformanceMonitor(component_name)
        
        # Attach monitor to function for external access
        func._performance_monitor = monitor
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.measure(operation_name):
                return func(*args, **kwargs)
        
        # Attach monitor to wrapper as well
        wrapper._performance_monitor = monitor
        
        return wrapper
    return decorator


class MetricsCollector:
    """
    Collect and aggregate metrics from active inference agents.
    """
    
    def __init__(self, collection_interval: float = 10.0):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Time between metric collections (seconds)
        """
        self.collection_interval = collection_interval
        self.metrics_storage = {}
        self.registered_agents = {}
        self.logger = get_logger("metrics_collector")
        self._last_collection_time = time.time()
    
    def register_agent(self, agent_id: str, agent) -> None:
        """
        Register an agent for metric collection.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to monitor
        """
        self.registered_agents[agent_id] = agent
        self.metrics_storage[agent_id] = []
        self.logger.info(f"Registered agent {agent_id} for monitoring")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current metrics from all registered agents.
        
        Returns:
            Dictionary of collected metrics
        """
        current_time = time.time()
        
        if current_time - self._last_collection_time < self.collection_interval:
            return {}  # Skip collection if interval hasn't passed
        
        collected_metrics = {
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'system': self._collect_system_metrics()
        }
        
        # Collect from each agent
        for agent_id, agent in self.registered_agents.items():
            try:
                if hasattr(agent, 'get_statistics'):
                    agent_metrics = agent.get_statistics()
                    collected_metrics['agents'][agent_id] = agent_metrics
                    
                    # Store in history
                    self.metrics_storage[agent_id].append({
                        'timestamp': current_time,
                        'metrics': agent_metrics
                    })
                    
                    # Keep last 1000 entries
                    if len(self.metrics_storage[agent_id]) > 1000:
                        self.metrics_storage[agent_id].pop(0)
                        
            except Exception as e:
                self.logger.error(f"Error collecting metrics from agent {agent_id}: {e}")
        
        self._last_collection_time = current_time
        return collected_metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
            }
        except ImportError:
            return {'note': 'psutil not available for system metrics'}
    
    def get_agent_history(self, agent_id: str, n_recent: Optional[int] = None) -> List[Dict]:
        """
        Get metric history for a specific agent.
        
        Args:
            agent_id: Agent identifier
            n_recent: Number of recent entries (None for all)
            
        Returns:
            List of historical metrics
        """
        if agent_id not in self.metrics_storage:
            return []
        
        history = self.metrics_storage[agent_id]
        if n_recent is None:
            return history.copy()
        else:
            return history[-n_recent:]
    
    def export_metrics(self, filepath: str, agent_id: Optional[str] = None) -> None:
        """
        Export metrics to file.
        
        Args:
            filepath: Path to export file
            agent_id: Specific agent to export (None for all)
        """
        if agent_id:
            data = {
                'agent_id': agent_id,
                'metrics_history': self.get_agent_history(agent_id)
            }
        else:
            data = {
                'all_agents': {
                    aid: self.get_agent_history(aid)
                    for aid in self.registered_agents.keys()
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")


class TelemetryLogger:
    """
    Log telemetry data for active inference systems.
    """
    
    def __init__(self, output_dir: str = "telemetry"):
        """
        Initialize telemetry logger.
        
        Args:
            output_dir: Directory to store telemetry files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_file = os.path.join(output_dir, f"telemetry_{self.session_id}.jsonl")
        
        self.logger = get_logger("telemetry")
        self.logger.info(f"Telemetry logging to {self.telemetry_file}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log a telemetry event.
        
        Args:
            event_type: Type of event (e.g., 'inference', 'planning', 'learning')
            data: Event data dictionary
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'event_type': event_type,
            'data': data
        }
        
        # Write to file
        with open(self.telemetry_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def log_inference_step(self, 
                          agent_id: str,
                          observations: np.ndarray,
                          beliefs_before: Dict,
                          beliefs_after: Dict,
                          free_energy: Dict) -> None:
        """Log an inference step."""
        self.log_event('inference_step', {
            'agent_id': agent_id,
            'observation_shape': observations.shape,
            'observation_stats': {
                'mean': float(observations.mean()),
                'std': float(observations.std()),
                'min': float(observations.min()),
                'max': float(observations.max())
            },
            'beliefs_entropy_change': {
                name: float(beliefs_after[name].get('entropy', 0) - beliefs_before[name].get('entropy', 0))
                for name in beliefs_before.keys()
                if name in beliefs_after
            },
            'free_energy_components': {
                'accuracy': float(free_energy.get('accuracy', 0)),
                'complexity': float(free_energy.get('complexity', 0)),
                'total': float(free_energy.get('total', 0))
            }
        })
    
    def log_planning_step(self,
                         agent_id: str,
                         selected_action: np.ndarray,
                         expected_free_energy: float,
                         n_candidates_evaluated: int) -> None:
        """Log a planning step."""
        self.log_event('planning_step', {
            'agent_id': agent_id,
            'action_shape': selected_action.shape,
            'action_stats': {
                'mean': float(selected_action.mean()),
                'std': float(selected_action.std()),
                'norm': float(np.linalg.norm(selected_action))
            },
            'expected_free_energy': float(expected_free_energy),
            'n_candidates_evaluated': int(n_candidates_evaluated)
        })


# Global instances
_default_metrics_collector = None
_default_telemetry_logger = None


def get_metrics_collector() -> MetricsCollector:
    """Get the default metrics collector instance."""
    global _default_metrics_collector
    if _default_metrics_collector is None:
        _default_metrics_collector = MetricsCollector()
    return _default_metrics_collector


def get_telemetry_logger() -> TelemetryLogger:
    """Get the default telemetry logger instance."""
    global _default_telemetry_logger
    if _default_telemetry_logger is None:
        _default_telemetry_logger = TelemetryLogger()
    return _default_telemetry_logger