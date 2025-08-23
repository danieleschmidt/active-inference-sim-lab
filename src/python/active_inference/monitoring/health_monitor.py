"""
Comprehensive health monitoring for Active Inference agents and systems.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
import json


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetric:
    """Individual health metric tracking."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Determine status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class HealthMonitor:
    """
    Advanced health monitoring system for Active Inference components.
    
    Provides real-time health tracking, alerting, and automated recovery.
    """
    
    def __init__(self,
                 check_interval: float = 5.0,
                 history_length: int = 1000,
                 enable_alerts: bool = True,
                 enable_auto_recovery: bool = True):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            history_length: Number of historical records to keep
            enable_alerts: Enable alert notifications
            enable_auto_recovery: Enable automatic recovery attempts
        """
        self.check_interval = check_interval
        self.history_length = history_length
        self.enable_alerts = enable_alerts
        self.enable_auto_recovery = enable_auto_recovery
        
        # Monitoring state
        self._monitored_components: Dict[str, Any] = {}
        self._metrics: Dict[str, HealthMetric] = {}
        self._history: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._recovery_actions: Dict[str, Callable] = {}
        
        # Threading
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"HealthMonitor")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # System metrics
        self._system_start_time = time.time()
        self._last_check_time = 0.0
        
        self.logger.info("HealthMonitor initialized")
    
    def register_component(self,
                          component_id: str,
                          component: Any,
                          health_check_fn: Optional[Callable] = None) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component: The component object to monitor
            health_check_fn: Optional custom health check function
        """
        with self._lock:
            self._monitored_components[component_id] = {
                'component': component,
                'health_check_fn': health_check_fn,
                'last_check': 0.0,
                'status': HealthStatus.HEALTHY,
                'error_count': 0,
                'last_error': None
            }
        
        self.logger.info(f"Registered component for monitoring: {component_id}")
    
    def add_metric(self, metric: HealthMetric) -> None:
        """Add or update a health metric."""
        with self._lock:
            self._metrics[metric.name] = metric
        
        # Check for immediate alerts
        if self.enable_alerts and metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            self._trigger_alert(metric)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self._lock:
            component_statuses = {}
            worst_status = HealthStatus.HEALTHY
            
            for comp_id, comp_data in self._monitored_components.items():
                status = comp_data['status']
                component_statuses[comp_id] = {
                    'status': status.value,
                    'error_count': comp_data['error_count'],
                    'last_check': comp_data['last_check'],
                    'last_error': comp_data['last_error']
                }
                
                # Track worst status
                if status == HealthStatus.CRITICAL:
                    worst_status = HealthStatus.CRITICAL
                elif status == HealthStatus.DEGRADED and worst_status != HealthStatus.CRITICAL:
                    worst_status = HealthStatus.DEGRADED
                elif status == HealthStatus.WARNING and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.WARNING
            
            # Aggregate metrics
            metric_statuses = {}
            for metric_name, metric in self._metrics.items():
                metric_statuses[metric_name] = {
                    'value': metric.value,
                    'status': metric.status.value,
                    'unit': metric.unit,
                    'description': metric.description,
                    'timestamp': metric.timestamp
                }
            
            return {
                'overall_status': worst_status.value,
                'uptime_seconds': time.time() - self._system_start_time,
                'last_check': self._last_check_time,
                'components': component_statuses,
                'metrics': metric_statuses,
                'active_alerts': len(self._alerts),
                'monitoring_enabled': self._monitoring_thread is not None and self._monitoring_thread.is_alive()
            }
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join(timeout=10.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while not self._stop_event.is_set():
            try:
                self._perform_health_checks()
                self._last_check_time = time.time()
                
                # Record system snapshot
                self._record_health_snapshot()
                
                # Sleep until next check
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(1.0)  # Brief pause before retrying
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all registered components."""
        current_time = time.time()
        
        with self._lock:
            for comp_id, comp_data in self._monitored_components.items():
                try:
                    component = comp_data['component']
                    health_check_fn = comp_data['health_check_fn']
                    
                    # Perform health check
                    if health_check_fn:
                        # Use custom health check
                        is_healthy = health_check_fn(component)
                    else:
                        # Use default health check
                        is_healthy = self._default_health_check(component)
                    
                    # Update status
                    if is_healthy:
                        comp_data['status'] = HealthStatus.HEALTHY
                    else:
                        # Escalate based on error count
                        error_count = comp_data['error_count']
                        if error_count > 10:
                            comp_data['status'] = HealthStatus.CRITICAL
                        elif error_count > 5:
                            comp_data['status'] = HealthStatus.DEGRADED
                        else:
                            comp_data['status'] = HealthStatus.WARNING
                    
                    comp_data['last_check'] = current_time
                    
                except Exception as e:
                    comp_data['error_count'] += 1
                    comp_data['last_error'] = str(e)
                    comp_data['status'] = HealthStatus.CRITICAL
                    
                    self.logger.error(f"Health check failed for {comp_id}: {e}")
                    
                    # Trigger recovery if enabled
                    if self.enable_auto_recovery and comp_id in self._recovery_actions:
                        self._attempt_recovery(comp_id)
    
    def _default_health_check(self, component: Any) -> bool:
        """Default health check implementation."""
        try:
            # Check if component has get_health_status method
            if hasattr(component, 'get_health_status'):
                status = component.get_health_status()
                return status.get('health_status', 'healthy') == 'healthy'
            
            # Check if component has basic status indicators
            if hasattr(component, '_is_initialized'):
                return getattr(component, '_is_initialized', False)
            
            # Check if component has error indicators
            if hasattr(component, '_error_count'):
                return getattr(component, '_error_count', 0) < 5
            
            # Default: assume healthy if no indicators
            return True
            
        except Exception:
            return False
    
    def _trigger_alert(self, metric: HealthMetric) -> None:
        """Trigger alert for unhealthy metric."""
        alert = {
            'timestamp': time.time(),
            'type': 'metric_alert',
            'severity': metric.status.value,
            'message': f"Metric '{metric.name}' is {metric.status.value}: {metric.value} {metric.unit}",
            'details': {
                'metric_name': metric.name,
                'current_value': metric.value,
                'warning_threshold': metric.threshold_warning,
                'critical_threshold': metric.threshold_critical,
                'unit': metric.unit,
                'description': metric.description
            }
        }
        
        with self._lock:
            self._alerts.append(alert)
            
            # Keep only recent alerts
            if len(self._alerts) > 100:
                self._alerts = self._alerts[-100:]
        
        self.logger.warning(f"ALERT: {alert['message']}")
    
    def _attempt_recovery(self, component_id: str) -> None:
        """Attempt automatic recovery for a component."""
        try:
            recovery_fn = self._recovery_actions[component_id]
            self.logger.info(f"Attempting recovery for component: {component_id}")
            
            success = recovery_fn()
            
            if success:
                self.logger.info(f"Recovery successful for: {component_id}")
                with self._lock:
                    self._monitored_components[component_id]['error_count'] = 0
                    self._monitored_components[component_id]['status'] = HealthStatus.HEALTHY
            else:
                self.logger.error(f"Recovery failed for: {component_id}")
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed for {component_id}: {e}")
    
    def _record_health_snapshot(self) -> None:
        """Record current health status to history."""
        snapshot = {
            'timestamp': time.time(),
            'overall_status': self.get_system_health()['overall_status'],
            'component_count': len(self._monitored_components),
            'metric_count': len(self._metrics),
            'alert_count': len(self._alerts)
        }
        
        with self._lock:
            self._history.append(snapshot)
            
            # Trim history
            if len(self._history) > self.history_length:
                self._history = self._history[-self.history_length:]
    
    def register_recovery_action(self, component_id: str, recovery_fn: Callable) -> None:
        """Register automatic recovery action for a component."""
        self._recovery_actions[component_id] = recovery_fn
        self.logger.info(f"Registered recovery action for: {component_id}")
    
    def get_alerts(self, since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by timestamp."""
        with self._lock:
            if since_timestamp is None:
                return self._alerts.copy()
            else:
                return [alert for alert in self._alerts if alert['timestamp'] > since_timestamp]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()
        self.logger.info("All alerts cleared")
    
    def get_health_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get health history for the specified time period."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            return [
                snapshot for snapshot in self._history 
                if snapshot['timestamp'] > cutoff_time
            ]
    
    def export_health_report(self, filepath: str) -> None:
        """Export comprehensive health report to file."""
        report = {
            'generation_time': time.time(),
            'system_health': self.get_system_health(),
            'recent_alerts': self.get_alerts(),
            'health_history': self.get_health_history(minutes=1440),  # 24 hours
            'monitoring_config': {
                'check_interval': self.check_interval,
                'history_length': self.history_length,
                'alerts_enabled': self.enable_alerts,
                'auto_recovery_enabled': self.enable_auto_recovery
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Health report exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export health report: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation of health monitor."""
        health = self.get_system_health()
        return (f"HealthMonitor(status={health['overall_status']}, "
                f"components={len(self._monitored_components)}, "
                f"alerts={len(self._alerts)})")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()