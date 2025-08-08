"""
Health check and monitoring system for Active Inference components.

This module provides comprehensive health monitoring, diagnostics, and
system status reporting for all framework components.
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback

from .logging_config import get_logger, LogCategory
from .validation import ValidationError


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component types for health monitoring."""
    AGENT = "agent"
    INFERENCE = "inference"
    PLANNING = "planning"
    ENVIRONMENT = "environment"
    MODEL = "model"
    SYSTEM = "system"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Union[float, int, str, bool]
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'status': self.status.value,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'unit': self.unit,
            'description': self.description,
            'timestamp': self.timestamp
        }


@dataclass
class ComponentHealth:
    """Health status for a component."""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0
    last_error: Optional[str] = None
    uptime: float = 0.0
    
    def add_metric(self, metric: HealthMetric) -> None:
        """Add or update a health metric."""
        self.metrics[metric.name] = metric
        self.last_updated = time.time()
        
        # Update overall status based on metric status
        if metric.status == HealthStatus.CRITICAL:
            self.status = HealthStatus.CRITICAL
        elif metric.status == HealthStatus.DEGRADED and self.status != HealthStatus.CRITICAL:
            self.status = HealthStatus.DEGRADED
        elif metric.status == HealthStatus.WARNING and self.status in [HealthStatus.HEALTHY, HealthStatus.UNKNOWN]:
            self.status = HealthStatus.WARNING
    
    def record_error(self, error: str) -> None:
        """Record an error for this component."""
        self.error_count += 1
        self.last_error = error
        self.last_updated = time.time()
        
        # Determine status based on error frequency
        if self.error_count >= 10:
            self.status = HealthStatus.CRITICAL
        elif self.error_count >= 5:
            self.status = HealthStatus.DEGRADED
        elif self.error_count >= 2:
            self.status = HealthStatus.WARNING
    
    def reset_errors(self) -> None:
        """Reset error count (for recovery)."""
        self.error_count = 0
        self.last_error = None
        if self.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED, HealthStatus.WARNING]:
            # Re-evaluate status based on metrics
            worst_status = HealthStatus.HEALTHY
            for metric in self.metrics.values():
                if metric.status.value == 'critical':
                    worst_status = HealthStatus.CRITICAL
                    break
                elif metric.status.value == 'degraded':
                    worst_status = HealthStatus.DEGRADED
                elif metric.status.value == 'warning' and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.WARNING
            self.status = worst_status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'metrics': {name: metric.to_dict() for name, metric in self.metrics.items()},
            'last_updated': self.last_updated,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'uptime': self.uptime
        }


class SystemHealthMonitor:
    """Monitor system-level health metrics."""
    
    def __init__(self):
        self.logger = get_logger("system")
        self.start_time = time.time()
    
    def check_memory_usage(self) -> HealthMetric:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent >= 90:
                status = HealthStatus.CRITICAL
            elif usage_percent >= 80:
                status = HealthStatus.DEGRADED
            elif usage_percent >= 70:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="memory_usage",
                value=usage_percent,
                status=status,
                threshold_warning=70.0,
                threshold_critical=90.0,
                unit="%",
                description="System memory usage percentage"
            )
        except Exception as e:
            self.logger.warning(f"Failed to check memory usage: {e}", LogCategory.SYSTEM)
            return HealthMetric(
                name="memory_usage",
                value="unknown",
                status=HealthStatus.UNKNOWN,
                description=f"Memory check failed: {e}"
            )
    
    def check_cpu_usage(self) -> HealthMetric:
        """Check system CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent >= 95:
                status = HealthStatus.CRITICAL
            elif cpu_percent >= 85:
                status = HealthStatus.DEGRADED
            elif cpu_percent >= 75:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=status,
                threshold_warning=75.0,
                threshold_critical=95.0,
                unit="%",
                description="System CPU usage percentage"
            )
        except Exception as e:
            self.logger.warning(f"Failed to check CPU usage: {e}", LogCategory.SYSTEM)
            return HealthMetric(
                name="cpu_usage",
                value="unknown",
                status=HealthStatus.UNKNOWN,
                description=f"CPU check failed: {e}"
            )
    
    def check_disk_usage(self) -> HealthMetric:
        """Check system disk usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent >= 95:
                status = HealthStatus.CRITICAL
            elif usage_percent >= 85:
                status = HealthStatus.DEGRADED
            elif usage_percent >= 75:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="disk_usage",
                value=usage_percent,
                status=status,
                threshold_warning=75.0,
                threshold_critical=95.0,
                unit="%",
                description="System disk usage percentage"
            )
        except Exception as e:
            self.logger.warning(f"Failed to check disk usage: {e}", LogCategory.SYSTEM)
            return HealthMetric(
                name="disk_usage",
                value="unknown",
                status=HealthStatus.UNKNOWN,
                description=f"Disk check failed: {e}"
            )
    
    def check_uptime(self) -> HealthMetric:
        """Check system uptime."""
        try:
            uptime_seconds = time.time() - self.start_time
            uptime_hours = uptime_seconds / 3600
            
            # Generally healthy if running, but flag if too long (potential memory leaks)
            if uptime_hours >= 168:  # 1 week
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="uptime",
                value=uptime_hours,
                status=status,
                threshold_warning=168.0,
                unit="hours",
                description="System uptime in hours"
            )
        except Exception as e:
            self.logger.warning(f"Failed to check uptime: {e}", LogCategory.SYSTEM)
            return HealthMetric(
                name="uptime",
                value="unknown",
                status=HealthStatus.UNKNOWN,
                description=f"Uptime check failed: {e}"
            )
    
    def get_all_metrics(self) -> List[HealthMetric]:
        """Get all system health metrics."""
        return [
            self.check_memory_usage(),
            self.check_cpu_usage(),
            self.check_disk_usage(),
            self.check_uptime()
        ]


class HealthCheckRegistry:
    """
    Registry for health checks across all system components.
    
    Provides centralized health monitoring, alerting, and reporting.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize health check registry.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.logger = get_logger("system")
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.system_monitor = SystemHealthMonitor()
        self.custom_checks: Dict[str, Callable[[], HealthMetric]] = {}
        
        # Threading for periodic checks
        self._stop_event = threading.Event()
        self._check_thread = None
        
        # Alerting
        self.alert_handlers: List[Callable[[ComponentHealth], None]] = []
        self.alert_thresholds = {
            HealthStatus.WARNING: True,
            HealthStatus.DEGRADED: True,
            HealthStatus.CRITICAL: True
        }
        
        self.logger.info(
            "Health check registry initialized",
            LogCategory.SYSTEM,
            {'check_interval': check_interval}
        )
    
    def register_component(self, component_id: str, component_type: ComponentType) -> ComponentHealth:
        """
        Register a component for health monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component
            
        Returns:
            ComponentHealth object for the component
        """
        if component_id in self.components:
            self.logger.warning(
                f"Component {component_id} already registered, updating",
                LogCategory.SYSTEM
            )
        
        component = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            status=HealthStatus.HEALTHY
        )
        
        self.components[component_id] = component
        
        self.logger.info(
            f"Registered component for health monitoring",
            LogCategory.SYSTEM,
            {'component_id': component_id, 'component_type': component_type.value}
        )
        
        return component
    
    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from health monitoring."""
        if component_id in self.components:
            del self.components[component_id]
            self.logger.info(
                f"Unregistered component {component_id}",
                LogCategory.SYSTEM
            )
    
    def update_component_health(self, component_id: str, metrics: List[HealthMetric]) -> None:
        """
        Update health metrics for a component.
        
        Args:
            component_id: Component identifier
            metrics: List of health metrics
        """
        if component_id not in self.components:
            self.logger.warning(
                f"Component {component_id} not registered, auto-registering",
                LogCategory.SYSTEM
            )
            self.register_component(component_id, ComponentType.SYSTEM)
        
        component = self.components[component_id]
        previous_status = component.status
        
        # Update metrics
        for metric in metrics:
            component.add_metric(metric)
        
        # Check if status changed and needs alerting
        if component.status != previous_status and self.alert_thresholds.get(component.status, False):
            self._trigger_alerts(component)
    
    def record_component_error(self, component_id: str, error: str) -> None:
        """
        Record an error for a component.
        
        Args:
            component_id: Component identifier
            error: Error description
        """
        if component_id not in self.components:
            self.register_component(component_id, ComponentType.SYSTEM)
        
        component = self.components[component_id]
        previous_status = component.status
        
        component.record_error(error)
        
        self.logger.warning(
            f"Error recorded for component {component_id}",
            LogCategory.SYSTEM,
            {
                'component_id': component_id,
                'error': error,
                'error_count': component.error_count,
                'status': component.status.value
            }
        )
        
        # Check if status changed and needs alerting
        if component.status != previous_status and self.alert_thresholds.get(component.status, False):
            self._trigger_alerts(component)
    
    def add_custom_check(self, name: str, check_func: Callable[[], HealthMetric]) -> None:
        """
        Add a custom health check function.
        
        Args:
            name: Name of the check
            check_func: Function that returns a HealthMetric
        """
        self.custom_checks[name] = check_func
        self.logger.info(f"Added custom health check: {name}", LogCategory.SYSTEM)
    
    def add_alert_handler(self, handler: Callable[[ComponentHealth], None]) -> None:
        """
        Add an alert handler function.
        
        Args:
            handler: Function to call when alerts are triggered
        """
        self.alert_handlers.append(handler)
        self.logger.info("Added alert handler", LogCategory.SYSTEM)
    
    def _trigger_alerts(self, component: ComponentHealth) -> None:
        """Trigger alerts for a component."""
        for handler in self.alert_handlers:
            try:
                handler(component)
            except Exception as e:
                self.logger.error(
                    f"Alert handler failed: {e}",
                    LogCategory.SYSTEM,
                    error=e
                )
    
    def start_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._check_thread is not None and self._check_thread.is_alive():
            self.logger.warning("Health monitoring already running", LogCategory.SYSTEM)
            return
        
        self._stop_event.clear()
        self._check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._check_thread.start()
        
        self.logger.info("Started health monitoring", LogCategory.SYSTEM)
    
    def stop_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if self._check_thread is not None:
            self._stop_event.set()
            self._check_thread.join(timeout=5)
            self._check_thread = None
        
        self.logger.info("Stopped health monitoring", LogCategory.SYSTEM)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._perform_health_checks()
            except Exception as e:
                self.logger.error(
                    f"Error in health monitoring loop: {e}",
                    LogCategory.SYSTEM,
                    error=e
                )
            
            # Wait for next check interval
            self._stop_event.wait(self.check_interval)
    
    def _perform_health_checks(self) -> None:
        """Perform all health checks."""
        # System health checks
        system_metrics = self.system_monitor.get_all_metrics()
        self.update_component_health("system", system_metrics)
        
        # Custom health checks
        for name, check_func in self.custom_checks.items():
            try:
                metric = check_func()
                self.update_component_health(f"custom_{name}", [metric])
            except Exception as e:
                self.logger.warning(
                    f"Custom health check {name} failed: {e}",
                    LogCategory.SYSTEM
                )
                error_metric = HealthMetric(
                    name=name,
                    value="error",
                    status=HealthStatus.CRITICAL,
                    description=f"Check failed: {e}"
                )
                self.update_component_health(f"custom_{name}", [error_metric])
        
        # Update component uptimes
        current_time = time.time()
        for component in self.components.values():
            if hasattr(component, 'start_time'):
                component.uptime = current_time - component.start_time
    
    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with overall health information
        """
        if not self.components:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'No components registered',
                'component_count': 0,
                'components': {}
            }
        
        # Determine worst status across all components
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.CRITICAL: 3,
            HealthStatus.UNKNOWN: 4
        }
        
        worst_status = HealthStatus.HEALTHY
        status_counts = {status: 0 for status in HealthStatus}
        
        for component in self.components.values():
            status_counts[component.status] += 1
            if status_priority[component.status] > status_priority[worst_status]:
                worst_status = component.status
        
        # Generate status message
        if worst_status == HealthStatus.HEALTHY:
            message = "All components healthy"
        elif worst_status == HealthStatus.WARNING:
            message = f"{status_counts[HealthStatus.WARNING]} components in warning state"
        elif worst_status == HealthStatus.DEGRADED:
            message = f"{status_counts[HealthStatus.DEGRADED]} components degraded"
        elif worst_status == HealthStatus.CRITICAL:
            message = f"{status_counts[HealthStatus.CRITICAL]} components critical"
        else:
            message = "Unknown component status"
        
        return {
            'status': worst_status.value,
            'message': message,
            'component_count': len(self.components),
            'status_counts': {status.value: count for status, count in status_counts.items()},
            'components': {
                comp_id: comp.to_dict() 
                for comp_id, comp in self.components.items()
            },
            'last_check': time.time()
        }
    
    def get_component_health(self, component_id: str) -> Optional[Dict[str, Any]]:
        """
        Get health status for a specific component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Component health dictionary or None if not found
        """
        component = self.components.get(component_id)
        return component.to_dict() if component else None
    
    def get_unhealthy_components(self) -> Dict[str, ComponentHealth]:
        """Get all components that are not healthy."""
        return {
            comp_id: comp 
            for comp_id, comp in self.components.items() 
            if comp.status != HealthStatus.HEALTHY
        }
    
    def reset_component_errors(self, component_id: str) -> bool:
        """
        Reset error count for a component (manual recovery).
        
        Args:
            component_id: Component identifier
            
        Returns:
            True if component was found and reset, False otherwise
        """
        if component_id in self.components:
            self.components[component_id].reset_errors()
            self.logger.info(
                f"Reset errors for component {component_id}",
                LogCategory.SYSTEM
            )
            return True
        return False
    
    def generate_health_report(self) -> str:
        """Generate a human-readable health report."""
        overall = self.get_overall_health()
        
        report = [
            "=== ACTIVE INFERENCE SYSTEM HEALTH REPORT ===",
            f"Overall Status: {overall['status'].upper()}",
            f"Message: {overall['message']}",
            f"Components Monitored: {overall['component_count']}",
            f"Last Check: {time.ctime(overall['last_check'])}",
            "",
            "Component Status Breakdown:"
        ]
        
        for status, count in overall['status_counts'].items():
            if count > 0:
                report.append(f"  {status.upper()}: {count}")
        
        report.append("\nComponent Details:")
        
        for comp_id, comp_data in overall['components'].items():
            report.append(f"\n{comp_id} ({comp_data['component_type']}):")
            report.append(f"  Status: {comp_data['status']}")
            report.append(f"  Errors: {comp_data['error_count']}")
            report.append(f"  Last Updated: {time.ctime(comp_data['last_updated'])}")
            
            if comp_data['last_error']:
                report.append(f"  Last Error: {comp_data['last_error']}")
            
            if comp_data['metrics']:
                report.append("  Metrics:")
                for metric_name, metric_data in comp_data['metrics'].items():
                    value = metric_data['value']
                    unit = metric_data.get('unit', '')
                    status = metric_data['status']
                    report.append(f"    {metric_name}: {value}{unit} ({status})")
        
        return "\n".join(report)


# Global health check registry
_health_registry: Optional[HealthCheckRegistry] = None


def get_health_registry() -> HealthCheckRegistry:
    """Get global health check registry."""
    global _health_registry
    if _health_registry is None:
        _health_registry = HealthCheckRegistry()
    return _health_registry


def health_check(component_id: Optional[str] = None):
    """
    Decorator for automatic health monitoring of functions.
    
    Args:
        component_id: Optional component ID (uses function name if not provided)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            registry = get_health_registry()
            comp_id = component_id or f"function_{func.__name__}"
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                metric = HealthMetric(
                    name="execution_time",
                    value=execution_time,
                    status=HealthStatus.HEALTHY if execution_time < 1.0 else HealthStatus.WARNING,
                    threshold_warning=1.0,
                    threshold_critical=5.0,
                    unit="seconds",
                    description=f"Execution time for {func.__name__}"
                )
                
                registry.update_component_health(comp_id, [metric])
                
                return result
                
            except Exception as e:
                # Record error
                registry.record_component_error(comp_id, str(e))
                raise
        
        return wrapper
    return decorator


# Export key classes and functions
__all__ = [
    'HealthCheckRegistry',
    'ComponentHealth',
    'HealthMetric',
    'HealthStatus',
    'ComponentType',
    'SystemHealthMonitor',
    'get_health_registry',
    'health_check'
]