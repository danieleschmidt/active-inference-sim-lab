"""
Graceful Degradation System for Active Inference Agents
Generation 2: MAKE IT ROBUST - Fault-Tolerant Operations
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of system degradation."""
    FULL = "full"                    # All features operational
    REDUCED = "reduced"              # Some features disabled
    MINIMAL = "minimal"              # Only core features
    EMERGENCY = "emergency"          # Bare minimum functionality
    OFFLINE = "offline"              # System offline


@dataclass
class DegradationRule:
    """Rule for system degradation based on conditions."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    target_level: DegradationLevel
    priority: int = 1
    description: str = ""
    recovery_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def __post_init__(self):
        if self.recovery_condition is None:
            # Default recovery: opposite of degradation condition
            self.recovery_condition = lambda metrics: not self.condition(metrics)


@dataclass
class FeatureConfig:
    """Configuration for a system feature."""
    name: str
    required_level: DegradationLevel = DegradationLevel.FULL
    fallback_impl: Optional[Callable] = None
    resource_weight: float = 1.0
    critical: bool = False
    dependencies: List[str] = field(default_factory=list)


class GracefulDegradationManager:
    """
    Manages graceful degradation of system functionality based on
    resource constraints, errors, and performance metrics.
    """
    
    def __init__(self, 
                 check_interval: float = 5.0,
                 metric_window: float = 300.0,
                 enable_auto_degradation: bool = True):
        """
        Initialize graceful degradation manager.
        
        Args:
            check_interval: Seconds between degradation checks
            metric_window: Window for metric aggregation (seconds)
            enable_auto_degradation: Enable automatic degradation
        """
        self.check_interval = check_interval
        self.metric_window = metric_window
        self.enable_auto_degradation = enable_auto_degradation
        
        # System state
        self.current_level = DegradationLevel.FULL
        self.target_level = DegradationLevel.FULL
        self._degradation_history = deque(maxlen=1000)
        
        # Features and rules
        self._features: Dict[str, FeatureConfig] = {}
        self._degradation_rules: List[DegradationRule] = []
        self._active_features: Dict[str, bool] = {}
        
        # Metrics tracking
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._metric_aggregates: Dict[str, Dict[str, float]] = {}
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Statistics
        self._degradation_count = 0
        self._recovery_count = 0
        self._last_degradation_time = 0.0
        self._last_recovery_time = 0.0
        
        # Built-in degradation rules
        self._setup_default_rules()
        
        logger.info("Graceful degradation manager initialized")
    
    def _setup_default_rules(self):
        """Setup default degradation rules."""
        # CPU usage rule
        self.add_degradation_rule(DegradationRule(
            name="high_cpu_usage",
            condition=lambda m: m.get('cpu_usage', 0) > 0.85,
            target_level=DegradationLevel.REDUCED,
            priority=2,
            description="High CPU usage detected"
        ))
        
        # Memory usage rule
        self.add_degradation_rule(DegradationRule(
            name="high_memory_usage",
            condition=lambda m: m.get('memory_usage', 0) > 0.90,
            target_level=DegradationLevel.MINIMAL,
            priority=3,
            description="High memory usage detected"
        ))
        
        # Error rate rule
        self.add_degradation_rule(DegradationRule(
            name="high_error_rate",
            condition=lambda m: m.get('error_rate', 0) > 0.10,
            target_level=DegradationLevel.REDUCED,
            priority=2,
            description="High error rate detected"
        ))
        
        # Response time rule
        self.add_degradation_rule(DegradationRule(
            name="slow_response_time",
            condition=lambda m: m.get('avg_response_time', 0) > 5.0,
            target_level=DegradationLevel.REDUCED,
            priority=1,
            description="Slow response times detected"
        ))
        
        # Critical failure rule
        self.add_degradation_rule(DegradationRule(
            name="critical_failure",
            condition=lambda m: m.get('critical_errors', 0) > 0,
            target_level=DegradationLevel.EMERGENCY,
            priority=10,
            description="Critical system failure detected"
        ))
    
    def register_feature(self, feature: FeatureConfig) -> None:
        """Register a system feature for degradation management."""
        with self._lock:
            self._features[feature.name] = feature
            self._active_features[feature.name] = True
        
        logger.info(f"Feature registered: {feature.name} (level: {feature.required_level.value})")
    
    def add_degradation_rule(self, rule: DegradationRule) -> None:
        """Add a degradation rule."""
        with self._lock:
            self._degradation_rules.append(rule)
            self._degradation_rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Degradation rule added: {rule.name} (priority: {rule.priority})")
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update system metrics for degradation decisions."""
        current_time = time.time()
        
        with self._lock:
            for metric_name, value in metrics.items():
                self._metrics[metric_name].append((current_time, value))
            
            # Update aggregates
            self._update_metric_aggregates()
        
        # Check degradation if auto-enabled
        if self.enable_auto_degradation:
            self._check_degradation_conditions()
    
    def _update_metric_aggregates(self):
        """Update metric aggregates for rule evaluation."""
        current_time = time.time()
        window_start = current_time - self.metric_window
        
        self._metric_aggregates.clear()
        
        for metric_name, values in self._metrics.items():
            # Filter to window
            windowed_values = [v for t, v in values if t >= window_start]
            
            if windowed_values:
                self._metric_aggregates[metric_name] = {
                    'avg': np.mean(windowed_values),
                    'max': np.max(windowed_values),
                    'min': np.min(windowed_values),
                    'std': np.std(windowed_values),
                    'current': windowed_values[-1],
                    'count': len(windowed_values)
                }
            else:
                self._metric_aggregates[metric_name] = {
                    'avg': 0, 'max': 0, 'min': 0, 'std': 0, 'current': 0, 'count': 0
                }
    
    def _check_degradation_conditions(self):
        """Check all degradation rules and apply if necessary."""
        with self._lock:
            if not self._degradation_rules:
                return
            
            # Flatten metric aggregates for rule evaluation
            flat_metrics = {}
            for metric_name, aggregates in self._metric_aggregates.items():
                for agg_type, value in aggregates.items():
                    flat_metrics[f"{metric_name}"] = aggregates['avg']  # Use average by default
                    flat_metrics[f"{metric_name}_{agg_type}"] = value
            
            # Check degradation rules (highest priority first)
            triggered_rule = None
            for rule in self._degradation_rules:
                try:
                    if rule.condition(flat_metrics):
                        triggered_rule = rule
                        break
                except Exception as e:
                    logger.error(f"Error evaluating degradation rule '{rule.name}': {e}")
            
            # Apply degradation if rule triggered
            if triggered_rule:
                if self.current_level.value != triggered_rule.target_level.value:
                    self._apply_degradation(triggered_rule.target_level, triggered_rule.name)
            else:
                # Check for recovery
                if self.current_level != DegradationLevel.FULL:
                    self._check_recovery_conditions(flat_metrics)
    
    def _check_recovery_conditions(self, metrics: Dict[str, float]):
        """Check if system can recover to a higher level."""
        # Find the least restrictive level we can recover to
        possible_level = DegradationLevel.FULL
        
        for rule in self._degradation_rules:
            try:
                if rule.recovery_condition and not rule.recovery_condition(metrics):
                    # This rule still applies, can't go above its target level
                    if self._level_more_restrictive(rule.target_level, possible_level):
                        possible_level = rule.target_level
            except Exception as e:
                logger.error(f"Error evaluating recovery condition for rule '{rule.name}': {e}")
        
        # Apply recovery if possible
        if self._level_less_restrictive(possible_level, self.current_level):
            self._apply_recovery(possible_level)
    
    def _level_more_restrictive(self, level1: DegradationLevel, level2: DegradationLevel) -> bool:
        """Check if level1 is more restrictive than level2."""
        order = [DegradationLevel.FULL, DegradationLevel.REDUCED, 
                DegradationLevel.MINIMAL, DegradationLevel.EMERGENCY, 
                DegradationLevel.OFFLINE]
        return order.index(level1) > order.index(level2)
    
    def _level_less_restrictive(self, level1: DegradationLevel, level2: DegradationLevel) -> bool:
        """Check if level1 is less restrictive than level2."""
        order = [DegradationLevel.FULL, DegradationLevel.REDUCED, 
                DegradationLevel.MINIMAL, DegradationLevel.EMERGENCY, 
                DegradationLevel.OFFLINE]
        return order.index(level1) < order.index(level2)
    
    def _apply_degradation(self, target_level: DegradationLevel, reason: str):
        """Apply system degradation to target level."""
        old_level = self.current_level
        self.current_level = target_level
        self._degradation_count += 1
        self._last_degradation_time = time.time()
        
        # Update feature availability
        self._update_feature_availability()
        
        # Record in history
        self._record_level_change(old_level, target_level, reason, "degradation")
        
        logger.warning(f"System degraded from {old_level.value} to {target_level.value}: {reason}")
    
    def _apply_recovery(self, target_level: DegradationLevel):
        """Apply system recovery to target level."""
        old_level = self.current_level
        self.current_level = target_level
        self._recovery_count += 1
        self._last_recovery_time = time.time()
        
        # Update feature availability
        self._update_feature_availability()
        
        # Record in history
        self._record_level_change(old_level, target_level, "Conditions improved", "recovery")
        
        logger.info(f"System recovered from {old_level.value} to {target_level.value}")
    
    def _update_feature_availability(self):
        """Update which features are available at current degradation level."""
        for feature_name, feature in self._features.items():
            # Feature is available if current level allows it
            feature_available = self._level_less_restrictive(self.current_level, feature.required_level) or \
                              self.current_level == feature.required_level
            
            self._active_features[feature_name] = feature_available or feature.critical
    
    def _record_level_change(self, old_level: DegradationLevel, new_level: DegradationLevel, 
                           reason: str, change_type: str):
        """Record level change in history."""
        record = {
            'timestamp': time.time(),
            'old_level': old_level.value,
            'new_level': new_level.value,
            'reason': reason,
            'change_type': change_type,
            'active_features': self._active_features.copy()
        }
        
        self._degradation_history.append(record)
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is currently available."""
        with self._lock:
            return self._active_features.get(feature_name, False)
    
    def get_feature_implementation(self, feature_name: str) -> Optional[Callable]:
        """Get the current implementation for a feature (main or fallback)."""
        with self._lock:
            if feature_name not in self._features:
                return None
            
            feature = self._features[feature_name]
            
            if self.is_feature_available(feature_name):
                return None  # Use main implementation
            elif feature.fallback_impl:
                return feature.fallback_impl
            else:
                return None  # Feature unavailable
    
    def execute_with_degradation(self, feature_name: str, main_impl: Callable, 
                               *args, **kwargs) -> Any:
        """
        Execute function with graceful degradation support.
        
        Args:
            feature_name: Name of the feature
            main_impl: Main implementation function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from main or fallback implementation
            
        Raises:
            Exception: If feature unavailable and no fallback
        """
        if self.is_feature_available(feature_name):
            # Use main implementation
            try:
                return main_impl(*args, **kwargs)
            except Exception as e:
                # Main implementation failed, try fallback
                logger.warning(f"Main implementation failed for {feature_name}: {e}")
                fallback = self.get_feature_implementation(feature_name)
                if fallback:
                    return fallback(*args, **kwargs)
                else:
                    raise
        else:
            # Feature not available, try fallback
            fallback = self.get_feature_implementation(feature_name)
            if fallback:
                return fallback(*args, **kwargs)
            else:
                raise RuntimeError(f"Feature '{feature_name}' unavailable at degradation level {self.current_level.value}")
    
    def force_degradation_level(self, level: DegradationLevel, reason: str = "Manual override"):
        """Manually force system to specific degradation level."""
        with self._lock:
            old_level = self.current_level
            self.current_level = level
            self._update_feature_availability()
            self._record_level_change(old_level, level, reason, "manual")
        
        logger.warning(f"System degradation manually forced to {level.value}: {reason}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and degradation information."""
        with self._lock:
            active_feature_count = sum(1 for active in self._active_features.values() if active)
            total_feature_count = len(self._features)
            
            return {
                'current_level': self.current_level.value,
                'degradation_count': self._degradation_count,
                'recovery_count': self._recovery_count,
                'last_degradation_time': self._last_degradation_time,
                'last_recovery_time': self._last_recovery_time,
                'active_features': active_feature_count,
                'total_features': total_feature_count,
                'feature_availability_rate': active_feature_count / max(1, total_feature_count),
                'active_feature_list': [name for name, active in self._active_features.items() if active],
                'disabled_feature_list': [name for name, active in self._active_features.items() if not active],
                'auto_degradation_enabled': self.enable_auto_degradation,
                'last_metrics_update': max((max(times) for times in self._metrics.values()), default=0) if self._metrics else 0
            }
    
    def get_degradation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get degradation history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [record for record in self._degradation_history 
                   if record['timestamp'] > cutoff_time]
    
    def start_monitoring(self) -> None:
        """Start continuous degradation monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Degradation monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="DegradationMonitor"
        )
        self._monitor_thread.start()
        logger.info("Degradation monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous degradation monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join(timeout=10.0)
        
        logger.info("Degradation monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                if self.enable_auto_degradation:
                    self._check_degradation_conditions()
                
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in degradation monitoring loop: {e}")
                self._stop_event.wait(1.0)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"GracefulDegradationManager(level={self.current_level.value}, "
                f"features={len(self._features)}, rules={len(self._degradation_rules)})")


def degradation_protected(feature_name: str, fallback: Optional[Callable] = None):
    """
    Decorator for functions that should be protected by graceful degradation.
    
    Args:
        feature_name: Name of the feature this function implements
        fallback: Optional fallback implementation
        
    Returns:
        Decorated function with degradation protection
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Assume self has a degradation_manager attribute
            if hasattr(self, 'degradation_manager') and self.degradation_manager:
                return self.degradation_manager.execute_with_degradation(
                    feature_name, func, self, *args, **kwargs
                )
            else:
                # No degradation manager, execute normally
                return func(self, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator