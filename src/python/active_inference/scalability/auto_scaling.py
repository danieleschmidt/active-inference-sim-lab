"""
Intelligent Auto-Scaling System for Active Inference Agents
Generation 3: MAKE IT SCALE - Dynamic Resource Management
"""

import time
import logging
import threading
import queue
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import psutil
import concurrent.futures
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ScalingTrigger(Enum):
    """Triggers for scaling operations."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


class ScalingDirection(Enum):
    """Direction of scaling operation."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class ScalingRule:
    """Configuration for auto-scaling rules."""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int = 1
    max_instances: int = 10
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    cooldown_period: float = 300.0  # 5 minutes
    evaluation_window: float = 60.0  # 1 minute
    priority: int = 1
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    rule_name: str
    direction: ScalingDirection
    old_count: int
    new_count: int
    trigger_value: float
    threshold: float
    reason: str
    duration: float = 0.0
    success: bool = True


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    active_instances: int = 0
    queue_depth: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class InstanceManager:
    """Manages lifecycle of agent instances."""
    
    def __init__(self, agent_factory: Callable[[], Any]):
        """
        Initialize instance manager.
        
        Args:
            agent_factory: Function that creates new agent instances
        """
        self.agent_factory = agent_factory
        self.instances: Dict[str, Any] = {}
        self.instance_pool = queue.Queue()
        self.instance_counter = 0
        self.instance_lock = threading.RLock()
        
        # Instance statistics
        self.instance_stats = defaultdict(lambda: {
            'created_at': 0.0,
            'requests_handled': 0,
            'total_processing_time': 0.0,
            'last_used': 0.0,
            'status': 'idle'  # idle, busy, failed, terminated
        })
        
        self.logger = logging.getLogger("InstanceManager")
    
    def create_instance(self) -> str:
        """Create a new agent instance."""
        with self.instance_lock:
            instance_id = f"instance_{self.instance_counter}"
            self.instance_counter += 1
            
            try:
                instance = self.agent_factory()
                self.instances[instance_id] = instance
                self.instance_stats[instance_id]['created_at'] = time.time()
                self.instance_stats[instance_id]['status'] = 'idle'
                
                self.logger.info(f"Created instance: {instance_id}")
                return instance_id
                
            except Exception as e:
                self.logger.error(f"Failed to create instance {instance_id}: {e}")
                raise
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an agent instance."""
        with self.instance_lock:
            if instance_id not in self.instances:
                return False
            
            try:
                instance = self.instances[instance_id]
                
                # Graceful shutdown if possible
                if hasattr(instance, 'shutdown'):
                    instance.shutdown()
                
                del self.instances[instance_id]
                self.instance_stats[instance_id]['status'] = 'terminated'
                
                self.logger.info(f"Terminated instance: {instance_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error terminating instance {instance_id}: {e}")
                return False
    
    def get_instance(self, instance_id: str) -> Optional[Any]:
        """Get instance by ID."""
        return self.instances.get(instance_id)
    
    def get_idle_instances(self) -> List[str]:
        """Get list of idle instance IDs."""
        with self.instance_lock:
            return [
                iid for iid, stats in self.instance_stats.items()
                if stats['status'] == 'idle' and iid in self.instances
            ]
    
    def mark_instance_busy(self, instance_id: str):
        """Mark instance as busy."""
        if instance_id in self.instance_stats:
            self.instance_stats[instance_id]['status'] = 'busy'
            self.instance_stats[instance_id]['last_used'] = time.time()
    
    def mark_instance_idle(self, instance_id: str):
        """Mark instance as idle."""
        if instance_id in self.instance_stats:
            self.instance_stats[instance_id]['status'] = 'idle'
    
    def update_instance_stats(self, instance_id: str, processing_time: float):
        """Update instance processing statistics."""
        if instance_id in self.instance_stats:
            stats = self.instance_stats[instance_id]
            stats['requests_handled'] += 1
            stats['total_processing_time'] += processing_time
    
    def get_instance_count(self) -> int:
        """Get current number of active instances."""
        with self.instance_lock:
            return len(self.instances)
    
    def get_instance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive instance statistics."""
        with self.instance_lock:
            active_instances = len(self.instances)
            idle_instances = len(self.get_idle_instances())
            busy_instances = active_instances - idle_instances
            
            # Calculate aggregate statistics
            total_requests = sum(stats['requests_handled'] for stats in self.instance_stats.values())
            total_processing_time = sum(stats['total_processing_time'] for stats in self.instance_stats.values())
            
            avg_processing_time = total_processing_time / max(1, total_requests)
            
            return {
                'active_instances': active_instances,
                'idle_instances': idle_instances,
                'busy_instances': busy_instances,
                'total_instances_created': self.instance_counter,
                'total_requests_handled': total_requests,
                'avg_processing_time': avg_processing_time,
                'instance_details': dict(self.instance_stats)
            }


class MetricsCollector:
    """Collects and aggregates system metrics for scaling decisions."""
    
    def __init__(self, collection_interval: float = 5.0, history_length: int = 200):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Seconds between metric collections
            history_length: Number of historical metrics to retain
        """
        self.collection_interval = collection_interval
        self.history_length = history_length
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_length)
        self.current_metrics = ResourceMetrics()
        
        # Custom metrics
        self.custom_metric_providers: Dict[str, Callable[[], float]] = {}
        
        # Threading
        self.collection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.metrics_lock = threading.RLock()
        
        # Application-specific counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times = deque(maxlen=100)
        self.last_metrics_time = time.time()
        
        self.logger = logging.getLogger("MetricsCollector")
    
    def start_collection(self):
        """Start continuous metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.logger.warning("Metrics collection already running")
            return
        
        self.stop_event.clear()
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        self.collection_thread.start()
        self.logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_event.set()
            self.collection_thread.join(timeout=10.0)
        
        self.logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while not self.stop_event.is_set():
            try:
                self._collect_system_metrics()
                self._collect_custom_metrics()
                
                with self.metrics_lock:
                    self.metrics_history.append(self.current_metrics)
                
                self.stop_event.wait(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                self.stop_event.wait(1.0)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Calculate application metrics
            current_time = time.time()
            time_delta = current_time - self.last_metrics_time
            
            requests_per_second = self.request_counter / max(0.1, time_delta)
            error_rate = self.error_counter / max(1, self.request_counter) if self.request_counter > 0 else 0.0
            avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0.0
            
            # Reset counters for next interval
            self.request_counter = 0
            self.error_counter = 0
            self.last_metrics_time = current_time
            
            # Update current metrics
            self.current_metrics = ResourceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                queue_depth=0,  # Will be updated by application
                avg_response_time=avg_response_time,
                requests_per_second=requests_per_second,
                error_rate=error_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_custom_metrics(self):
        """Collect custom application metrics."""
        for metric_name, provider_func in self.custom_metric_providers.items():
            try:
                value = provider_func()
                self.current_metrics.custom_metrics[metric_name] = value
            except Exception as e:
                self.logger.error(f"Error collecting custom metric '{metric_name}': {e}")
    
    def register_custom_metric(self, name: str, provider: Callable[[], float]):
        """Register a custom metric provider."""
        self.custom_metric_providers[name] = provider
        self.logger.info(f"Registered custom metric: {name}")
    
    def record_request(self, response_time: float, error: bool = False):
        """Record a request for metrics calculation."""
        self.request_counter += 1
        if error:
            self.error_counter += 1
        
        self.response_times.append(response_time)
    
    def update_queue_depth(self, depth: int):
        """Update current queue depth."""
        self.current_metrics.queue_depth = depth
    
    def update_active_instances(self, count: int):
        """Update active instance count."""
        self.current_metrics.active_instances = count
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        with self.metrics_lock:
            return self.current_metrics
    
    def get_metric_history(self, metric_name: str, window_seconds: float = 300.0) -> List[float]:
        """Get historical values for a specific metric."""
        cutoff_time = time.time() - window_seconds
        
        with self.metrics_lock:
            history = []
            for metrics in self.metrics_history:
                if metrics.timestamp >= cutoff_time:
                    if hasattr(metrics, metric_name):
                        history.append(getattr(metrics, metric_name))
                    elif metric_name in metrics.custom_metrics:
                        history.append(metrics.custom_metrics[metric_name])
        
        return history
    
    def get_average_metric(self, metric_name: str, window_seconds: float = 60.0) -> float:
        """Get average value of a metric over time window."""
        history = self.get_metric_history(metric_name, window_seconds)
        return np.mean(history) if history else 0.0


class AutoScaler:
    """
    Intelligent auto-scaling system for Active Inference agents.
    
    Monitors system metrics and automatically scales agent instances
    up or down based on configurable rules and thresholds.
    """
    
    def __init__(self,
                 instance_manager: InstanceManager,
                 metrics_collector: MetricsCollector,
                 scaling_rules: Optional[List[ScalingRule]] = None,
                 decision_interval: float = 30.0,
                 enable_predictive_scaling: bool = True):
        """
        Initialize auto-scaler.
        
        Args:
            instance_manager: Manages agent instance lifecycle
            metrics_collector: Collects system metrics
            scaling_rules: List of scaling rules to apply
            decision_interval: Seconds between scaling decisions
            enable_predictive_scaling: Enable predictive scaling based on trends
        """
        self.instance_manager = instance_manager
        self.metrics_collector = metrics_collector
        self.decision_interval = decision_interval
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Scaling rules
        self.scaling_rules = scaling_rules or self._default_scaling_rules()
        self._validate_scaling_rules()
        
        # Scaling state
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling_times: Dict[str, float] = {}
        
        # Threading
        self.scaling_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.scaling_lock = threading.RLock()
        
        # Statistics
        self.total_scale_ups = 0
        self.total_scale_downs = 0
        self.total_scaling_decisions = 0
        
        self.logger = logging.getLogger("AutoScaler")
        self.logger.info("Auto-scaler initialized")
    
    def _default_scaling_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            ScalingRule(
                name="cpu_based_scaling",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=75.0,
                scale_down_threshold=25.0,
                min_instances=1,
                max_instances=10,
                priority=2
            ),
            ScalingRule(
                name="memory_based_scaling",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=8,
                priority=3
            ),
            ScalingRule(
                name="response_time_scaling",
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=2.0,  # 2 seconds
                scale_down_threshold=0.5,  # 0.5 seconds
                min_instances=1,
                max_instances=12,
                priority=1
            ),
            ScalingRule(
                name="queue_depth_scaling",
                trigger=ScalingTrigger.QUEUE_DEPTH,
                scale_up_threshold=20.0,
                scale_down_threshold=2.0,
                min_instances=1,
                max_instances=15,
                priority=2
            )
        ]
    
    def _validate_scaling_rules(self):
        """Validate scaling rule configuration."""
        for rule in self.scaling_rules:
            if rule.min_instances < 1:
                raise ValueError(f"Rule '{rule.name}': min_instances must be >= 1")
            
            if rule.max_instances < rule.min_instances:
                raise ValueError(f"Rule '{rule.name}': max_instances must be >= min_instances")
            
            if rule.scale_up_threshold <= rule.scale_down_threshold:
                raise ValueError(f"Rule '{rule.name}': scale_up_threshold must be > scale_down_threshold")
    
    def start_auto_scaling(self):
        """Start automatic scaling decisions."""
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.logger.warning("Auto-scaling already running")
            return
        
        # Start metrics collection if not already running
        self.metrics_collector.start_collection()
        
        self.stop_event.clear()
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True,
            name="AutoScaler"
        )
        self.scaling_thread.start()
        
        self.logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling decisions."""
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.stop_event.set()
            self.scaling_thread.join(timeout=15.0)
        
        self.logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while not self.stop_event.is_set():
            try:
                self._make_scaling_decisions()
                self.total_scaling_decisions += 1
                
                # Predictive scaling
                if self.enable_predictive_scaling:
                    self._predictive_scaling()
                
                self.stop_event.wait(self.decision_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                self.stop_event.wait(5.0)
    
    def _make_scaling_decisions(self):
        """Make scaling decisions based on current metrics."""
        current_metrics = self.metrics_collector.get_current_metrics()
        current_instances = self.instance_manager.get_instance_count()
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.scaling_rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            last_scaling = self.last_scaling_times.get(rule.name, 0)
            if time.time() - last_scaling < rule.cooldown_period:
                continue
            
            # Get metric value for evaluation
            metric_value = self._get_metric_value(current_metrics, rule.trigger)
            if metric_value is None:
                continue
            
            # Use averaged metric over evaluation window
            avg_metric_value = self.metrics_collector.get_average_metric(
                self._trigger_to_metric_name(rule.trigger), rule.evaluation_window
            )
            
            if avg_metric_value == 0.0:
                avg_metric_value = metric_value  # Fallback to current value
            
            # Make scaling decision
            scaling_decision = self._evaluate_scaling_rule(rule, avg_metric_value, current_instances)
            
            if scaling_decision != ScalingDirection.STABLE:
                success = self._execute_scaling_decision(rule, scaling_decision, avg_metric_value, current_instances)
                
                if success:
                    self.last_scaling_times[rule.name] = time.time()
                    # Only apply one rule per cycle to avoid conflicts
                    break
    
    def _get_metric_value(self, metrics: ResourceMetrics, trigger: ScalingTrigger) -> Optional[float]:
        """Extract metric value based on trigger type."""
        metric_map = {
            ScalingTrigger.CPU_UTILIZATION: metrics.cpu_percent,
            ScalingTrigger.MEMORY_UTILIZATION: metrics.memory_percent,
            ScalingTrigger.QUEUE_DEPTH: float(metrics.queue_depth),
            ScalingTrigger.RESPONSE_TIME: metrics.avg_response_time,
            ScalingTrigger.THROUGHPUT: metrics.requests_per_second,
            ScalingTrigger.ERROR_RATE: metrics.error_rate * 100  # Convert to percentage
        }
        
        return metric_map.get(trigger)
    
    def _trigger_to_metric_name(self, trigger: ScalingTrigger) -> str:
        """Convert trigger enum to metric attribute name."""
        name_map = {
            ScalingTrigger.CPU_UTILIZATION: "cpu_percent",
            ScalingTrigger.MEMORY_UTILIZATION: "memory_percent",
            ScalingTrigger.QUEUE_DEPTH: "queue_depth",
            ScalingTrigger.RESPONSE_TIME: "avg_response_time",
            ScalingTrigger.THROUGHPUT: "requests_per_second",
            ScalingTrigger.ERROR_RATE: "error_rate"
        }
        
        return name_map.get(trigger, "cpu_percent")
    
    def _evaluate_scaling_rule(self, rule: ScalingRule, metric_value: float, current_instances: int) -> ScalingDirection:
        """Evaluate if scaling is needed based on rule."""
        # Check for scale up
        if metric_value >= rule.scale_up_threshold and current_instances < rule.max_instances:
            return ScalingDirection.UP
        
        # Check for scale down
        if metric_value <= rule.scale_down_threshold and current_instances > rule.min_instances:
            return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _execute_scaling_decision(self, rule: ScalingRule, direction: ScalingDirection, 
                                trigger_value: float, current_instances: int) -> bool:
        """Execute scaling decision."""
        start_time = time.time()
        
        try:
            if direction == ScalingDirection.UP:
                new_instances = min(current_instances + rule.scale_up_increment, rule.max_instances)
                instances_to_add = new_instances - current_instances
                
                # Create new instances
                created_instances = []
                for _ in range(instances_to_add):
                    try:
                        instance_id = self.instance_manager.create_instance()
                        created_instances.append(instance_id)
                    except Exception as e:
                        self.logger.error(f"Failed to create instance during scale-up: {e}")
                        break
                
                actual_new_count = current_instances + len(created_instances)
                threshold = rule.scale_up_threshold
                self.total_scale_ups += len(created_instances)
                
            elif direction == ScalingDirection.DOWN:
                new_instances = max(current_instances - rule.scale_down_increment, rule.min_instances)
                instances_to_remove = current_instances - new_instances
                
                # Terminate idle instances first
                idle_instances = self.instance_manager.get_idle_instances()
                instances_to_terminate = idle_instances[:instances_to_remove]
                
                terminated_count = 0
                for instance_id in instances_to_terminate:
                    if self.instance_manager.terminate_instance(instance_id):
                        terminated_count += 1
                
                actual_new_count = current_instances - terminated_count
                threshold = rule.scale_down_threshold
                self.total_scale_downs += terminated_count
            
            else:
                return False
            
            # Record scaling event
            duration = time.time() - start_time
            event = ScalingEvent(
                timestamp=start_time,
                rule_name=rule.name,
                direction=direction,
                old_count=current_instances,
                new_count=actual_new_count,
                trigger_value=trigger_value,
                threshold=threshold,
                reason=f"{rule.trigger.value} {direction.value}",
                duration=duration,
                success=True
            )
            
            with self.scaling_lock:
                self.scaling_events.append(event)
                
                # Keep only recent events
                if len(self.scaling_events) > 1000:
                    self.scaling_events = self.scaling_events[-1000:]
            
            # Update metrics collector with new instance count
            self.metrics_collector.update_active_instances(actual_new_count)
            
            self.logger.info(f"Scaling {direction.value}: {current_instances} -> {actual_new_count} instances "
                           f"(rule: {rule.name}, metric: {trigger_value:.2f}, threshold: {threshold:.2f})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
            
            # Record failed scaling event
            event = ScalingEvent(
                timestamp=start_time,
                rule_name=rule.name,
                direction=direction,
                old_count=current_instances,
                new_count=current_instances,
                trigger_value=trigger_value,
                threshold=rule.scale_up_threshold if direction == ScalingDirection.UP else rule.scale_down_threshold,
                reason=f"Failed: {str(e)}",
                duration=time.time() - start_time,
                success=False
            )
            
            with self.scaling_lock:
                self.scaling_events.append(event)
            
            return False
    
    def _predictive_scaling(self):
        """Implement predictive scaling based on metric trends."""
        # Analyze trends in key metrics
        cpu_history = self.metrics_collector.get_metric_history("cpu_percent", 600.0)  # 10 minutes
        memory_history = self.metrics_collector.get_metric_history("memory_percent", 600.0)
        response_time_history = self.metrics_collector.get_metric_history("avg_response_time", 300.0)  # 5 minutes
        
        if len(cpu_history) < 5:  # Need minimum data points
            return
        
        try:
            # Calculate trends using simple linear regression
            def calculate_trend(values: List[float]) -> float:
                if len(values) < 2:
                    return 0.0
                x = np.arange(len(values))
                return np.polyfit(x, values, 1)[0]  # Slope of trend line
            
            cpu_trend = calculate_trend(cpu_history[-20:])  # Last 20 data points
            memory_trend = calculate_trend(memory_history[-20:])
            response_time_trend = calculate_trend(response_time_history[-10:])
            
            # Predictive scaling based on trends
            current_instances = self.instance_manager.get_instance_count()
            
            # If CPU is trending up rapidly, pre-emptively scale up
            if cpu_trend > 2.0 and np.mean(cpu_history[-5:]) > 60.0:  # 2% per interval, current >60%
                self.logger.info(f"Predictive scaling: CPU trending up ({cpu_trend:.2f}%/interval)")
                # Could trigger early scaling here
            
            # If response times are trending up, scale up proactively
            if response_time_trend > 0.1 and np.mean(response_time_history[-3:]) > 1.0:
                self.logger.info(f"Predictive scaling: Response time trending up ({response_time_trend:.3f}s/interval)")
                # Could trigger early scaling here
                
        except Exception as e:
            self.logger.error(f"Error in predictive scaling: {e}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a new scaling rule."""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str) -> bool:
        """Remove a scaling rule by name."""
        for i, rule in enumerate(self.scaling_rules):
            if rule.name == rule_name:
                del self.scaling_rules[i]
                self.logger.info(f"Removed scaling rule: {rule_name}")
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a scaling rule."""
        for rule in self.scaling_rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.info(f"Enabled scaling rule: {rule_name}")
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a scaling rule."""
        for rule in self.scaling_rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.info(f"Disabled scaling rule: {rule_name}")
                return True
        return False
    
    def manual_scale(self, target_instances: int, reason: str = "Manual scaling") -> bool:
        """Manually scale to target instance count."""
        current_instances = self.instance_manager.get_instance_count()
        
        if target_instances == current_instances:
            return True
        
        start_time = time.time()
        
        try:
            if target_instances > current_instances:
                # Scale up
                instances_to_add = target_instances - current_instances
                created_count = 0
                
                for _ in range(instances_to_add):
                    try:
                        self.instance_manager.create_instance()
                        created_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to create instance during manual scale-up: {e}")
                        break
                
                actual_new_count = current_instances + created_count
                direction = ScalingDirection.UP
                self.total_scale_ups += created_count
                
            else:
                # Scale down
                instances_to_remove = current_instances - target_instances
                idle_instances = self.instance_manager.get_idle_instances()
                instances_to_terminate = idle_instances[:instances_to_remove]
                
                terminated_count = 0
                for instance_id in instances_to_terminate:
                    if self.instance_manager.terminate_instance(instance_id):
                        terminated_count += 1
                
                actual_new_count = current_instances - terminated_count
                direction = ScalingDirection.DOWN
                self.total_scale_downs += terminated_count
            
            # Record manual scaling event
            event = ScalingEvent(
                timestamp=start_time,
                rule_name="manual_scaling",
                direction=direction,
                old_count=current_instances,
                new_count=actual_new_count,
                trigger_value=0.0,
                threshold=0.0,
                reason=reason,
                duration=time.time() - start_time,
                success=True
            )
            
            with self.scaling_lock:
                self.scaling_events.append(event)
            
            # Update metrics
            self.metrics_collector.update_active_instances(actual_new_count)
            
            self.logger.info(f"Manual scaling: {current_instances} -> {actual_new_count} instances ({reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Manual scaling failed: {e}")
            return False
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling statistics."""
        with self.scaling_lock:
            recent_events = [e for e in self.scaling_events if time.time() - e.timestamp < 3600]  # Last hour
            
            successful_events = [e for e in recent_events if e.success]
            scale_up_events = [e for e in successful_events if e.direction == ScalingDirection.UP]
            scale_down_events = [e for e in successful_events if e.direction == ScalingDirection.DOWN]
            
            # Calculate effectiveness metrics
            avg_scale_up_duration = np.mean([e.duration for e in scale_up_events]) if scale_up_events else 0.0
            avg_scale_down_duration = np.mean([e.duration for e in scale_down_events]) if scale_down_events else 0.0
            
            current_instances = self.instance_manager.get_instance_count()
            instance_stats = self.instance_manager.get_instance_statistics()
            
            return {
                'current_instances': current_instances,
                'total_scale_ups': self.total_scale_ups,
                'total_scale_downs': self.total_scale_downs,
                'total_scaling_decisions': self.total_scaling_decisions,
                'recent_scaling_events': len(recent_events),
                'successful_scaling_rate': len(successful_events) / max(1, len(recent_events)),
                'avg_scale_up_duration': avg_scale_up_duration,
                'avg_scale_down_duration': avg_scale_down_duration,
                'active_rules': len([r for r in self.scaling_rules if r.enabled]),
                'total_rules': len(self.scaling_rules),
                'predictive_scaling_enabled': self.enable_predictive_scaling,
                'instance_statistics': instance_stats,
                'rule_configurations': [
                    {
                        'name': rule.name,
                        'trigger': rule.trigger.value,
                        'enabled': rule.enabled,
                        'scale_up_threshold': rule.scale_up_threshold,
                        'scale_down_threshold': rule.scale_down_threshold,
                        'min_instances': rule.min_instances,
                        'max_instances': rule.max_instances,
                        'priority': rule.priority
                    }
                    for rule in self.scaling_rules
                ]
            }
    
    def get_recent_scaling_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent scaling events."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.scaling_lock:
            recent_events = [
                {
                    'timestamp': event.timestamp,
                    'rule_name': event.rule_name,
                    'direction': event.direction.value,
                    'old_count': event.old_count,
                    'new_count': event.new_count,
                    'trigger_value': event.trigger_value,
                    'threshold': event.threshold,
                    'reason': event.reason,
                    'duration': event.duration,
                    'success': event.success
                }
                for event in self.scaling_events
                if event.timestamp > cutoff_time
            ]
            
            return sorted(recent_events, key=lambda x: x['timestamp'], reverse=True)
    
    def __repr__(self) -> str:
        """String representation of auto-scaler."""
        return (f"AutoScaler(instances={self.instance_manager.get_instance_count()}, "
                f"rules={len(self.scaling_rules)}, scale_ups={self.total_scale_ups}, "
                f"scale_downs={self.total_scale_downs})")


# Context manager for load testing with auto-scaling
@asynccontextmanager
async def auto_scaled_load_test(auto_scaler: AutoScaler, 
                               duration_minutes: float = 10.0,
                               initial_instances: int = 2):
    """
    Context manager for running load tests with auto-scaling.
    
    Args:
        auto_scaler: AutoScaler instance
        duration_minutes: Duration of load test
        initial_instances: Initial number of instances
    """
    # Set initial instances
    auto_scaler.manual_scale(initial_instances, "Load test initialization")
    
    # Start auto-scaling
    auto_scaler.start_auto_scaling()
    
    start_time = time.time()
    
    try:
        yield auto_scaler
        
        # Wait for test duration
        await asyncio.sleep(duration_minutes * 60)
        
    finally:
        # Stop auto-scaling
        auto_scaler.stop_auto_scaling()
        
        # Get final statistics
        end_time = time.time()
        test_duration = end_time - start_time
        
        final_stats = auto_scaler.get_scaling_statistics()
        auto_scaler.logger.info(f"Load test completed after {test_duration:.1f}s")
        auto_scaler.logger.info(f"Final scaling statistics: {final_stats}")


def create_agent_factory(agent_class, *init_args, **init_kwargs) -> Callable[[], Any]:
    """
    Create an agent factory function for the instance manager.
    
    Args:
        agent_class: Class to instantiate
        *init_args: Arguments for agent initialization
        **init_kwargs: Keyword arguments for agent initialization
        
    Returns:
        Factory function that creates agent instances
    """
    def factory():
        return agent_class(*init_args, **init_kwargs)
    
    return factory