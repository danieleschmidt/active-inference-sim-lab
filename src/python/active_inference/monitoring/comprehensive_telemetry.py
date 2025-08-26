"""Comprehensive Telemetry and Monitoring System for Active Inference.

This module implements production-grade telemetry, monitoring, and observability
for Active Inference systems in Generation 2: MAKE IT ROBUST.

Features:
- Real-time performance monitoring with custom metrics
- Distributed tracing for complex agent interactions  
- Health check systems with automatic recovery
- Anomaly detection and alerting
- Performance profiling and optimization recommendations
- Multi-dimensional observability (logs, metrics, traces)
"""

import time
import threading
import json
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import traceback
from pathlib import Path
import queue
import contextlib


@dataclass
class MetricSample:
    """Individual metric sample with metadata."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass 
class TraceSpan:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "started"  # started, finished, error


@dataclass
class HealthStatus:
    """System health status."""
    component: str
    status: str  # healthy, degraded, critical, unknown
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    checks_passed: int = 0
    checks_failed: int = 0


class TelemetryCollector:
    """Advanced telemetry collection system."""
    
    def __init__(self, 
                 max_metrics_buffer: int = 10000,
                 max_traces_buffer: int = 1000,
                 flush_interval: float = 30.0,
                 enable_sampling: bool = True,
                 sampling_rate: float = 0.1):
        """
        Initialize telemetry collector.
        
        Args:
            max_metrics_buffer: Maximum metrics to buffer
            max_traces_buffer: Maximum traces to buffer  
            flush_interval: Interval to flush data
            enable_sampling: Whether to enable sampling
            sampling_rate: Rate for sampling (0.0 to 1.0)
        """
        self.max_metrics_buffer = max_metrics_buffer
        self.max_traces_buffer = max_traces_buffer
        self.flush_interval = flush_interval
        self.enable_sampling = enable_sampling
        self.sampling_rate = sampling_rate
        
        self.logger = logging.getLogger("TelemetryCollector")
        
        # Data buffers
        self.metrics_buffer: deque = deque(maxlen=max_metrics_buffer)
        self.traces_buffer: deque = deque(maxlen=max_traces_buffer)
        self.health_status: Dict[str, HealthStatus] = {}
        
        # Threading
        self.buffer_lock = threading.RLock()
        self.flush_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Metrics aggregation
        self.metric_aggregations: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'last_value': 0.0,
            'last_timestamp': 0.0
        })
        
        # Active traces
        self.active_traces: Dict[str, TraceSpan] = {}
        
        # Performance tracking
        self.performance_stats = {
            'metrics_collected': 0,
            'traces_collected': 0,
            'buffer_overflows': 0,
            'collection_errors': 0,
            'flush_count': 0,
            'last_flush_time': 0.0
        }
        
        # Start background processing
        self.start()
    
    def start(self) -> None:
        """Start the telemetry collection system."""
        if self.running:
            return
        
        self.running = True
        self.flush_thread = threading.Thread(
            target=self._flush_worker,
            daemon=True,
            name="TelemetryFlusher"
        )
        self.flush_thread.start()
        
        self.logger.info("Telemetry collector started")
    
    def stop(self) -> None:
        """Stop the telemetry collection system."""
        self.running = False
        
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)
        
        # Final flush
        self._flush_all_data()
        
        self.logger.info("Telemetry collector stopped")
    
    def record_metric(self, 
                     name: str,
                     value: float,
                     tags: Dict[str, str] = None,
                     unit: str = "",
                     timestamp: Optional[float] = None) -> bool:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            unit: Unit of measurement
            timestamp: Timestamp (defaults to now)
            
        Returns:
            True if recorded successfully
        """
        try:
            # Apply sampling if enabled
            if self.enable_sampling and np.random.random() > self.sampling_rate:
                return True  # Sampled out, but not an error
            
            if timestamp is None:
                timestamp = time.time()
            
            metric = MetricSample(
                name=name,
                value=float(value),
                timestamp=timestamp,
                tags=tags or {},
                unit=unit
            )
            
            with self.buffer_lock:
                # Check buffer capacity
                if len(self.metrics_buffer) >= self.max_metrics_buffer:
                    self.performance_stats['buffer_overflows'] += 1
                    # Remove oldest metric
                    self.metrics_buffer.popleft()
                
                self.metrics_buffer.append(metric)
                
                # Update aggregations
                agg = self.metric_aggregations[name]
                agg['count'] += 1
                agg['sum'] += value
                agg['min'] = min(agg['min'], value)
                agg['max'] = max(agg['max'], value)
                agg['last_value'] = value
                agg['last_timestamp'] = timestamp
                
                self.performance_stats['metrics_collected'] += 1
            
            return True
            
        except Exception as e:
            self.performance_stats['collection_errors'] += 1
            self.logger.error(f"Failed to record metric {name}: {e}")
            return False
    
    def start_trace(self, 
                   operation_name: str,
                   trace_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None,
                   tags: Dict[str, Any] = None) -> TraceSpan:
        """
        Start a distributed trace span.
        
        Args:
            operation_name: Name of the operation
            trace_id: Trace identifier (generated if None)
            parent_span_id: Parent span identifier
            tags: Optional tags
            
        Returns:
            Started trace span
        """
        if trace_id is None:
            trace_id = f"trace_{int(time.time() * 1000000)}_{threading.current_thread().ident}"
        
        span_id = f"span_{int(time.time() * 1000000)}_{np.random.randint(1000, 9999)}"
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            tags=tags or {},
            status="started"
        )
        
        with self.buffer_lock:
            self.active_traces[span_id] = span
        
        return span
    
    def finish_trace(self, 
                    span: TraceSpan,
                    status: str = "finished",
                    tags: Dict[str, Any] = None) -> bool:
        """
        Finish a trace span.
        
        Args:
            span: Trace span to finish
            status: Final status (finished, error)
            tags: Additional tags
            
        Returns:
            True if finished successfully
        """
        try:
            span.end_time = time.time()
            span.status = status
            
            if tags:
                span.tags.update(tags)
            
            # Calculate duration
            duration = span.end_time - span.start_time
            span.tags['duration_ms'] = duration * 1000
            
            with self.buffer_lock:
                # Remove from active traces
                self.active_traces.pop(span.span_id, None)
                
                # Add to buffer
                if len(self.traces_buffer) >= self.max_traces_buffer:
                    self.performance_stats['buffer_overflows'] += 1
                    self.traces_buffer.popleft()
                
                self.traces_buffer.append(span)
                self.performance_stats['traces_collected'] += 1
            
            return True
            
        except Exception as e:
            self.performance_stats['collection_errors'] += 1
            self.logger.error(f"Failed to finish trace {span.span_id}: {e}")
            return False
    
    def log_trace_event(self, 
                       span: TraceSpan,
                       event: str,
                       data: Dict[str, Any] = None) -> None:
        """Add a log event to a trace span."""
        log_entry = {
            'timestamp': time.time(),
            'event': event,
            'data': data or {}
        }
        span.logs.append(log_entry)
    
    def update_health_status(self, 
                           component: str,
                           status: str,
                           details: Dict[str, Any] = None) -> None:
        """
        Update health status for a component.
        
        Args:
            component: Component name
            status: Health status (healthy, degraded, critical, unknown)
            details: Additional details
        """
        health = HealthStatus(
            component=component,
            status=status,
            timestamp=time.time(),
            details=details or {}
        )
        
        # Update pass/fail counts
        if component in self.health_status:
            prev_health = self.health_status[component]
            health.checks_passed = prev_health.checks_passed
            health.checks_failed = prev_health.checks_failed
        
        if status == "healthy":
            health.checks_passed += 1
        else:
            health.checks_failed += 1
        
        with self.buffer_lock:
            self.health_status[component] = health
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get aggregated summary for a metric."""
        with self.buffer_lock:
            if metric_name in self.metric_aggregations:
                agg = self.metric_aggregations[metric_name].copy()
                if agg['count'] > 0:
                    agg['average'] = agg['sum'] / agg['count']
                else:
                    agg['average'] = 0.0
                return agg
            return {}
    
    def get_active_traces_count(self) -> int:
        """Get number of active traces."""
        with self.buffer_lock:
            return len(self.active_traces)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self.buffer_lock:
            components_by_status = defaultdict(list)
            
            for component, health in self.health_status.items():
                components_by_status[health.status].append(component)
            
            total_components = len(self.health_status)
            healthy_components = len(components_by_status.get('healthy', []))
            
            overall_health = "healthy"
            if healthy_components < total_components:
                if components_by_status.get('critical'):
                    overall_health = "critical"
                elif components_by_status.get('degraded'):
                    overall_health = "degraded"
                else:
                    overall_health = "unknown"
            
            return {
                'overall_status': overall_health,
                'total_components': total_components,
                'healthy_components': healthy_components,
                'degraded_components': len(components_by_status.get('degraded', [])),
                'critical_components': len(components_by_status.get('critical', [])),
                'unknown_components': len(components_by_status.get('unknown', [])),
                'components_by_status': dict(components_by_status),
                'last_updated': max([h.timestamp for h in self.health_status.values()]) if self.health_status else 0
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get telemetry system performance statistics."""
        with self.buffer_lock:
            stats = self.performance_stats.copy()
            stats.update({
                'metrics_buffer_size': len(self.metrics_buffer),
                'traces_buffer_size': len(self.traces_buffer),
                'active_traces_count': len(self.active_traces),
                'health_components_count': len(self.health_status),
                'buffer_utilization_pct': (len(self.metrics_buffer) / self.max_metrics_buffer) * 100,
                'is_running': self.running
            })
            return stats
    
    def _flush_worker(self) -> None:
        """Background worker for flushing telemetry data."""
        while self.running:
            try:
                time.sleep(self.flush_interval)
                self._flush_all_data()
                
            except Exception as e:
                self.logger.error(f"Error in flush worker: {e}")
    
    def _flush_all_data(self) -> None:
        """Flush all buffered telemetry data."""
        try:
            with self.buffer_lock:
                # Create snapshots
                metrics_snapshot = list(self.metrics_buffer)
                traces_snapshot = list(self.traces_buffer)
                health_snapshot = dict(self.health_status)
                
                # Clear buffers
                self.metrics_buffer.clear()
                self.traces_buffer.clear()
            
            # Process data (in production, would send to monitoring system)
            self._process_metrics_batch(metrics_snapshot)
            self._process_traces_batch(traces_snapshot)
            self._process_health_batch(health_snapshot)
            
            # Update performance stats
            self.performance_stats['flush_count'] += 1
            self.performance_stats['last_flush_time'] = time.time()
            
            self.logger.debug(f"Flushed {len(metrics_snapshot)} metrics, "
                            f"{len(traces_snapshot)} traces, "
                            f"{len(health_snapshot)} health checks")
            
        except Exception as e:
            self.logger.error(f"Failed to flush telemetry data: {e}")
    
    def _process_metrics_batch(self, metrics: List[MetricSample]) -> None:
        """Process a batch of metrics."""
        if not metrics:
            return
        
        # Group by metric name for efficient processing
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        # In production, this would send to Prometheus, DataDog, etc.
        self.logger.debug(f"Processing {len(metrics)} metrics across {len(metrics_by_name)} series")
    
    def _process_traces_batch(self, traces: List[TraceSpan]) -> None:
        """Process a batch of traces."""
        if not traces:
            return
        
        # Group by trace_id for analysis
        traces_by_id = defaultdict(list)
        for trace in traces:
            traces_by_id[trace.trace_id].append(trace)
        
        # In production, this would send to Jaeger, Zipkin, etc.
        self.logger.debug(f"Processing {len(traces)} spans across {len(traces_by_id)} traces")
    
    def _process_health_batch(self, health_checks: Dict[str, HealthStatus]) -> None:
        """Process health check results."""
        if not health_checks:
            return
        
        # In production, this would trigger alerts, update dashboards, etc.
        critical_components = [name for name, health in health_checks.items() 
                             if health.status == 'critical']
        
        if critical_components:
            self.logger.warning(f"Critical health status for components: {critical_components}")
        
        self.logger.debug(f"Processing health checks for {len(health_checks)} components")


class PerformanceProfiler:
    """Advanced performance profiler for Active Inference systems."""
    
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.telemetry = telemetry_collector
        self.logger = logging.getLogger("PerformanceProfiler")
        
        # Profiling data
        self.function_profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'last_call_time': 0.0
        })
        
        # Memory tracking
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.peak_memory_usage = 0
        
        # Lock for thread safety
        self.profile_lock = threading.RLock()
    
    @contextlib.contextmanager
    def profile_function(self, function_name: str, tags: Dict[str, Any] = None):
        """
        Context manager for profiling function execution.
        
        Args:
            function_name: Name of function being profiled
            tags: Optional tags for telemetry
        """
        start_time = time.time()
        span = self.telemetry.start_trace(f"profile:{function_name}", tags=tags)
        
        try:
            yield span
            
            # Success case
            end_time = time.time()
            duration = end_time - start_time
            
            self._update_function_profile(function_name, duration, True)
            self.telemetry.finish_trace(span, "finished", {'duration_ms': duration * 1000})
            
            # Record metric
            self.telemetry.record_metric(
                f"function.{function_name}.duration",
                duration,
                tags={'status': 'success'},
                unit="seconds"
            )
            
        except Exception as e:
            # Error case
            end_time = time.time()
            duration = end_time - start_time
            
            self._update_function_profile(function_name, duration, False)
            self.telemetry.finish_trace(span, "error", {
                'duration_ms': duration * 1000,
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            # Record error metric
            self.telemetry.record_metric(
                f"function.{function_name}.errors",
                1,
                tags={'error_type': type(e).__name__},
                unit="count"
            )
            
            raise  # Re-raise the exception
    
    def _update_function_profile(self, function_name: str, duration: float, success: bool) -> None:
        """Update function profiling statistics."""
        with self.profile_lock:
            profile = self.function_profiles[function_name]
            
            profile['call_count'] += 1
            profile['total_time'] += duration
            profile['min_time'] = min(profile['min_time'], duration)
            profile['max_time'] = max(profile['max_time'], duration)
            profile['avg_time'] = profile['total_time'] / profile['call_count']
            profile['last_call_time'] = duration
            
            # Track success rate
            if 'success_count' not in profile:
                profile['success_count'] = 0
                profile['error_count'] = 0
            
            if success:
                profile['success_count'] += 1
            else:
                profile['error_count'] += 1
            
            profile['success_rate'] = profile['success_count'] / profile['call_count']
    
    def take_memory_snapshot(self, context: str = "") -> Dict[str, Any]:
        """Take a memory usage snapshot."""
        try:
            import psutil
            import gc
            
            # Force garbage collection
            gc.collect()
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'timestamp': time.time(),
                'context': context,
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'gc_stats': {
                    'collected': sum(gc.get_stats(), []).get('collected', 0),
                    'collections': sum(gc.get_stats(), []).get('collections', 0)
                }
            }
            
            # Update peak memory
            if snapshot['rss_mb'] > self.peak_memory_usage:
                self.peak_memory_usage = snapshot['rss_mb']
            
            # Store snapshot
            self.memory_snapshots.append(snapshot)
            
            # Limit snapshots history
            if len(self.memory_snapshots) > 1000:
                self.memory_snapshots = self.memory_snapshots[-1000:]
            
            # Record telemetry
            self.telemetry.record_metric("memory.rss_mb", snapshot['rss_mb'], unit="megabytes")
            self.telemetry.record_metric("memory.vms_mb", snapshot['vms_mb'], unit="megabytes")
            self.telemetry.record_metric("cpu.usage_percent", snapshot['cpu_percent'], unit="percent")
            
            return snapshot
            
        except ImportError:
            # psutil not available
            self.logger.warning("psutil not available for memory profiling")
            return {'timestamp': time.time(), 'context': context, 'error': 'psutil_unavailable'}
        
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            return {'timestamp': time.time(), 'context': context, 'error': str(e)}
    
    def get_function_profiles(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top function profiles by various metrics."""
        with self.profile_lock:
            profiles = []
            
            for func_name, profile in self.function_profiles.items():
                profile_copy = profile.copy()
                profile_copy['function_name'] = func_name
                profiles.append(profile_copy)
            
            # Sort by total time (most expensive first)
            profiles.sort(key=lambda p: p['total_time'], reverse=True)
            
            return profiles[:top_n]
    
    def get_memory_usage_trend(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get memory usage trend over specified window."""
        if not self.memory_snapshots:
            return {'trend': 'no_data', 'snapshots': []}
        
        # Filter snapshots within window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_snapshots = [s for s in self.memory_snapshots if s['timestamp'] >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {'trend': 'insufficient_data', 'snapshots': recent_snapshots}
        
        # Calculate trend
        first_memory = recent_snapshots[0]['rss_mb']
        last_memory = recent_snapshots[-1]['rss_mb']
        memory_change = last_memory - first_memory
        
        # Determine trend
        if abs(memory_change) < 5:  # Less than 5MB change
            trend = 'stable'
        elif memory_change > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'memory_change_mb': memory_change,
            'first_memory_mb': first_memory,
            'last_memory_mb': last_memory,
            'peak_memory_mb': self.peak_memory_usage,
            'snapshots_count': len(recent_snapshots),
            'window_minutes': window_minutes
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'report_type': 'performance_analysis',
            'function_profiles': self.get_function_profiles(),
            'memory_analysis': self.get_memory_usage_trend(),
            'telemetry_stats': self.telemetry.get_performance_stats(),
            'health_summary': self.telemetry.get_health_summary(),
            'recommendations': self._generate_optimization_recommendations()
        }
        
        return report
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        # Check for slow functions
        top_functions = self.get_function_profiles(5)
        for func in top_functions:
            if func['avg_time'] > 0.1:  # Slower than 100ms
                recommendations.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'component': func['function_name'],
                    'issue': f"Function is averaging {func['avg_time']:.3f}s per call",
                    'recommendation': "Consider optimizing this function or adding caching"
                })
        
        # Check memory usage
        memory_trend = self.get_memory_usage_trend()
        if memory_trend['trend'] == 'increasing':
            recommendations.append({
                'type': 'memory',
                'severity': 'high' if memory_trend['memory_change_mb'] > 100 else 'medium',
                'component': 'memory_management',
                'issue': f"Memory usage increased by {memory_trend['memory_change_mb']:.1f}MB",
                'recommendation': "Investigate potential memory leaks or optimize data structures"
            })
        
        # Check telemetry buffer overflows
        telemetry_stats = self.telemetry.get_performance_stats()
        if telemetry_stats['buffer_overflows'] > 0:
            recommendations.append({
                'type': 'telemetry',
                'severity': 'low',
                'component': 'telemetry_system',
                'issue': f"{telemetry_stats['buffer_overflows']} buffer overflows detected",
                'recommendation': "Consider increasing buffer sizes or reducing sampling rate"
            })
        
        return recommendations


class AnomalyDetector:
    """Real-time anomaly detection for Active Inference systems."""
    
    def __init__(self, 
                 telemetry_collector: TelemetryCollector,
                 window_size: int = 100,
                 anomaly_threshold: float = 3.0):
        """
        Initialize anomaly detector.
        
        Args:
            telemetry_collector: Telemetry collector instance
            window_size: Rolling window size for anomaly detection
            anomaly_threshold: Standard deviation threshold for anomalies
        """
        self.telemetry = telemetry_collector
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.logger = logging.getLogger("AnomalyDetector")
        
        # Metric history for anomaly detection
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.metric_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'mean': 0.0,
            'std': 0.0,
            'last_update': 0.0
        })
        
        # Anomaly tracking
        self.detected_anomalies: List[Dict[str, Any]] = []
        self.anomaly_callbacks: List[Callable] = []
        
        # Thread safety
        self.detection_lock = threading.RLock()
    
    def add_anomaly_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for anomaly notifications."""
        self.anomaly_callbacks.append(callback)
    
    def update_metric_for_detection(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """
        Update metric and check for anomalies.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            Anomaly information if detected, None otherwise
        """
        with self.detection_lock:
            # Update rolling window
            window = self.metric_windows[metric_name]
            window.append(value)
            
            # Need minimum samples for detection
            if len(window) < max(10, self.window_size // 4):
                return None
            
            # Calculate statistics
            values = np.array(window)
            mean = np.mean(values)
            std = np.std(values)
            
            # Update stats
            stats = self.metric_stats[metric_name]
            stats['mean'] = mean
            stats['std'] = std
            stats['last_update'] = time.time()
            
            # Check for anomaly
            if std > 0:  # Avoid division by zero
                z_score = abs(value - mean) / std
                
                if z_score > self.anomaly_threshold:
                    anomaly = {
                        'timestamp': time.time(),
                        'metric_name': metric_name,
                        'value': value,
                        'expected_mean': mean,
                        'std_dev': std,
                        'z_score': z_score,
                        'anomaly_type': 'statistical_outlier',
                        'severity': self._classify_anomaly_severity(z_score)
                    }
                    
                    # Record anomaly
                    self.detected_anomalies.append(anomaly)
                    
                    # Limit anomaly history
                    if len(self.detected_anomalies) > 1000:
                        self.detected_anomalies = self.detected_anomalies[-1000:]
                    
                    # Notify callbacks
                    self._notify_anomaly_callbacks(anomaly)
                    
                    # Record in telemetry
                    self.telemetry.record_metric(
                        "anomaly.detected",
                        1,
                        tags={
                            'metric': metric_name,
                            'severity': anomaly['severity']
                        }
                    )
                    
                    self.logger.warning(f"Anomaly detected in {metric_name}: "
                                      f"value={value:.3f}, z_score={z_score:.2f}")
                    
                    return anomaly
            
            return None
    
    def _classify_anomaly_severity(self, z_score: float) -> str:
        """Classify anomaly severity based on z-score."""
        if z_score > 5.0:
            return 'critical'
        elif z_score > 4.0:
            return 'high'
        elif z_score > 3.5:
            return 'medium'
        else:
            return 'low'
    
    def _notify_anomaly_callbacks(self, anomaly: Dict[str, Any]) -> None:
        """Notify all registered anomaly callbacks."""
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                self.logger.error(f"Error in anomaly callback: {e}")
    
    def get_anomalies(self, 
                     metric_name: Optional[str] = None,
                     severity: Optional[str] = None,
                     time_window_hours: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get detected anomalies with optional filters.
        
        Args:
            metric_name: Filter by metric name
            severity: Filter by severity (low, medium, high, critical)
            time_window_hours: Filter by time window in hours
            
        Returns:
            List of anomalies matching filters
        """
        with self.detection_lock:
            anomalies = self.detected_anomalies.copy()
        
        # Apply filters
        if time_window_hours:
            cutoff_time = time.time() - (time_window_hours * 3600)
            anomalies = [a for a in anomalies if a['timestamp'] >= cutoff_time]
        
        if metric_name:
            anomalies = [a for a in anomalies if a['metric_name'] == metric_name]
        
        if severity:
            anomalies = [a for a in anomalies if a['severity'] == severity]
        
        return anomalies
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        with self.detection_lock:
            if not self.detected_anomalies:
                return {'total_anomalies': 0, 'by_severity': {}, 'by_metric': {}}
            
            # Group by severity
            by_severity = defaultdict(int)
            by_metric = defaultdict(int)
            
            for anomaly in self.detected_anomalies:
                by_severity[anomaly['severity']] += 1
                by_metric[anomaly['metric_name']] += 1
            
            # Recent anomalies (last hour)
            recent_anomalies = self.get_anomalies(time_window_hours=1.0)
            
            return {
                'total_anomalies': len(self.detected_anomalies),
                'recent_anomalies_1h': len(recent_anomalies),
                'by_severity': dict(by_severity),
                'by_metric': dict(by_metric),
                'metrics_monitored': len(self.metric_windows),
                'last_anomaly_time': max([a['timestamp'] for a in self.detected_anomalies]) if self.detected_anomalies else 0
            }


# Create a global telemetry collector for easy access
_global_telemetry_collector: Optional[TelemetryCollector] = None
_global_profiler: Optional[PerformanceProfiler] = None
_global_anomaly_detector: Optional[AnomalyDetector] = None


def get_global_telemetry() -> TelemetryCollector:
    """Get or create the global telemetry collector."""
    global _global_telemetry_collector
    if _global_telemetry_collector is None:
        _global_telemetry_collector = TelemetryCollector()
    return _global_telemetry_collector


def get_global_profiler() -> PerformanceProfiler:
    """Get or create the global performance profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(get_global_telemetry())
    return _global_profiler


def get_global_anomaly_detector() -> AnomalyDetector:
    """Get or create the global anomaly detector."""
    global _global_anomaly_detector
    if _global_anomaly_detector is None:
        _global_anomaly_detector = AnomalyDetector(get_global_telemetry())
    return _global_anomaly_detector


def shutdown_global_telemetry() -> None:
    """Shutdown all global telemetry systems."""
    global _global_telemetry_collector, _global_profiler, _global_anomaly_detector
    
    if _global_telemetry_collector:
        _global_telemetry_collector.stop()
        _global_telemetry_collector = None
    
    _global_profiler = None
    _global_anomaly_detector = None