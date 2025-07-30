# Comprehensive Observability Setup for Active Inference Sim Lab

## Overview

This document provides a complete monitoring and observability strategy for the Active Inference Sim Lab, covering metrics, logging, tracing, and alerting across development, testing, and production environments.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Application │───▶│ OpenTelemetry │───▶│ Observability │
│   Metrics   │    │   Collector   │    │   Backend     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Logging   │───▶│  Structured │───▶│   Storage   │
│  Framework  │    │   Logging   │    │  & Analysis │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 1. Application Metrics

### Core Performance Metrics

Create `src/python/active_inference/monitoring/metrics.py`:

```python
"""
Comprehensive metrics collection for Active Inference Sim Lab
"""
import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from collections import defaultdict, deque

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

@dataclass
class MetricValue:
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float

class MetricsCollector:
    """Centralized metrics collection for active inference operations"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self._metrics = defaultdict(deque)
        self._lock = threading.Lock()
        
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Performance metrics
        self.inference_duration = Histogram(
            'ai_inference_duration_seconds',
            'Time spent on inference operations',
            ['method', 'model_type']
        )
        
        self.belief_update_duration = Histogram(
            'ai_belief_update_duration_seconds',
            'Time spent updating beliefs',
            ['update_method']
        )
        
        self.free_energy_computation = Histogram(
            'ai_free_energy_computation_seconds',
            'Time spent computing free energy',
            ['complexity_level']
        )
        
        # Accuracy metrics
        self.prediction_accuracy = Histogram(
            'ai_prediction_accuracy_ratio',
            'Prediction accuracy ratios',
            ['environment', 'agent_type']
        )
        
        self.convergence_iterations = Histogram(
            'ai_convergence_iterations_total',
            'Number of iterations to convergence',
            ['algorithm', 'tolerance']
        )
        
        # Resource utilization
        self.memory_usage = Gauge(
            'ai_memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        self.cpu_usage = Gauge(
            'ai_cpu_usage_percent',
            'CPU usage percentage',
            ['process']
        )
        
        # Error metrics
        self.error_count = Counter(
            'ai_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Model information
        self.model_info = Info(
            'ai_model_info',
            'Information about the active inference model'
        )
    
    @contextmanager
    def measure_time(self, metric_name: str, labels: Dict[str, str] = None):
        """Context manager for measuring execution time"""
        labels = labels or {}
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(metric_name, duration, labels)
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        labels = labels or {}
        
        with self._lock:
            self._metrics[name].append(MetricValue(
                name=name,
                value=value,
                labels=labels,
                timestamp=time.time()
            ))
            
            # Keep only last 1000 measurements per metric
            if len(self._metrics[name]) > 1000:
                self._metrics[name].popleft()
        
        # Update Prometheus metrics if available
        if self.enable_prometheus:
            self._update_prometheus_metric(name, value, labels)
    
    def _update_prometheus_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Update Prometheus metrics"""
        if name == 'inference_duration':
            self.inference_duration.labels(**labels).observe(value)
        elif name == 'belief_update_duration':
            self.belief_update_duration.labels(**labels).observe(value)
        elif name == 'free_energy_computation':
            self.free_energy_computation.labels(**labels).observe(value)
        elif name == 'prediction_accuracy':
            self.prediction_accuracy.labels(**labels).observe(value)
        elif name == 'convergence_iterations':
            self.convergence_iterations.labels(**labels).observe(value)
        elif name == 'memory_usage':
            self.memory_usage.labels(**labels).set(value)
        elif name == 'cpu_usage':
            self.cpu_usage.labels(**labels).set(value)
        elif name.startswith('error_'):
            self.error_count.labels(error_type=name, **labels).inc()
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics"""
        process = psutil.Process()
        
        metrics = {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'open_files': len(process.open_files()),
        }
        
        # Record system metrics
        for name, value in metrics.items():
            self.record_metric(f'system_{name}', value)
        
        return metrics
    
    def get_metric_summary(self, name: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_values = [
                m.value for m in self._metrics[name]
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'mean': sum(recent_values) / len(recent_values),
            'latest': recent_values[-1] if recent_values else 0
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()
```

### Structured Logging Configuration

Create `src/python/active_inference/monitoring/logging_config.py`:

```python
"""
Structured logging configuration for Active Inference Sim Lab
"""
import os
import sys
import json
import logging
import logging.config
from typing import Dict, Any
from datetime import datetime, timezone

# JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'exc_info', 'exc_text', 'stack_info', 'getMessage'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    structured: bool = True,
    include_performance: bool = True
) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Whether to use structured JSON logging
        include_performance: Whether to include performance logging
    """
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(funcName)s(): %(message)s'
            },
            'json': {
                '()': JSONFormatter,
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'json' if structured else 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'active_inference': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'active_inference.performance': {
                'level': 'DEBUG' if include_performance else 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'active_inference.security': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'json' if structured else 'detailed',
            'filename': log_file,
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5
        }
        
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            logger_config['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    logging.config.dictConfig(config)

# Performance logging decorator
def log_performance(func):
    """Decorator to log function performance metrics"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f'active_inference.performance.{func.__module__}')
        start_time = datetime.now(timezone.utc)
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(
                f"Function executed successfully",
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'duration_seconds': duration,
                    'success': True
                }
            )
            return result
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(
                f"Function execution failed",
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'duration_seconds': duration,
                    'success': False,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )
            raise
    
    return wrapper
```

## 2. OpenTelemetry Integration

Create `src/python/active_inference/monitoring/tracing.py`:

```python
"""
OpenTelemetry tracing configuration for distributed tracing
"""
from typing import Optional, Dict, Any
import os

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

class TracingManager:
    """Manages OpenTelemetry tracing configuration"""
    
    def __init__(self):
        self.tracer = None
        self.initialized = False
    
    def initialize(
        self,
        service_name: str = "active-inference-sim-lab",
        service_version: str = "0.1.0",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None
    ) -> bool:
        """Initialize OpenTelemetry tracing"""
        
        if not OTEL_AVAILABLE:
            return False
        
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": os.getenv("ENVIRONMENT", "development")
        })
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Configure exporters
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(':')[0],
                agent_port=int(jaeger_endpoint.split(':')[1]) if ':' in jaeger_endpoint else 14268,
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Set up auto-instrumentation
        RequestsInstrumentor().instrument()
        LoggingInstrumentor().instrument()
        
        self.tracer = trace.get_tracer(__name__)
        self.initialized = True
        return True
    
    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a new span for tracing"""
        if not self.initialized or not self.tracer:
            return nullcontext()
        
        return self.tracer.start_as_current_span(name, attributes=attributes or {})

# Global tracing manager
tracing_manager = TracingManager()

# Utility context manager for null operations
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
```

## 3. Health Check Endpoints

Create `src/python/active_inference/monitoring/health.py`:

```python
"""
Health check endpoints and system status monitoring
"""
import json
import time
import psutil
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Dict[str, Any] = None

class HealthMonitor:
    """Comprehensive health monitoring for the application"""
    
    def __init__(self):
        self.checks = {}
        self.startup_time = time.time()
    
    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self.checks.items():
            start_time = time.time()
            
            try:
                status, message, details = check_func()
                duration_ms = (time.time() - start_time) * 1000
                
                check_result = HealthCheck(
                    name=name,
                    status=status,
                    message=message,
                    duration_ms=duration_ms,
                    details=details
                )
                
                results.append(check_result)
                
                # Update overall status
                if status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                check_result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    duration_ms=duration_ms
                )
                results.append(check_result)
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.startup_time,
            "checks": [asdict(result) for result in results]
        }

# Default health checks
def check_memory_usage():
    """Check system memory usage"""
    memory = psutil.virtual_memory()
    
    if memory.percent > 90:
        return HealthStatus.UNHEALTHY, f"Memory usage critical: {memory.percent}%", {
            "memory_percent": memory.percent,
            "available_mb": memory.available / 1024 / 1024
        }
    elif memory.percent > 80:
        return HealthStatus.DEGRADED, f"Memory usage high: {memory.percent}%", {
            "memory_percent": memory.percent,
            "available_mb": memory.available / 1024 / 1024
        }
    else:
        return HealthStatus.HEALTHY, f"Memory usage normal: {memory.percent}%", {
            "memory_percent": memory.percent,
            "available_mb": memory.available / 1024 / 1024
        }

def check_cpu_usage():
    """Check CPU usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    
    if cpu_percent > 95:
        return HealthStatus.UNHEALTHY, f"CPU usage critical: {cpu_percent}%", {
            "cpu_percent": cpu_percent
        }
    elif cpu_percent > 85:
        return HealthStatus.DEGRADED, f"CPU usage high: {cpu_percent}%", {
            "cpu_percent": cpu_percent
        }
    else:
        return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent}%", {
            "cpu_percent": cpu_percent
        }

def check_disk_space():
    """Check disk space"""
    disk = psutil.disk_usage('/')
    
    if disk.percent > 95:
        return HealthStatus.UNHEALTHY, f"Disk usage critical: {disk.percent}%", {
            "disk_percent": disk.percent,
            "free_gb": disk.free / 1024 / 1024 / 1024
        }
    elif disk.percent > 85:
        return HealthStatus.DEGRADED, f"Disk usage high: {disk.percent}%", {
            "disk_percent": disk.percent,
            "free_gb": disk.free / 1024 / 1024 / 1024
        }
    else:
        return HealthStatus.HEALTHY, f"Disk usage normal: {disk.percent}%", {
            "disk_percent": disk.percent,
            "free_gb": disk.free / 1024 / 1024 / 1024
        }

def check_active_inference_components():
    """Check core active inference components"""
    try:
        # Basic import check
        import active_inference
        
        # Check if core modules are available
        required_modules = ['core', 'inference', 'planning']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(f'active_inference.{module}')
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            return HealthStatus.DEGRADED, f"Missing modules: {missing_modules}", {
                "missing_modules": missing_modules
            }
        
        return HealthStatus.HEALTHY, "All components available", {}
        
    except ImportError as e:
        return HealthStatus.UNHEALTHY, f"Active inference module not available: {e}", {}

# Initialize global health monitor
health_monitor = HealthMonitor()
health_monitor.register_check("memory", check_memory_usage)
health_monitor.register_check("cpu", check_cpu_usage)
health_monitor.register_check("disk", check_disk_space)
health_monitor.register_check("components", check_active_inference_components)
```

## 4. Alerting Configuration

Create `docs/monitoring/alerting-rules.yml`:

```yaml
# Alerting rules for Active Inference Sim Lab
groups:
  - name: active_inference_performance
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, ai_inference_duration_seconds) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile inference latency is {{ $value }}s"

      - alert: HighMemoryUsage
        expr: ai_memory_usage_bytes / (1024*1024*1024) > 8
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"

      - alert: FrequentErrors
        expr: rate(ai_errors_total[5m]) > 0.1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value }} errors/second"

  - name: active_inference_accuracy
    rules:
      - alert: LowPredictionAccuracy
        expr: histogram_quantile(0.5, ai_prediction_accuracy_ratio) < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low prediction accuracy"
          description: "Median prediction accuracy is {{ $value }}"

      - alert: SlowConvergence
        expr: histogram_quantile(0.95, ai_convergence_iterations_total) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow algorithm convergence"
          description: "95th percentile convergence iterations: {{ $value }}"

  - name: system_health
    rules:
      - alert: SystemDown
        expr: up == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "System is down"
          description: "Active Inference Sim Lab is not responding"

      - alert: HighCPUUsage
        expr: ai_cpu_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
```

## 5. Integration Instructions

### Required Dependencies

Add to `requirements-dev.txt`:

```
# Monitoring and Observability
prometheus-client>=0.17.1
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-jaeger-thrift>=1.20.0
opentelemetry-exporter-otlp-proto-grpc>=1.20.0
opentelemetry-instrumentation-requests>=0.41b0
opentelemetry-instrumentation-logging>=0.41b0
psutil>=5.9.0
```

### Environment Variables

```bash
# Monitoring Configuration
ENABLE_METRICS=true
ENABLE_TRACING=true
PROMETHEUS_PORT=8000
JAEGER_ENDPOINT=localhost:14268
OTLP_ENDPOINT=http://localhost:4317
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true
```

### Docker Compose for Local Development

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./docs/monitoring/alerting-rules.yml:/etc/prometheus/alerts.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"

volumes:
  grafana-storage:
```

This comprehensive observability setup provides enterprise-grade monitoring, logging, tracing, and alerting capabilities for the Active Inference Sim Lab.