# Observability and Monitoring Setup

## Overview

This document outlines the monitoring and observability setup for the active-inference-sim-lab project, including logging, metrics, tracing, and alerting configurations.

## Architecture

```
Application Layer
    ↓
Instrumentation Layer (OpenTelemetry)
    ↓
Collection Layer (OTEL Collector)
    ↓
Storage & Analysis Layer (Prometheus, Jaeger, ELK)
    ↓
Visualization Layer (Grafana, Kibana)
    ↓
Alerting Layer (AlertManager, PagerDuty)
```

## Logging Configuration

### Python Logging Setup

Create `src/python/active_inference/utils/logging_config.py`:

```python
"""
Centralized logging configuration with structured logging and observability integration.
"""

import logging
import logging.config
import os
import sys
from typing import Dict, Any
import json
from datetime import datetime

# Structured log formatter
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
            
        return json.dumps(log_entry)

# Default logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'structured': {
            '()': StructuredFormatter,
        },
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': sys.stdout
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'structured',
            'filename': 'logs/active_inference.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'structured',
            'filename': 'logs/errors.log',
            'maxBytes': 10485760,
            'backupCount': 3
        }
    },
    'loggers': {
        'active_inference': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        },
        'active_inference.performance': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}

def setup_logging(config: Dict[str, Any] = None) -> None:
    """Setup logging configuration."""
    if config is None:
        config = LOGGING_CONFIG
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Apply configuration
    logging.config.dictConfig(config)
```

### C++ Logging Setup

Create `cpp/include/logging/logger.hpp`:

```cpp
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>
#include <memory>
#include <string>

namespace active_inference {
namespace logging {

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }
    
    void initialize(const std::string& log_level = "info") {
        try {
            // Create sinks
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
            
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                "logs/active_inference_cpp.log", 10 * 1024 * 1024, 5);
            file_sink->set_level(spdlog::level::debug);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [%s:%#] %v");
            
            // Create logger
            spdlog::sinks_init_list sinks = {console_sink, file_sink};
            auto logger = std::make_shared<spdlog::logger>("active_inference", sinks);
            
            // Set level
            if (log_level == "debug") logger->set_level(spdlog::level::debug);
            else if (log_level == "info") logger->set_level(spdlog::level::info);
            else if (log_level == "warn") logger->set_level(spdlog::level::warn);
            else if (log_level == "error") logger->set_level(spdlog::level::err);
            
            spdlog::register_logger(logger);
            spdlog::set_default_logger(logger);
            
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        }
    }
    
private:
    Logger() = default;
};

// Convenience macros
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)

} // namespace logging
} // namespace active_inference
```

## Metrics Collection

### OpenTelemetry Configuration

Create `monitoring/otel-config.yaml`:

```yaml
# OpenTelemetry Collector Configuration
receivers:
  # OTLP receiver for direct instrumentation
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
  
  # Prometheus receiver for existing metrics
  prometheus:
    config:
      scrape_configs:
        - job_name: 'active-inference-app'
          static_configs:
            - targets: ['localhost:8080']
          scrape_interval: 30s
          metrics_path: /metrics

processors:
  # Batch processor for performance
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  # Resource processor to add metadata
  resource:
    attributes:
      - key: service.name
        value: active-inference-sim-lab
        action: insert
      - key: service.version
        value: ${SERVICE_VERSION}
        action: insert
      - key: deployment.environment
        value: ${ENVIRONMENT}
        action: insert

exporters:
  # Prometheus exporter
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: active_inference
    const_labels:
      service: active-inference-sim-lab
  
  # Jaeger exporter for tracing
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  # Logging exporter for debugging
  logging:
    loglevel: info

service:
  pipelines:
    metrics:
      receivers: [otlp, prometheus]
      processors: [resource, batch]
      exporters: [prometheus, logging]
    
    traces:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [jaeger, logging]
    
    logs:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [logging]
  
  extensions: [health_check, pprof]
```

### Custom Metrics

Create `src/python/active_inference/monitoring/metrics.py`:

```python
"""
Custom metrics collection for active inference experiments.
"""

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

class ActiveInferenceMetrics:
    def __init__(self, service_name: str = "active-inference-sim-lab"):
        # Setup OpenTelemetry metrics
        resource = Resource.create({"service.name": service_name})
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        set_meter_provider(provider)
        
        self.meter = get_meter_provider().get_meter(__name__)
        
        # Define custom metrics
        self.experiment_duration = self.meter.create_histogram(
            "experiment_duration_seconds",
            description="Duration of experiments in seconds",
            unit="s"
        )
        
        self.free_energy_values = self.meter.create_histogram(
            "free_energy_value",
            description="Free energy values during inference",
            unit="nats"
        )
        
        self.inference_iterations = self.meter.create_counter(
            "inference_iterations_total",
            description="Total inference iterations performed"
        )
        
        self.active_experiments = self.meter.create_up_down_counter(
            "active_experiments",
            description="Number of currently active experiments"
        )
        
        self.model_accuracy = self.meter.create_histogram(
            "model_accuracy",
            description="Model prediction accuracy",
            unit="ratio"
        )
    
    @contextmanager
    def track_experiment(self, experiment_id: str, **labels):
        """Context manager to track experiment duration and status."""
        start_time = time.time()
        self.active_experiments.add(1, labels)
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.experiment_duration.record(duration, {
                "experiment_id": experiment_id,
                **labels
            })
            self.active_experiments.add(-1, labels)
    
    def record_free_energy(self, value: float, **labels):
        """Record free energy value."""
        self.free_energy_values.record(value, labels)
    
    def increment_inference_iterations(self, count: int = 1, **labels):
        """Increment inference iteration counter."""
        self.inference_iterations.add(count, labels)
    
    def record_model_accuracy(self, accuracy: float, **labels):
        """Record model accuracy metric."""
        self.model_accuracy.record(accuracy, labels)

# Global metrics instance
metrics_instance = ActiveInferenceMetrics()
```

## Alerting Configuration

### Prometheus Alerting Rules

Create `monitoring/alert-rules.yml`:

```yaml
groups:
  - name: active_inference_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(active_inference_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      # Experiment duration anomaly
      - alert: ExperimentDurationAnomaly
        expr: histogram_quantile(0.95, rate(experiment_duration_seconds_bucket[5m])) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Experiment taking too long"
          description: "95th percentile experiment duration is {{ $value }} seconds"
      
      # Free energy divergence
      - alert: FreeEnergyDivergence
        expr: rate(free_energy_value_sum[5m]) > 1000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Free energy values diverging"
          description: "Free energy sum rate is {{ $value }}"
      
      # Memory usage
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / process_virtual_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
      
      # Service down
      - alert: ServiceDown
        expr: up{job="active-inference-app"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Active inference service has been down for more than 30 seconds"

  - name: performance_alerts
    rules:
      # Inference latency
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 1
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "95th percentile inference latency is {{ $value }} seconds"
      
      # Low accuracy
      - alert: LowModelAccuracy
        expr: avg_over_time(model_accuracy[10m]) < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model accuracy degradation"
          description: "Average model accuracy over 10 minutes is {{ $value }}"
```

## Docker Compose Monitoring Stack

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: otel-collector
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8889:8889"   # Prometheus metrics
    volumes:
      - ./monitoring/otel-config.yaml:/etc/otel-collector-config.yaml
    command: ["--config=/etc/otel-collector-config.yaml"]
    depends_on:
      - prometheus
      - jaeger

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert-rules.yml:/etc/prometheus/alert-rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    depends_on:
      - alertmanager

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  # AlertManager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
```

## Grafana Dashboard Configuration

Create `monitoring/grafana/dashboards/active-inference-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "Active Inference Monitoring",
    "description": "Monitoring dashboard for active inference experiments",
    "tags": ["active-inference", "ai", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Experiment Duration",
        "type": "histogram",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(experiment_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Free Energy Values",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(free_energy_value_sum[5m])",
            "legendFormat": "Free Energy Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Active Experiments",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_experiments",
            "legendFormat": "Active"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Model Accuracy",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(model_accuracy)",
            "legendFormat": "Accuracy"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "30s"
  }
}
```

## Integration Instructions

### 1. Setup Monitoring Stack

```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
```

### 2. Application Instrumentation

Add to your Python application:

```python
from active_inference.monitoring.metrics import metrics_instance
from active_inference.utils.logging_config import setup_logging

# Setup logging
setup_logging()

# Track experiment
with metrics_instance.track_experiment("exp_001", agent_type="variational"):
    # Your experiment code
    free_energy = agent.compute_free_energy()
    metrics_instance.record_free_energy(free_energy)
```

### 3. Environment Variables

```bash
export SERVICE_VERSION=0.1.0
export ENVIRONMENT=development
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=active-inference-sim-lab
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**: Check OTEL collector logs
2. **High memory usage**: Adjust batch processor settings
3. **Missing traces**: Verify OTLP endpoint configuration
4. **Alert not firing**: Check Prometheus rule evaluation

### Performance Tuning

- Adjust sampling rates for high-volume metrics
- Use batch processing for better performance
- Configure appropriate retention policies
- Monitor collector resource usage

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [Jaeger Tracing](https://www.jaegertracing.io/docs/)