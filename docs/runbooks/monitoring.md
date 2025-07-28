# Monitoring and Observability Runbook

## Overview

This runbook provides comprehensive guidance for monitoring Active Inference Sim Lab applications in production environments.

## Monitoring Stack

### Core Components

- **Metrics**: Prometheus + Grafana
- **Logging**: Structured logging with correlation IDs
- **Tracing**: OpenTelemetry integration
- **Health Checks**: Built-in endpoints for service health
- **Alerting**: Prometheus AlertManager + PagerDuty

## Health Check Endpoints

### Application Health
```bash
GET /health
```
**Response**: 200 OK with system status

```json
{
  "status": "healthy",
  "timestamp": "2025-01-28T10:00:00Z",
  "version": "0.1.0",
  "checks": {
    "memory": "ok",
    "disk": "ok",
    "dependencies": "ok"
  }
}
```

### Readiness Check
```bash
GET /ready
```
**Response**: 200 OK when ready to serve traffic

### Liveness Check
```bash
GET /live
```
**Response**: 200 OK when application is running

### Metrics Endpoint
```bash
GET /metrics
```
**Response**: Prometheus-formatted metrics

## Key Metrics to Monitor

### Application Metrics

| Metric | Description | Type | Alerts |
|--------|-------------|------|--------|
| `ai_inference_duration_seconds` | Time taken for inference | Histogram | p95 > 1s |
| `ai_model_accuracy` | Model prediction accuracy | Gauge | < 0.8 |
| `ai_belief_entropy` | Agent belief uncertainty | Gauge | > 2.0 |
| `ai_free_energy` | Free energy computation | Gauge | Sudden spikes |
| `ai_training_loss` | Model training loss | Gauge | Plateau detection |

### System Metrics

| Metric | Description | Alerts |
|--------|-------------|--------|
| `cpu_usage_percent` | CPU utilization | > 80% |
| `memory_usage_bytes` | Memory consumption | > 80% of limit |
| `disk_usage_percent` | Disk space usage | > 85% |
| `network_io_bytes` | Network traffic | Anomaly detection |

### Business Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| `ai_episodes_completed` | Training episodes | Progress tracking |
| `ai_convergence_time` | Time to convergence | Performance optimization |
| `ai_sample_efficiency` | Samples needed for learning | Algorithm comparison |

## Logging Strategy

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARN**: Recoverable errors or unexpected conditions
- **ERROR**: Error conditions that require attention
- **FATAL**: Critical errors causing application shutdown

### Structured Logging Format

```json
{
  "timestamp": "2025-01-28T10:00:00Z",
  "level": "INFO",
  "logger": "active_inference.core",
  "message": "Inference completed successfully",
  "correlation_id": "req-12345",
  "user_id": "user-67890",
  "model_id": "model-abc123",
  "duration_ms": 150,
  "context": {
    "episode": 42,
    "step": 100,
    "accuracy": 0.95
  }
}
```

### Log Aggregation

Logs are collected using:
- **Filebeat** or **Fluent Bit** for log shipping
- **Elasticsearch** for storage and indexing
- **Kibana** for visualization and analysis

## Alert Definitions

### Critical Alerts (P1)

#### Application Down
```yaml
alert: ApplicationDown
expr: up{job="active-inference"} == 0
for: 1m
severity: critical
summary: "Active Inference application is down"
```

#### High Error Rate
```yaml
alert: HighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
for: 2m
severity: critical
summary: "High error rate detected"
```

#### Memory Leak
```yaml
alert: MemoryLeak
expr: increase(process_resident_memory_bytes[1h]) > 100000000
for: 5m
severity: critical
summary: "Potential memory leak detected"
```

### Warning Alerts (P2)

#### High Response Time
```yaml
alert: HighResponseTime
expr: histogram_quantile(0.95, rate(ai_inference_duration_seconds_bucket[5m])) > 1
for: 5m
severity: warning
summary: "95th percentile response time is high"
```

#### Low Model Accuracy
```yaml
alert: LowModelAccuracy
expr: ai_model_accuracy < 0.8
for: 10m
severity: warning
summary: "Model accuracy has dropped below threshold"
```

### Info Alerts (P3)

#### Training Progress
```yaml
alert: TrainingStalled
expr: increase(ai_episodes_completed[1h]) < 10
for: 1h
severity: info
summary: "Training progress appears stalled"
```

## Dashboards

### Main Dashboard

Key widgets:
- Service availability (uptime %)
- Request rate and response times
- Error rate trends
- Resource utilization (CPU, Memory, Disk)
- Active user sessions

### Performance Dashboard

Key widgets:
- Inference latency distribution
- Model accuracy over time
- Free energy convergence
- Training progress metrics
- Sample efficiency trends

### Infrastructure Dashboard

Key widgets:
- Container resource usage
- Network traffic patterns
- Disk I/O metrics
- Database connection pools
- Cache hit rates

## Troubleshooting Guide

### High Memory Usage

1. **Check for memory leaks**:
   ```bash
   # Monitor memory growth
   watch -n 1 'ps aux | grep active-inference'
   
   # Check heap usage
   python -m memory_profiler app.py
   ```

2. **Analyze memory allocation**:
   ```python
   import tracemalloc
   tracemalloc.start()
   # Run problematic code
   snapshot = tracemalloc.take_snapshot()
   ```

3. **Reduce memory footprint**:
   - Implement batch processing
   - Clear unnecessary caches
   - Optimize model size

### Slow Inference

1. **Profile inference pipeline**:
   ```python
   import cProfile
   cProfile.run('agent.infer(observations)')
   ```

2. **Check resource constraints**:
   - CPU utilization
   - Memory bandwidth
   - I/O bottlenecks

3. **Optimization strategies**:
   - Model quantization
   - Parallel processing
   - Caching frequently used computations

### Training Issues

1. **Monitor convergence**:
   ```python
   # Track loss and accuracy trends
   tensorboard --logdir=./logs
   ```

2. **Debug divergence**:
   - Check learning rate
   - Validate gradient computation
   - Monitor numerical stability

## Incident Response

### Severity Levels

- **P1 (Critical)**: Complete service outage
- **P2 (High)**: Significant performance degradation
- **P3 (Medium)**: Minor issues, workarounds available
- **P4 (Low)**: Documentation, enhancement requests

### Response Procedures

#### P1 Incidents

1. **Immediate response** (within 5 minutes):
   - Acknowledge alert
   - Assess impact scope
   - Implement quick fixes or rollback

2. **Investigation** (within 15 minutes):
   - Check recent deployments
   - Review error logs
   - Identify root cause

3. **Resolution** (within 1 hour):
   - Apply permanent fix
   - Verify service restoration
   - Update monitoring

4. **Post-incident** (within 24 hours):
   - Conduct post-mortem
   - Document lessons learned
   - Implement preventive measures

## Performance Baselines

### Expected Performance

| Metric | Target | Acceptable | Critical |
|--------|---------|------------|----------|
| Inference latency (p95) | < 100ms | < 500ms | > 1s |
| Model accuracy | > 0.95 | > 0.8 | < 0.7 |
| Memory usage | < 50% | < 80% | > 90% |
| CPU utilization | < 60% | < 80% | > 90% |
| Error rate | < 0.1% | < 1% | > 5% |

### Capacity Planning

- **Scale up** when CPU > 70% for 5+ minutes
- **Scale out** when request queue > 100
- **Preemptive scaling** during known high-traffic periods

## Contact Information

### On-Call Rotation

- **Primary**: @oncall-primary
- **Secondary**: @oncall-secondary
- **Escalation**: @engineering-manager

### Communication Channels

- **Incidents**: #incidents
- **Monitoring**: #monitoring-alerts
- **General**: #active-inference-team

### External Contacts

- **Cloud Provider**: [Support link]
- **Third-party Services**: [Contact details]
- **Security Team**: security@terragonlabs.com

---

*This runbook is updated monthly and during post-incident reviews.*