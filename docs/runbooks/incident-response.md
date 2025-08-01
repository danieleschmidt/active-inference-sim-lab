# Incident Response Runbook
## Active Inference Simulation Lab

### Overview
This runbook provides step-by-step procedures for responding to incidents in the Active Inference Simulation Lab system.

### Incident Classification

#### Severity Levels

**P0 - Critical**
- System completely down
- Data loss or corruption
- Security breach
- **SLA**: Response within 15 minutes, resolution within 2 hours

**P1 - High**
- Major functionality impacted
- Performance degradation >50%
- Multiple users affected
- **SLA**: Response within 1 hour, resolution within 8 hours

**P2 - Medium**
- Minor functionality impacted
- Performance degradation <50%
- Few users affected
- **SLA**: Response within 4 hours, resolution within 24 hours

**P3 - Low**
- Cosmetic issues
- Documentation problems
- Single user affected
- **SLA**: Response within 24 hours, resolution within 1 week

### Incident Response Process

#### 1. Detection and Alert
```bash
# Check system status
curl -f https://api.active-inference.com/health || echo "System down"

# Check monitoring dashboards
# - Grafana: https://monitoring.active-inference.com
# - Application logs: kubectl logs -f deployment/active-inference-api
# - Error tracking: Check Sentry dashboard
```

#### 2. Initial Assessment (5 minutes)
- Determine severity level
- Identify affected components
- Estimate user impact
- Document initial findings

#### 3. Escalation Matrix
```
P0/P1: Immediate escalation to on-call engineer
P2: Escalate during business hours
P3: Standard support queue

On-call rotation:
- Primary: DevOps Engineer
- Secondary: Senior Backend Engineer  
- Escalation: Engineering Manager
```

#### 4. Communication Protocol

**Internal Communication (Slack #incidents)**
```
🚨 INCIDENT ALERT 🚨
Severity: P1
Component: Active Inference API
Impact: Inference requests failing
Status: Investigating
Lead: @engineer-name
War room: #incident-YYYY-MM-DD-001
```

**External Communication (Status Page)**
- P0/P1: Immediate status page update
- P2: Update within 2 hours
- P3: No external communication required

### Common Incident Types

#### 1. API Service Down

**Symptoms:**
- Health check failures
- High error rates
- Timeout errors

**Investigation Steps:**
```bash
# Check service status
kubectl get pods -l app=active-inference-api
kubectl describe pod <pod-name>

# Check logs
kubectl logs -f deployment/active-inference-api --tail=100

# Check resource usage
kubectl top pods -l app=active-inference-api

# Check dependencies
curl -f https://database.internal/health
curl -f https://cache.internal/health
```

**Common Resolutions:**
```bash
# Restart deployment
kubectl rollout restart deployment/active-inference-api

# Scale up replicas
kubectl scale deployment/active-inference-api --replicas=5

# Rollback if recent deployment
kubectl rollout undo deployment/active-inference-api
```

#### 2. Database Performance Issues

**Symptoms:**
- Slow query responses
- Connection timeouts
- High CPU/memory usage

**Investigation Steps:**
```bash
# Check database status
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT version();"

# Check active connections
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Check slow queries
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check locks
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Common Resolutions:**
```bash
# Kill long-running queries
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# Restart database (last resort)
kubectl delete pod postgres-0

# Scale up database resources
kubectl patch statefulset postgres -p '{"spec":{"template":{"spec":{"containers":[{"name":"postgres","resources":{"requests":{"memory":"4Gi","cpu":"2000m"}}}]}}}}'
```

#### 3. High Memory Usage / OOM Kills

**Symptoms:**
- Pods being killed
- CrashLoopBackOff status
- Out of memory errors in logs

**Investigation Steps:**
```bash
# Check pod resource usage
kubectl top pods --sort-by=memory

# Check pod events
kubectl describe pod <pod-name> | grep -A 10 Events

# Check memory limits
kubectl get pods <pod-name> -o jsonpath='{.spec.containers[*].resources}'

# Check memory usage over time
# Use Grafana dashboard: Pod Memory Usage
```

**Common Resolutions:**
```bash
# Increase memory limits
kubectl patch deployment active-inference-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi"},"requests":{"memory":"1Gi"}}}]}}}}'

# Scale horizontally
kubectl scale deployment/active-inference-api --replicas=3

# Implement circuit breakers for high-memory operations
```

#### 4. Security Incident

**Symptoms:**
- Unauthorized access attempts
- Suspicious API calls
- Security scanner alerts

**IMMEDIATE ACTIONS:**
```bash
# 1. Isolate affected systems
kubectl scale deployment/active-inference-api --replicas=0

# 2. Preserve evidence
kubectl logs deployment/active-inference-api > incident-logs-$(date +%Y%m%d-%H%M%S).txt

# 3. Check for indicators of compromise
grep -i "unauthorized\|suspicious\|attack" /var/log/nginx/access.log

# 4. Review recent access patterns
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT * FROM audit_log WHERE created_at > now() - interval '24 hours' ORDER BY created_at DESC;"
```

**Escalation:**
- Immediately escalate to Security Team
- Contact legal/compliance if data breach suspected
- Document all actions taken

### Recovery Procedures

#### 1. Service Recovery Checklist
- [ ] Service restored and stable
- [ ] All dependencies functioning
- [ ] Monitoring alerts cleared
- [ ] Performance within normal ranges
- [ ] Affected users notified

#### 2. Data Recovery (if needed)
```bash
# Check backup status
kubectl get backup latest-backup -o yaml

# Restore from backup
kubectl apply -f restore-job.yaml

# Verify data integrity
kubectl exec -it postgres-0 -- psql -U postgres -c "SELECT count(*) FROM critical_table;"
```

#### 3. Load Testing Before Full Recovery
```bash
# Run smoke tests
python tests/smoke/test_basic_functionality.py

# Run load tests with reduced traffic
locust -f tests/load/locustfile.py --users=10 --spawn-rate=1 -t 5m

# Gradually increase traffic
kubectl scale deployment/active-inference-api --replicas=3
# Monitor for 15 minutes
kubectl scale deployment/active-inference-api --replicas=5
```

### Post-Incident Actions

#### 1. Immediate (within 2 hours)
- [ ] Service fully restored
- [ ] Stakeholders notified
- [ ] Initial incident summary documented
- [ ] Monitoring enhanced if needed

#### 2. Short-term (within 24 hours)
- [ ] Detailed incident timeline created
- [ ] Root cause analysis initiated
- [ ] Temporary fixes documented
- [ ] Customer communication sent

#### 3. Long-term (within 1 week)
- [ ] Post-mortem meeting scheduled
- [ ] Action items identified and assigned
- [ ] Process improvements documented
- [ ] Preventive measures implemented

### Incident Documentation Template

```markdown
# Incident Report: YYYY-MM-DD-001

## Summary
Brief description of what happened

## Timeline
- HH:MM - Initial detection
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix implemented
- HH:MM - Service restored

## Root Cause
Detailed explanation of what caused the incident

## Impact
- Duration: X hours Y minutes
- Users affected: ~N users
- Services affected: List of services
- Revenue impact: $X (if applicable)

## Resolution
What was done to resolve the incident

## Action Items
- [ ] Item 1 (Owner: @person, Due: date)
- [ ] Item 2 (Owner: @person, Due: date)

## Lessons Learned
What we learned and how we can prevent similar incidents
```

### Emergency Contacts

**Internal Escalation:**
- DevOps On-call: +1-XXX-XXX-XXXX
- Engineering Manager: +1-XXX-XXX-XXXX
- CTO: +1-XXX-XXX-XXXX

**External Vendors:**
- Cloud Provider Support: Portal + Priority Case
- Database Vendor: +1-XXX-XXX-XXXX
- Security Vendor: +1-XXX-XXX-XXXX

### Useful Commands Reference

```bash
# Kubernetes debugging
kubectl get pods --all-namespaces
kubectl describe pod <pod-name>
kubectl logs -f <pod-name> --previous
kubectl exec -it <pod-name> -- /bin/bash

# Network debugging
nslookup <service-name>
curl -v http://<service-name>:<port>/health
tcpdump -i any host <ip-address>

# System monitoring
top
iostat 1
netstat -tulpn
df -h

# Database debugging
psql -h <host> -U <user> -d <database>
SELECT * FROM pg_stat_activity;
SELECT * FROM pg_locks;
EXPLAIN ANALYZE <query>;
```