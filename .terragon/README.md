# Terragon Autonomous SDLC Value Discovery System

This directory contains the autonomous value discovery and execution system for continuous SDLC enhancement.

## 🎯 System Overview

The Terragon system implements a perpetual value discovery loop that:

1. **Discovers** work items from multiple sources (git history, static analysis, security scans)
2. **Scores** items using hybrid WSJF + ICE + Technical Debt methodology  
3. **Prioritizes** based on adaptive weights for repository maturity level
4. **Executes** highest-value work autonomously with proper validation
5. **Learns** from outcomes to improve future prioritization

## 📁 Files

- **`config.yaml`** - Configuration including scoring weights and thresholds
- **`scoring-engine.py`** - Core value scoring and prioritization logic
- **`autonomous-executor.py`** - Main execution engine with discovery and automation
- **`value-metrics.json`** - Historical metrics and execution tracking
- **`README.md`** - This documentation

## 🚀 Usage

### Discovery Only (Safe)
```bash
cd .terragon
python autonomous-executor.py
```

### Full Autonomous Execution (Requires Review)
```bash
cd .terragon  
python autonomous-executor.py --execute
```

### Manual Scoring Analysis
```bash
cd .terragon
python scoring-engine.py
```

## 📊 Scoring Methodology

### WSJF (Weighted Shortest Job First)
```
WSJF = Cost of Delay / Job Size
Cost of Delay = User Value + Time Criticality + Risk Reduction + Opportunity
```

### ICE (Impact × Confidence × Ease)
```
ICE = Impact Score × Confidence Score × Ease Score
(Each scored 1-10)
```

### Technical Debt Scoring
```
Debt Score = (Debt Impact + Debt Interest) × Hotspot Multiplier
```

### Composite Score
```
Composite = (0.6×WSJF + 0.1×ICE + 0.2×TechDebt + 0.1×Security) × 100
With Security/Compliance boost multipliers applied
```

## 🔍 Discovery Sources

1. **Git History Analysis**
   - TODO/FIXME/HACK comments
   - Commit message patterns
   - File churn vs complexity analysis

2. **Static Analysis**
   - MyPy type checking errors
   - Flake8 code quality issues
   - Bandit security warnings
   - Coverage gaps

3. **Security Scanning**
   - Dependency vulnerability audits
   - SBOM analysis
   - Configuration security review

4. **Performance Monitoring**
   - Benchmark regression detection
   - Profiling hot-spot identification
   - Memory usage analysis

## 🎯 Adaptive Weighting

Scoring weights adapt based on repository maturity:

- **Nascent (0-25%)**: Foundation focus (WSJF 40%, Structure 40%, Quality 20%)
- **Developing (25-50%)**: Growth focus (WSJF 50%, Quality 30%, Tech Debt 20%)  
- **Maturing (50-75%)**: **Current Level** (WSJF 60%, Tech Debt 20%, ICE 10%, Security 10%)
- **Advanced (75%+)**: Optimization focus (WSJF 50%, Tech Debt 30%, Performance 20%)

## 📈 Value Tracking

The system tracks comprehensive metrics:

- **Execution velocity** (items/hour, cycle time)
- **Value delivery** (composite score delivered)
- **Prediction accuracy** (estimated vs actual effort/impact)
- **Learning effectiveness** (scoring model improvements)

## 🔄 Continuous Operation

### Trigger Events
- **PR Merge**: Immediate value discovery and next item selection
- **Hourly**: Security vulnerability scans
- **Daily**: Comprehensive static analysis  
- **Weekly**: Deep architectural assessment
- **Monthly**: Strategic value alignment review

### Safety Mechanisms
- **Test validation** before any changes
- **Rollback procedures** for failed executions
- **Human review** required for breaking changes
- **Risk thresholds** prevent high-risk automatic execution

## 🎓 Learning Loop

The system continuously improves through:

1. **Outcome tracking** - Compare predicted vs actual impact/effort
2. **Weight adjustment** - Adapt scoring based on what delivers value
3. **Pattern recognition** - Learn from similar work item outcomes
4. **Velocity optimization** - Improve cycle time through process refinement

## 🛡️ Security & Compliance

- **Security items** get 2.0x score boost for prioritization
- **Compliance items** get 1.8x score boost
- **Audit trail** maintained for all autonomous actions
- **Change approval** workflow for sensitive modifications

## 📋 Current Status

**Repository Maturity**: Maturing (68/100)  
**Active Items**: 4 discovered, 1 ready for execution  
**Next Item**: Update vulnerable dependencies (Score: 195.2)  
**Autonomous Mode**: Manual oversight required initially  

---

*🤖 Terragon Autonomous SDLC - Perpetual Value Discovery & Delivery*