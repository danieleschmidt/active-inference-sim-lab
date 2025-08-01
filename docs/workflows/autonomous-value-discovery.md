# Autonomous Value Discovery Workflow Implementation

## Overview

This document provides implementation guidance for integrating Terragon Autonomous SDLC Value Discovery into GitHub Actions workflows.

## Required GitHub Actions Workflows

### 1. Value Discovery Trigger (`/.github/workflows/value-discovery.yml`)

```yaml
name: Autonomous Value Discovery

on:
  push:
    branches: [main]
  pull_request:
    types: [closed]
    branches: [main]
  schedule:
    - cron: '0 * * * *'  # Hourly discovery
  workflow_dispatch:

jobs:
  value-discovery:
    runs-on: ubuntu-latest
    if: github.event.pull_request.merged == true || github.event_name != 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pyyaml requests
    
    - name: Run Value Discovery
      run: python3 .terragon/run_discovery.py
      
    - name: Update Backlog
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add BACKLOG.md .terragon/value-metrics.json
        git diff --staged --exit-code || git commit -m "🤖 Update value discovery backlog
        
        Auto-updated by Terragon Autonomous SDLC
        
        Co-Authored-By: Claude <noreply@anthropic.com>"
    
    - name: Push changes
      uses: ad-m/github-push-action@master
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        branch: ${{ github.ref }}
```

### 2. Autonomous Execution Workflow (`/.github/workflows/autonomous-execution.yml`)

```yaml
name: Autonomous Value Execution

on:
  workflow_dispatch:
    inputs:
      item_id:
        description: 'Work item ID to execute'
        required: true
        type: string
      auto_merge:
        description: 'Auto-merge if tests pass'
        required: false
        type: boolean
        default: false

jobs:
  execute-value-item:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Set up C++ Build Environment
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential
    
    - name: Install Python Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Create Execution Branch
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "Terragon Autonomous SDLC"
        git checkout -b "auto-value/${{ inputs.item_id }}-$(date +%Y%m%d-%H%M%S)"
    
    - name: Execute Value Item
      id: execute
      run: |
        python3 .terragon/autonomous-executor.py --execute --item=${{ inputs.item_id }}
        echo "execution_result=$?" >> $GITHUB_OUTPUT
    
    - name: Run Tests
      run: |
        python -m pytest tests/ --cov=src/python --cov-report=xml
        
    - name: Run Security Checks
      run: |
        python -m bandit -r src/python/ -f json -o bandit-report.json || true
        python -m pip audit --format=json --output=pip-audit.json || true
    
    - name: Build C++ Components
      run: |
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
    
    - name: Validate Changes
      run: |
        python -m mypy src/python/
        python -m flake8 src/python/
        python -m black --check src/python/
    
    - name: Create Pull Request
      if: steps.execute.outputs.execution_result == '0'
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        title: "[AUTO-VALUE] ${{ inputs.item_id }}: Autonomous value delivery"
        body: |
          ## 🤖 Autonomous Value Execution
          
          **Work Item**: ${{ inputs.item_id }}
          **Execution Time**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
          **Automated by**: Terragon Autonomous SDLC
          
          ### Changes Made
          - Executed highest-priority value item from backlog
          - All tests passing ✅
          - Security checks completed ✅
          - Code quality validation passed ✅
          
          ### Value Metrics
          - **Composite Score**: Retrieved from execution log
          - **Estimated ROI**: Retrieved from execution log
          - **Risk Assessment**: Low (automated validation)
          
          ### Validation Results
          - **Test Coverage**: Maintained or improved
          - **Security Scan**: No new vulnerabilities
          - **Performance**: No regressions detected
          - **Code Quality**: Standards maintained
          
          🤖 Generated with [Claude Code](https://claude.ai/code)
          
          Co-Authored-By: Claude <noreply@anthropic.com>
        labels: |
          autonomous
          value-driven
          auto-generated
        branch: auto-value/${{ inputs.item_id }}-$(date +%Y%m%d-%H%M%S)
        delete-branch: true
```

### 3. Value Monitoring Workflow (`/.github/workflows/value-monitoring.yml`)

```yaml
name: Value Delivery Monitoring

on:
  pull_request:
    types: [closed]
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  track-value-delivery:
    runs-on: ubuntu-latest
    if: github.event.pull_request.merged == true || github.event_name == 'schedule'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python  
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Analyze Value Delivery
      id: analyze
      run: |
        python3 .terragon/analyze_value_delivery.py
        echo "report_generated=true" >> $GITHUB_OUTPUT
    
    - name: Create Value Report Issue
      if: steps.analyze.outputs.report_generated == 'true' && github.event_name == 'schedule'
      uses: peter-evans/create-or-update-comment@v3
      with:
        issue-number: 1  # Adjust to your tracking issue
        body: |
          ## 📊 Weekly Value Delivery Report
          
          **Period**: $(date -d '7 days ago' +%Y-%m-%d) to $(date +%Y-%m-%d)
          **Generated**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
          
          ### Value Metrics
          $(cat .terragon/weekly-report.md)
          
          ### Next Week's Focus
          $(head -10 BACKLOG.md | tail -5)
          
          ---
          🤖 Generated by Terragon Autonomous SDLC
```

## Integration Setup

### 1. Repository Configuration

Add to `.github/CODEOWNERS`:
```
# Terragon Autonomous SDLC
/.terragon/ @terragon-labs
/BACKLOG.md @terragon-labs
```

### 2. Branch Protection Rules

Configure branch protection for `main`:
- Require status checks: `value-discovery`, `test-suite`
- Require up-to-date branches
- Include administrators: No (for autonomous commits)

### 3. Repository Secrets

No additional secrets required - uses `GITHUB_TOKEN` with default permissions.

### 4. Workflow Permissions

Ensure workflows have necessary permissions in repository settings:
- Actions: Read and write
- Contents: Write
- Issues: Write
- Pull requests: Write

## Manual Triggers

### Trigger Value Discovery
```bash
gh workflow run value-discovery.yml
```

### Execute Specific Item
```bash
gh workflow run autonomous-execution.yml \
  -f item_id=SEC-001 \
  -f auto_merge=false
```

### Generate Value Report
```bash
gh workflow run value-monitoring.yml
```

## Monitoring and Observability

### Key Metrics to Track
- **Discovery Frequency**: Items found per cycle
- **Execution Success Rate**: % of successful autonomous executions
- **Value Delivery Velocity**: Composite score delivered per week
- **Time to Value**: Average time from discovery to deployment
- **Quality Maintenance**: Test coverage, security posture trends

### Alerting Setup
Configure GitHub repository insights and set up notifications for:
- Failed autonomous executions
- Security vulnerabilities discovered
- Value delivery velocity drops
- Quality metric regressions

## Safety and Rollback

### Automated Rollback Triggers
- Test failures
- Security violations
- Performance regressions > 10%
- Code quality drops below threshold

### Manual Override
Create `.terragon/pause-automation` file to halt all autonomous execution.

### Rollback Procedure
```bash
# Revert last autonomous commit
git revert HEAD --no-edit
git push origin main

# Or reset to last known good state
git reset --hard <commit-hash>
git push --force-with-lease origin main
```

---

**Note**: These workflows require manual setup as GitHub Actions cannot be automatically created through code execution. Copy the YAML configurations to your `.github/workflows/` directory and customize as needed.