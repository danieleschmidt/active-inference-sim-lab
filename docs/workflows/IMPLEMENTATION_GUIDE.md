# GitHub Actions Implementation Guide

## Required Workflows for Active Inference Sim Lab

This document outlines the GitHub Actions workflows needed for this maturing repository. **Manual setup required** - copy the YAML files from `docs/workflows/templates/` to `.github/workflows/` and customize as needed.

## Core Workflows Required

### 1. Main CI/CD Pipeline (`ci.yml`)
- **Triggers**: Push to main, PRs, scheduled runs
- **Matrix**: Python 3.9-3.12, Ubuntu/Windows/macOS
- **Steps**: Lint → Test → Build → Security scan → Deploy docs
- **Artifacts**: Coverage reports, build artifacts, test results

### 2. Security Scanning (`security.yml`)
- **Triggers**: Push, PR, schedule (daily)
- **Scans**: CodeQL, dependency vulnerabilities, secrets
- **Tools**: Bandit, Safety, Semgrep
- **Reports**: SARIF uploads to Security tab

### 3. Release Automation (`release.yml`)
- **Triggers**: Tag push (v*)
- **Steps**: Build → Test → Package → Publish to PyPI
- **Artifacts**: Wheels, source distribution
- **Notifications**: Slack/Discord integration

### 4. Dependency Management (`dependencies.yml`)
- **Triggers**: Schedule (weekly), manual
- **Tools**: Dependabot, pip-audit
- **Actions**: Auto-update, vulnerability alerts
- **Integration**: With existing dependabot.yml

## Advanced Workflows

### 5. Performance Benchmarking (`benchmarks.yml`)
- **Triggers**: PR, schedule (nightly)
- **Tests**: C++ benchmarks, Python performance
- **Reports**: Performance regression detection
- **Storage**: Benchmark history tracking

### 6. Documentation (`docs.yml`)
- **Triggers**: Push to docs/, main branch
- **Tools**: Sphinx, MkDocs integration
- **Deploy**: GitHub Pages, ReadTheDocs
- **Validation**: Link checking, spell check

### 7. Container Security (`container-security.yml`)
- **Triggers**: Dockerfile changes, schedule
- **Scans**: Trivy, Snyk, Hadolint
- **Images**: Multi-architecture builds
- **Registry**: Push to secure registries

## Implementation Steps

1. **Copy templates**: `cp docs/workflows/templates/*.yml .github/workflows/`
2. **Configure secrets**: Add PyPI tokens, registry credentials
3. **Update branch protection**: Require CI checks to pass
4. **Enable security features**: CodeQL, Dependabot alerts
5. **Test incrementally**: Start with ci.yml, add others gradually

## Required Repository Secrets

```bash
# PyPI publishing
PYPI_API_TOKEN

# Container registries  
DOCKER_HUB_USERNAME
DOCKER_HUB_TOKEN

# Notifications
SLACK_WEBHOOK_URL

# Security scanning
SNYK_TOKEN
```

## Branch Protection Rules

```yaml
Required status checks:
  - CI / test (ubuntu-latest, 3.9)
  - CI / test (ubuntu-latest, 3.12) 
  - Security / codeql
  - Security / dependency-scan

Require review: true
Dismiss stale reviews: true
Require linear history: true
```

For complete implementation details, see individual workflow templates in `docs/workflows/templates/`.