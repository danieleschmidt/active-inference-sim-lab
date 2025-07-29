# CI/CD Workflow Setup Guide

## Overview

This guide provides comprehensive documentation for setting up GitHub Actions workflows for the active-inference-sim-lab project. Since GitHub Actions workflows require repository permissions to create, this document provides the exact workflow definitions and setup instructions.

## Required Workflows

### 1. Main CI/CD Pipeline (`ci.yml`)

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: '3.9'
  CMAKE_BUILD_TYPE: Release

jobs:
  lint-and-format:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install pre-commit
        run: pip install pre-commit
        
      - name: Run pre-commit hooks
        run: pre-commit run --all-files

  test-python:
    name: Python Tests
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
          
      - name: Run Python tests
        run: |
          pytest tests/ -v --cov=active_inference --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: python
          name: python-${{ matrix.python-version }}-${{ matrix.os }}

  test-cpp:
    name: C++ Tests
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        build-type: [Release, Debug]
        
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Install dependencies (Ubuntu)
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake build-essential libeigen3-dev
          
      - name: Install dependencies (macOS)
        if: matrix.os == 'macos-latest'
        run: |
          brew install cmake eigen
          
      - name: Build C++ core
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=${{ matrix.build-type }} -DBUILD_TESTS=ON
          make -j$(nproc)
          
      - name: Run C++ tests
        run: |
          cd build && ctest --output-on-failure

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install security tools
        run: |
          pip install bandit safety pip-audit
          
      - name: Run bandit security scan
        run: bandit -r src/ -f json -o bandit-report.json
        
      - name: Run safety vulnerability check
        run: safety check --json --output safety-report.json
        
      - name: Run pip-audit
        run: pip-audit --format=json --output=pip-audit-report.json
        
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            pip-audit-report.json

  build-package:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [lint-and-format, test-python, test-cpp]
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine
          
      - name: Build package
        run: python -m build
        
      - name: Check package
        run: twine check dist/*
        
      - name: Upload package artifacts
        uses: actions/upload-artifact@v3
        with:
          name: python-package
          path: dist/

  publish-pypi:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: build-package
    if: github.event_name == 'release' && github.event.action == 'published'
    environment: pypi
    permissions:
      id-token: write
    steps:
      - name: Download package artifacts
        uses: actions/download-artifact@v3
        with:
          name: python-package
          path: dist/
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

### 2. Security Workflow (`security.yml`)

Create `.github/workflows/security.yml`:

```yaml
name: Security Audit

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Mondays
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  security-audit:
    name: Comprehensive Security Audit
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install security tools
        run: |
          pip install bandit safety pip-audit detect-secrets
          
      - name: Run detect-secrets scan
        run: |
          detect-secrets scan --baseline .secrets.baseline
          
      - name: Run comprehensive security audit
        run: |
          # Bandit security linting
          bandit -r src/ -f json -o bandit-results.json || true
          
          # Safety vulnerability check
          safety check --json --output safety-results.json || true
          
          # Pip audit for known vulnerabilities
          pip-audit --format=json --output=pip-audit-results.json || true
          
      - name: Upload security artifacts
        uses: actions/upload-artifact@v3
        with:
          name: security-audit-results
          path: |
            bandit-results.json
            safety-results.json
            pip-audit-results.json

  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
      
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python', 'cpp' ]
        
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
          
      - name: Build (C++)
        if: matrix.language == 'cpp'
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          make -j$(nproc)
          
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
```

### 3. Documentation Workflow (`docs.yml`)

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths: 
      - 'docs/**'
      - 'README.md'
      - 'src/**/*.py'
  pull_request:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'README.md'
      - 'src/**/*.py'

jobs:
  build-docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install documentation dependencies
        run: |
          pip install -e .[dev]
          pip install sphinx sphinx-rtd-theme myst-parser
          
      - name: Build Sphinx documentation
        run: |
          cd docs && make html
          
      - name: Upload documentation artifacts
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/build/html/

  deploy-docs:
    name: Deploy Documentation
    runs-on: ubuntu-latest
    needs: build-docs
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Download documentation artifacts
        uses: actions/download-artifact@v3
        with:
          name: documentation  
          path: docs/build/html/
          
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/build/html/
```

## Repository Configuration

### Required Secrets

Add these secrets in GitHub repository settings:

```
PYPI_API_TOKEN - For automated PyPI publishing
CODECOV_TOKEN - For code coverage reporting
```

### Branch Protection

Configure branch protection for `main`:

- Require pull request reviews (2 reviewers minimum)
- Require status checks to pass before merging
- Require branches to be up to date
- Restrict pushes to administrators only

### Environment Configuration

Create a `pypi` environment for package publishing:

- Add deployment protection rules
- Configure required reviewers for production deployments
- Add environment-specific secrets

## Integration Setup

### Enable Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "terragon-labs/core-maintainers"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "terragon-labs/devops-team"
```

### CodeQL Configuration

Enable CodeQL scanning in repository settings:

1. Go to Security tab > Code scanning
2. Enable CodeQL analysis
3. Configure for Python and C++ languages
4. Set scan frequency to on-push and weekly schedule

## Monitoring and Alerting

### Status Badges

Add these badges to README.md:

```markdown
[![CI/CD](https://github.com/terragon-labs/active-inference-sim-lab/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/terragon-labs/active-inference-sim-lab/actions)
[![Security](https://github.com/terragon-labs/active-inference-sim-lab/workflows/Security%20Audit/badge.svg)](https://github.com/terragon-labs/active-inference-sim-lab/actions)
[![Docs](https://github.com/terragon-labs/active-inference-sim-lab/workflows/Documentation/badge.svg)](https://github.com/terragon-labs/active-inference-sim-lab/actions)
[![codecov](https://codecov.io/gh/terragon-labs/active-inference-sim-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/terragon-labs/active-inference-sim-lab)
```

## Troubleshooting

### Common Issues

1. **Build failures on Windows**: Ensure Visual Studio Build Tools are available
2. **Permission denied for PyPI**: Verify PYPI_API_TOKEN is correctly configured
3. **CodeQL timeout**: Increase analysis timeout for large codebases
4. **Pre-commit failures**: Run `pre-commit install` locally and fix issues

### Performance Optimization

- Use dependency caching for faster builds
- Parallelize test execution with pytest-xdist
- Use matrix builds for multi-platform testing
- Enable ccache for C++ compilation caching

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Security Best Practices](https://docs.github.com/en/code-security)