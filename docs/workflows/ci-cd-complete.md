# Complete CI/CD Pipeline for Active Inference Sim Lab

## Overview

This document provides the complete GitHub Actions workflows needed for a production-ready CI/CD pipeline. These workflows must be manually created in `.github/workflows/` due to security restrictions.

## Required Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.11"
  CMAKE_BUILD_TYPE: Release

jobs:
  test-python:
    name: Python Tests (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
      fail-fast: false

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements-dev.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .[dev]

    - name: Run Python tests
      run: |
        pytest tests/ -v --cov=active_inference --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        flags: python
        name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}

  test-cpp:
    name: C++ Tests (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
      fail-fast: false

    steps:
    - uses: actions/checkout@v4

    - name: Install system dependencies (Ubuntu)
      if: matrix.os == 'ubuntu-latest'
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libeigen3-dev

    - name: Install system dependencies (macOS)
      if: matrix.os == 'macos-latest'
      run: |
        brew install cmake eigen

    - name: Setup MSVC (Windows)
      if: matrix.os == 'windows-latest'
      uses: microsoft/setup-msbuild@v1

    - name: Configure CMake
      run: |
        cmake -B build -DCMAKE_BUILD_TYPE=${{ env.CMAKE_BUILD_TYPE }} -DBUILD_TESTS=ON

    - name: Build C++
      run: cmake --build build

    - name: Run C++ tests
      run: |
        cd build
        ctest --output-on-failure

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read

    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install security tools
      run: |
        pip install bandit safety pip-audit

    - name: Run Bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json || true

    - name: Run Safety dependency check
      run: safety check --json --output safety-report.json || true

    - name: Run pip-audit
      run: pip-audit --desc --format=json --output=audit-report.json || true

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          audit-report.json

  lint-and-format:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt

    - name: Check code formatting (Black)
      run: black --check src/ tests/ examples/ scripts/

    - name: Check import sorting (isort)
      run: isort --check-only src/ tests/ examples/ scripts/

    - name: Run flake8 linting
      run: flake8 src/ tests/ examples/ scripts/

    - name: Run MyPy type checking
      run: mypy src/

    - name: Check C++ formatting
      run: |
        find cpp/ include/ -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | \
        xargs clang-format --dry-run --Werror

  build-and-publish:
    name: Build and Publish
    runs-on: ubuntu-latest
    needs: [test-python, test-cpp, security-scan, lint-and-format]
    if: github.event_name == 'release' && github.event.action == 'published'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install build dependencies
      run: |
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Verify package
      run: twine check dist/*

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*

  docker-build:
    name: Docker Build and Push
    runs-on: ubuntu-latest
    needs: [test-python, test-cpp]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to GitHub Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 2. CodeQL Security Analysis (`.github/workflows/codeql.yml`)

```yaml
name: "CodeQL Advanced Security"

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  analyze:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'cpp', 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        config-file: ./.github/codeql-config.yml

    - name: Setup C++ build environment
      if: matrix.language == 'cpp'
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libeigen3-dev

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
      if: matrix.language == 'python'

    - name: Manual build for C++
      if: matrix.language == 'cpp'
      run: |
        cmake -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"
```

### 3. Performance Benchmarking (`.github/workflows/benchmark.yml`)

```yaml
name: Performance Benchmarks

on:
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    name: Run Benchmarks
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libeigen3-dev
        pip install -r requirements-dev.txt
        pip install -e .[dev,performance]

    - name: Build C++ optimized
      run: |
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
        cmake --build build

    - name: Run benchmarks
      run: |
        cd build && ./run_benchmarks --benchmark_format=json > benchmark_results.json

    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        name: Active Inference Benchmarks
        tool: 'googlecpp'
        output-file-path: build/benchmark_results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '150%'
        fail-on-alert: true

    - name: Upload benchmark artifacts
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: build/benchmark_results.json
```

### 4. Dependency Updates (`.github/workflows/dependency-update.yml`)

```yaml
name: Dependency Security Updates

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    name: Update and Test Dependencies
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install pip-tools
      run: pip install pip-tools

    - name: Update requirements
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in

    - name: Test updated dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .[dev]
        pytest tests/ -x

    - name: Check for security vulnerabilities
      run: |
        pip install safety pip-audit
        safety check
        pip-audit

    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'deps: update dependencies with security patches'
        title: 'Automated dependency security updates'
        body: |
          ## Automated Dependency Updates
          
          This PR contains automated security updates for Python dependencies.
          
          ### Changes
          - Updated all dependencies to latest secure versions
          - Security vulnerabilities addressed
          - All tests passing
          
          Please review and merge if tests pass.
        branch: automated/dependency-updates
        delete-branch: true
```

## Required Manual Setup

### Repository Secrets
Add these secrets in GitHub repository settings:

```
PYPI_API_TOKEN          # For automated package publishing
CODECOV_TOKEN           # For coverage reporting
DOCKER_REGISTRY_TOKEN   # For container registry access
```

### Branch Protection Rules
Configure branch protection for `main`:
- Require pull request reviews (2 reviewers)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Restrict pushes to administrators

### GitHub Security Features
Enable in repository Security tab:
- Dependency graph
- Dependabot alerts
- Dependabot security updates
- Code scanning (CodeQL)
- Secret scanning

## Integration Points

These workflows integrate with existing repository features:
- ✅ Makefile build targets
- ✅ Pre-commit hooks
- ✅ pyproject.toml configuration
- ✅ Docker containers
- ✅ Security policies
- ✅ Issue templates

## Monitoring and Alerts

The workflows provide:
- Performance regression detection
- Security vulnerability alerts
- Build failure notifications
- Coverage reporting
- Automated dependency updates

## Next Steps

1. Create `.github/workflows/` directory
2. Add the workflow files above
3. Configure repository secrets
4. Enable branch protection
5. Test workflows with a small PR
6. Monitor initial runs and adjust as needed

This complete CI/CD setup transforms the repository into a production-ready, enterprise-grade project with comprehensive automation, security, and quality assurance.