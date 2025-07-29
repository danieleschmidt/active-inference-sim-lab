# Advanced CI/CD Workflows

This document provides comprehensive GitHub Actions workflow configurations for the Active Inference Sim Lab project.

## Security-Enhanced CI Pipeline

### Main CI Workflow (.github/workflows/ci.yml)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  CMAKE_BUILD_TYPE: Release

jobs:
  security-scan:
    runs-on: ubuntu-latest
    name: Security Scanning
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Run GitGuardian scan
        uses: GitGuardian/ggshield-action@v1.25.0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITGUARDIAN_API_KEY: ${{ secrets.GITGUARDIAN_API_KEY }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install security tools
        run: |
          pip install bandit safety semgrep
      
      - name: Run Bandit security scan
        run: bandit -r src/ -f json -o bandit-report.json
      
      - name: Run Safety vulnerability check
        run: safety check --json --output safety-report.json
      
      - name: Run Semgrep SAST
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
        run: semgrep --config=auto src/

  code-quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements-dev.txt') }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Run pre-commit hooks
        uses: pre-commit/action@v3.0.0
      
      - name: Run tests with coverage
        run: |
          pytest tests/ --cov=active_inference --cov-report=xml --cov-report=term
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  cpp-build-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        build_type: [Release, Debug]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup CMake
        uses: jwlawson/actions-setup-cmake@v1.14
      
      - name: Install dependencies (Ubuntu)
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y libboost-all-dev libeigen3-dev
      
      - name: Configure CMake
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
                -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON
      
      - name: Build
        run: cmake --build build --config ${{ matrix.build_type }}
      
      - name: Test
        working-directory: build
        run: ctest --output-on-failure

  docker-build:
    runs-on: ubuntu-latest
    needs: [security-scan, code-quality]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Security Scanning Workflow (.github/workflows/security.yml)

```yaml
name: Security Analysis

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  codeql:
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
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  dependency-review:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
      
      - name: Dependency Review
        uses: actions/dependency-review-action@v3

  sbom-generation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate SBOM
        uses: anchore/sbom-action@v0.14.3
        with:
          path: .
          format: spdx-json
      
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: ./*.spdx.json
```

### Performance Benchmarking (.github/workflows/benchmark.yml)

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * 0'  # Weekly Sunday 4 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install pytest-benchmark asv
      
      - name: Run Python benchmarks
        run: |
          pytest tests/benchmark/ --benchmark-json=benchmark-results.json
      
      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark-results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '120%'
          comment-on-alert: true
```

## Manual Setup Requirements

### Repository Settings

1. **Branch Protection Rules**:
   ```
   Branch: main
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require up-to-date branches before merging
   - Require signed commits
   - Include administrators
   ```

2. **Security Settings**:
   ```
   - Enable Dependabot alerts
   - Enable Dependabot security updates
   - Enable private vulnerability reporting
   - Enable secret scanning
   - Enable push protection for secrets
   ```

### Required Secrets

Add these secrets in repository settings:

```bash
# Security scanning
GITGUARDIAN_API_KEY=your_gitguardian_token
SEMGREP_APP_TOKEN=your_semgrep_token

# Package publishing
PYPI_API_TOKEN=your_pypi_token

# Code coverage
CODECOV_TOKEN=your_codecov_token

# Container registry (automatically provided)
GITHUB_TOKEN=automatically_provided
```

### Branch Protection Setup

```bash
# Using GitHub CLI
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["security-scan","code-quality","cpp-build-test"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null
```

### Integration with External Services

1. **Codecov Integration**:
   - Sign up at codecov.io
   - Install Codecov GitHub App
   - Add `CODECOV_TOKEN` to repository secrets

2. **SonarCloud Integration**:
   - Connect repository to SonarCloud
   - Add `SONAR_TOKEN` to repository secrets
   - Configure sonar-project.properties

3. **Dependabot Configuration**:
   - Already configured in `.github/dependabot.yml`
   - Automatically creates PRs for dependency updates

## Monitoring and Observability

### OpenTelemetry Integration

```yaml
# Add to workflow for observability
- name: Setup OpenTelemetry
  env:
    OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.OTEL_ENDPOINT }}
    OTEL_EXPORTER_OTLP_HEADERS: "authorization=Bearer ${{ secrets.OTEL_TOKEN }}"
  run: |
    export OTEL_SERVICE_NAME="active-inference-ci"
    export OTEL_SERVICE_VERSION="${{ github.sha }}"
    # Run instrumented tests
```

### Metrics Collection

The workflows automatically collect:
- Test execution times
- Code coverage metrics
- Security scan results
- Build performance data
- Dependency vulnerability counts

## Rollback Procedures

### Failed Deployment Rollback

```bash
# Rollback to previous version
git revert <commit-hash>
git push origin main

# Or rollback Docker image
docker pull ghcr.io/repo:previous-tag
```

### Security Issue Response

1. Immediate containment via branch protection
2. Automated security scanning halt
3. Incident response team notification
4. Coordinated vulnerability disclosure process

## Success Metrics

Track these KPIs:
- Security scan pass rate: >95%
- Test coverage: >90%
- Build success rate: >98%
- Deployment frequency: Daily
- Lead time for changes: <2 hours
- Mean time to recovery: <30 minutes