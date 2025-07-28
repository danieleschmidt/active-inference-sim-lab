# Workflow Requirements

## Overview

This document outlines the CI/CD workflow requirements for the Active Inference Sim Lab project.

## Required Workflows

### 1. Continuous Integration
- **Purpose**: Run tests, linting, and security checks on every PR
- **Triggers**: Pull requests, pushes to main branch
- **Requirements**: 
  • Python 3.9+ testing across multiple versions
  • C++ compilation and testing
  • Code quality checks (black, flake8, mypy)
  • Security scanning (bandit, safety)

### 2. Documentation Build
- **Purpose**: Build and deploy documentation
- **Triggers**: Pushes to main, documentation changes
- **Requirements**: Sphinx documentation generation

### 3. Release Management
- **Purpose**: Automated versioning and package publishing
- **Triggers**: Release tags
- **Requirements**: PyPI publishing, GitHub releases

### 4. Security Scanning
- **Purpose**: Dependency vulnerability scanning
- **Triggers**: Daily schedule, dependency changes
- **Requirements**: SAST/DAST integration

## Manual Setup Required

The following items require manual GitHub repository configuration:

1. **Repository Settings**:
   • Branch protection rules for main branch
   • Required status checks before merge
   • Dismiss stale reviews when new commits are pushed

2. **Secrets Configuration**:
   • `PYPI_API_TOKEN` for package publishing
   • `CODECOV_TOKEN` for coverage reporting

3. **GitHub Apps Integration**:
   • Dependabot for dependency updates
   • CodeQL for security analysis

## External References

• [GitHub Actions Documentation](https://docs.github.com/en/actions)
• [Python CI/CD Best Practices](https://docs.python.org/3/distributing/)
• [C++ GitHub Actions Examples](https://github.com/marketplace/actions/run-cmake)