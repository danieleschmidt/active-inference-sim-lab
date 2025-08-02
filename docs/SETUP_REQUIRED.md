# Manual Setup Requirements

## Repository Configuration

The following items require manual setup with appropriate permissions:

### ⚠️ CRITICAL: CI/CD Workflow Activation

**HIGHEST PRIORITY - Required for production deployment**

The repository contains comprehensive CI/CD workflow templates in `docs/workflows/templates/` that must be manually copied to activate the CI/CD pipeline:

```bash
# Copy workflow templates to activate CI/CD
mkdir -p .github/workflows
cp docs/workflows/templates/*.yml .github/workflows/

# Files to copy:
# - ci.yml (Primary CI/CD pipeline)
# - security-scan.yml (Security scanning)
# - performance.yml (Performance testing)
# - security.yml (Additional security checks)
# - value-discovery.yml (Autonomous value discovery)
```

**Why Manual Setup is Required:**
- GitHub repository permissions prevent automated workflow creation
- Templates are production-ready and comprehensive
- Includes security scanning, testing, and deployment automation

**Impact of Not Setting Up:**
- No automated testing on pull requests
- No security scanning or vulnerability detection
- No automated deployments or releases
- Manual quality gates instead of automated enforcement

### GitHub Repository Settings

1. **Branch Protection Rules**
   ```
   Branch: main
   • Require pull request reviews (2 reviewers)
   • Require status checks to pass
   • Restrict pushes to admins only
   ```

2. **Repository Secrets**
   ```
   PYPI_API_TOKEN - For automated package publishing
   CODECOV_TOKEN - For coverage reporting integration
   ```

3. **Repository Topics**
   ```
   Topics: active-inference, free-energy-principle, ai, python, cpp
   ```

### External Integrations

1. **Enable Dependabot**
   • Navigate to Security tab > Dependabot
   • Enable vulnerability alerts and security updates

2. **Enable CodeQL Analysis**
   • Go to Security tab > Code scanning
   • Set up CodeQL analysis

3. **Configure Issue Templates**
   • Bug report template (provided in .github/ISSUE_TEMPLATE/)
   • Feature request template
   • Security vulnerability template

### Monitoring Setup

• **Uptime monitoring** for documentation site
• **Performance monitoring** with application metrics
• **Security monitoring** with vulnerability scanning

## Reference Documentation

• [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
• [Repository Security Settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-security-and-analysis-settings-for-your-repository)