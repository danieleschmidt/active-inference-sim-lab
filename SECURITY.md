# Security Policy

## Supported Versions

We actively maintain security updates for the following versions of Active Inference Sim Lab:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

### Quick Summary

- **Do not** create public GitHub issues for security vulnerabilities
- Send reports to **security@terragonlabs.com**
- Include detailed reproduction steps and impact assessment
- Expect acknowledgment within 48 hours
- Full response within 7 business days

### Detailed Process

If you discover a security vulnerability in Active Inference Sim Lab, please follow these steps:

1. **Email us privately** at security@terragonlabs.com with:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes (optional)

2. **Do not disclose** the vulnerability publicly until we've had a chance to address it

3. **Wait for our response** - we aim to acknowledge receipt within 48 hours and provide a full response within 7 business days

### What to Include

Please include as much information as possible in your report:

- **Component**: Which part of the system is affected (Python bindings, C++ core, etc.)
- **Severity**: Your assessment of the impact (Critical/High/Medium/Low)
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Environment**: Operating system, Python version, dependencies
- **Proof of Concept**: Code samples or screenshots (if applicable)
- **Suggested Fix**: If you have ideas for remediation

### Security Scope

We're particularly interested in vulnerabilities related to:

- **Memory safety issues** in C++ components
- **Arbitrary code execution** through model files or data
- **Input validation** bypasses leading to crashes or exploits
- **Dependency vulnerabilities** with active exploits
- **Authentication/authorization** bypasses in any components
- **Data exposure** or unauthorized access to sensitive information

### Out of Scope

The following are generally considered out of scope:

- Vulnerabilities in third-party dependencies without active exploits
- Theoretical attacks without practical exploitation paths
- Issues requiring physical access to the machine
- Social engineering attacks
- Denial of service through resource exhaustion (unless exceptionally severe)

### Response Timeline

| Timeline | Action |
|----------|--------|
| 0-48 hours | Acknowledgment of receipt |
| 3-7 days | Initial assessment and triage |
| 7-14 days | Detailed investigation and fix development |
| 14-30 days | Testing, validation, and release preparation |
| 30+ days | Public disclosure coordination |

### Coordinated Disclosure

We follow responsible disclosure practices:

1. **Private coordination** during investigation and fix development
2. **Joint disclosure** timing agreed upon between reporter and maintainers
3. **Credit** given to security researchers in release notes and security advisories
4. **CVE assignment** for qualifying vulnerabilities

### Security Measures

Our security practices include:

#### Development
- **Static analysis** with bandit, semgrep, and CodeQL
- **Dependency scanning** with automated Dependabot updates
- **Memory safety** tools including valgrind and AddressSanitizer
- **Code review** requirements for all changes
- **Signed commits** verification on protected branches

#### CI/CD
- **Container scanning** with Trivy and Snyk
- **SAST/DAST** integration in all pull requests
- **Secrets scanning** to prevent credential leaks
- **Supply chain security** with SLSA attestations

#### Infrastructure
- **Minimal container images** with distroless base images
- **Non-root execution** by default in containers
- **Network segmentation** in deployment environments
- **Regular security updates** for all dependencies

### Security Contact

- **Primary**: security@terragonlabs.com
- **PGP Key**: [Available on request]
- **Response Time**: 48 hours acknowledgment, 7 days full response

### Recognition

We appreciate security researchers who help us maintain the security of Active Inference Sim Lab. Qualifying reports may receive:

- Public recognition in our security advisories
- Credit in release notes
- Swag and other tokens of appreciation
- Potential bounty rewards for critical vulnerabilities (contact us for details)

### Legal Safe Harbor

Terragon Labs supports security research and wants to encourage responsible reporting. We will not pursue legal action against researchers who:

- Follow responsible disclosure practices
- Avoid privacy violations, data destruction, or service disruption
- Report vulnerabilities in good faith
- Do not access more data than necessary to demonstrate the vulnerability

### Additional Resources

- [GitHub Security Advisories](https://github.com/terragon-labs/active-inference-sim-lab/security/advisories)
- [Terragon Labs Security Policy](https://terragonlabs.com/security)
- [Industry Best Practices](https://owasp.org/www-project-top-ten/)

---

*This security policy is regularly reviewed and updated. Last updated: January 2025*