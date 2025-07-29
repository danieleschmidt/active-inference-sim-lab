# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive SDLC enhancements for repository maturity
- GitHub issue and PR templates for better contribution workflow
- CODEOWNERS file for automated code review assignments
- Security baseline configurations and scanning setup
- Comprehensive CI/CD workflow documentation
- Release automation documentation and templates
- Monitoring and observability configurations

### Changed
- Enhanced pre-commit configuration with additional security checks
- Updated documentation structure for better organization
- Improved security scanning and vulnerability management

### Fixed
- Security baseline establishment for detect-secrets

## [0.1.0] - 2024-01-15

### Added
- Initial release of Active Inference Simulation Lab
- Core C++ implementation of active inference algorithms
- Python bindings for ease of use
- Free Energy Principle-based agent architecture
- Integration with popular RL environments
- MuJoCo physics simulation support
- Comprehensive testing infrastructure
- Docker containerization support
- Documentation and examples

### Features
- Fast C++ core with optimized algorithms
- Free energy minimization for perception and action
- Belief-based planning with uncertainty modeling
- AXIOM compatibility for reproducible results
- Minimal dependencies for edge device deployment
- Multi-platform support (Linux, macOS, Windows)

### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Architecture documentation
- Development and contribution guidelines
- Security policy and procedures

### Infrastructure
- CMake build system for C++ components
- Python packaging with setuptools and pyproject.toml
- Pre-commit hooks for code quality
- Multi-language testing with pytest and Google Test
- Continuous integration setup documentation
- Container support with Docker and docker-compose

---

## Release Process

This project follows semantic versioning:

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in backward-compatible manner  
- **PATCH**: Backward-compatible bug fixes

### Release Types

- **patch**: Bug fixes and minor improvements
- **minor**: New features and enhancements
- **major**: Breaking changes and major updates

To create a release:

1. Update CHANGELOG.md with new version and changes
2. Run `make release-{patch|minor|major}` to create release
3. Push tags and create GitHub release
4. Automated CI/CD will handle PyPI publication

### Version History

All releases are tagged in git and published to:
- [GitHub Releases](https://github.com/terragon-labs/active-inference-sim-lab/releases)
- [PyPI Package](https://pypi.org/project/active-inference-sim-lab/)
- [Docker Hub](https://hub.docker.com/r/terragon-labs/active-inference-sim-lab)