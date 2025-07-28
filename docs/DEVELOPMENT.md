# Development Guide

## Quick Setup

```bash
# Clone and setup
git clone <repo-url>
cd active-inference-sim-lab
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

## Build Commands

```bash
# Build C++ components
make build-cpp

# Run tests
make test

# Check code quality
make lint
```

## Development Workflow

1. Create feature branch: `git checkout -b feature/name`
2. Make changes following [CONTRIBUTING.md](../CONTRIBUTING.md)
3. Run tests: `make test`
4. Submit PR with clear description

## Additional Resources

• [Contributing Guidelines](../CONTRIBUTING.md)
• [Architecture Documentation](../ARCHITECTURE.md)
• [Security Policy](../SECURITY.md)