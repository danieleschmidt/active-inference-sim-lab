# Contributing to Active Inference Sim Lab

Welcome! We're excited that you're interested in contributing to the Active Inference Simulation Laboratory. This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Ways to Contribute

- **Bug Reports**: Help us identify and fix bugs
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes, new features, or improvements
- **Documentation**: Improve our documentation
- **Examples**: Add examples and tutorials
- **Testing**: Help improve test coverage
- **Reviews**: Review pull requests from other contributors

### Before You Start

1. Check if there's already an issue for what you want to work on
2. If not, create an issue to discuss your proposed changes
3. Wait for feedback before starting significant work
4. Fork the repository and create a feature branch

## Development Setup

### Prerequisites

- Python 3.9+ 
- C++17 compatible compiler (GCC 8+, Clang 9+, or MSVC 2019+)
- CMake 3.18+
- Git

### Local Development

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/active-inference-sim-lab.git
   cd active-inference-sim-lab
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Build the Project**
   ```bash
   # Build C++ components
   make build-cpp
   
   # Install Python package in development mode
   pip install -e .
   ```

4. **Verify Setup**
   ```bash
   # Run tests
   make test
   
   # Check code quality
   make lint
   ```

### Using DevContainers

For a consistent development environment, you can use the provided DevContainer:

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open the project in VS Code
4. When prompted, click "Reopen in Container"

## Contributing Guidelines

### Code Style

We follow these coding standards:

**Python:**
- [PEP 8](https://pep8.org/) with line length of 88 characters
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where possible

**C++:**
- Google C++ Style Guide with modifications
- Use [clang-format](.clang-format) for formatting
- Use modern C++17 features

**General:**
- Write clear, descriptive commit messages
- Keep changes focused and atomic
- Include tests for new functionality
- Update documentation as needed

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(core): add belief-based planning algorithm

fix(inference): correct KL divergence calculation in variational update

docs(api): update free energy computation examples

test(integration): add end-to-end agent training tests
```

### Branch Naming

Use descriptive branch names:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

## Pull Request Process

### Before Creating a PR

1. **Update your fork**
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Creating the PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### PR Template

Please include the following in your PR description:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented hard-to-understand areas
- [ ] Updated documentation
- [ ] No breaking changes (or justified)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, a maintainer will merge

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run Python tests only
pytest tests/

# Run C++ tests only
make test-cpp

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Writing Tests

**Test Categories:**
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Benchmark Tests**: Performance and regression testing
- **End-to-End Tests**: Complete workflow testing

**Guidelines:**
- Write tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names
- Include edge cases and error conditions
- Mock external dependencies

**Example Test:**
```python
def test_free_energy_computation():
    """Test free energy computation with known values."""
    # Arrange
    observation = np.array([1.0, 2.0])
    prediction = np.array([1.1, 1.9])
    
    # Act
    free_energy = compute_free_energy(observation, prediction)
    
    # Assert
    expected = 0.01  # 0.5 * (0.1^2 + 0.1^2)
    assert np.isclose(free_energy, expected)
```

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guide**: How to use the library
3. **Developer Guide**: Development information
4. **Examples**: Practical usage examples
5. **Architecture**: System design documentation

### Writing Documentation

**Docstrings:**
```python
def compute_free_energy(observation: np.ndarray, prediction: np.ndarray) -> float:
    """Compute free energy between observation and prediction.
    
    Args:
        observation: Actual observation vector
        prediction: Predicted observation vector
        
    Returns:
        Free energy value (non-negative)
        
    Raises:
        ValueError: If arrays have different shapes
        
    Example:
        >>> obs = np.array([1.0, 2.0])
        >>> pred = np.array([1.1, 1.9])
        >>> fe = compute_free_energy(obs, pred)
        >>> print(f"Free energy: {fe:.3f}")
        Free energy: 0.010
    """
```

**Markdown Documentation:**
- Use clear headings and structure
- Include code examples
- Link to related sections
- Keep examples current and working

### Building Documentation

```bash
# Build documentation
make docs

# View documentation
open docs/build/html/index.html
```

## Community

### Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Discussions**: General questions and community interaction
- **Discord**: Real-time chat (link in README)
- **Email**: For security issues or private concerns

### Communication Guidelines

- Be respectful and inclusive
- Provide context and details
- Search existing issues before creating new ones
- Use appropriate tags and labels
- Follow up on your issues and PRs

### Recognition

Contributors are recognized in several ways:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- GitHub contributor stats
- Special recognition for significant contributions

## Development Workflow

### Issue Lifecycle

1. **Created**: Issue reported or feature requested
2. **Triaged**: Labeled and prioritized by maintainers
3. **Assigned**: Taken by contributor
4. **In Progress**: Work is ongoing
5. **Review**: Pull request submitted
6. **Closed**: Merged or resolved

### Release Process

1. **Feature Freeze**: No new features
2. **Testing**: Comprehensive testing phase
3. **Documentation**: Update docs and changelog
4. **Release**: Tag and publish new version
5. **Post-Release**: Monitor for issues

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Questions?

If you have questions about contributing:

1. Check this document and existing issues
2. Create a new issue with the "question" label
3. Join our community discussions
4. Reach out to maintainers

Thank you for contributing to Active Inference Sim Lab! 🚀