# VS Code Configuration

This directory contains recommended VS Code configuration files for optimal development experience with the Active Inference Simulation Lab.

## Setup Instructions

1. **Copy configuration files to your workspace:**
   ```bash
   mkdir -p .vscode
   cp docs/vscode-config/settings.json .vscode/
   cp docs/vscode-config/launch.json .vscode/
   cp docs/vscode-config/tasks.json .vscode/
   cp docs/vscode-config/extensions.json .vscode/
   ```

2. **Install recommended extensions:**
   VS Code will automatically prompt you to install the recommended extensions when you open the workspace.

## Configuration Files

### `settings.json`
- Python development with proper linting, formatting, and testing
- C++ development with CMake integration
- Jupyter notebook support
- Documentation tools (Sphinx, reStructuredText)
- Code formatting on save with Black and isort

### `launch.json`
- Debug configurations for Python and C++ components
- Test debugging with pytest integration
- Active inference agent debugging configurations

### `tasks.json`
- Build automation for C++ and Python components
- Test execution with coverage reporting
- Code formatting and linting tasks
- Docker container management
- Documentation building

### `extensions.json`
- Curated list of recommended extensions for optimal development experience
- Python, C++, CMake, Jupyter, and documentation tools
- Code quality and collaboration extensions

## Features Enabled

- **One-click debugging** for Python and C++ components
- **Integrated testing** with coverage reporting
- **Automatic code formatting** with Black and clang-format
- **Comprehensive linting** with flake8, mypy, and C++ tools
- **IntelliSense** for Python and C++ with proper include paths
- **Task automation** for common development operations
- **Jupyter notebook integration** for research and experimentation

## Customization

These configurations provide sensible defaults but can be customized for individual preferences:

- Modify `settings.json` for personal formatting preferences
- Add custom debug configurations in `launch.json`
- Extend `tasks.json` for project-specific automation
- Remove unwanted extensions from `extensions.json`

The configurations are designed to work seamlessly with the DevContainer environment and repository structure.