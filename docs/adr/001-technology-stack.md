# ADR-001: Technology Stack Selection

## Status
Accepted

## Context
We need to select core technologies for the active inference simulation lab that balance performance, maintainability, and accessibility.

## Decision
We will use:
- **C++ Core**: For high-performance numerical computation with Eigen for linear algebra
- **Python Bindings**: pybind11 for seamless C++/Python integration
- **Build System**: CMake for C++, setuptools for Python packaging
- **Testing**: GoogleTest (C++) + pytest (Python) for comprehensive testing
- **Dependencies**: Minimal external dependencies to reduce complexity

## Consequences
**Positive:**
- High performance for inference algorithms
- Accessible Python API for researchers
- Industry-standard build tools
- Comprehensive testing coverage

**Negative:**
- Additional complexity from dual-language setup
- Build system maintenance overhead
- Potential version compatibility issues

## Alternatives Considered
- Pure Python with NumPy (rejected: performance concerns)
- Pure C++ (rejected: accessibility issues)
- Julia (rejected: smaller ecosystem)
- Rust (rejected: team expertise considerations)