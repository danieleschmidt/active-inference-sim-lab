# ADR-0001: Use Hybrid C++/Python Architecture

## Status
Accepted

## Context
Active Inference algorithms require intensive mathematical computations including:
- Matrix operations for belief state updates
- Optimization for action planning
- Real-time inference for interactive environments

The framework needs to be both performant and accessible to researchers.

## Decision
We will use a hybrid C++/Python architecture:
- C++ core for performance-critical computations
- Python bindings for user-friendly API
- Pybind11 for seamless integration

## Consequences

### Positive
- Sub-millisecond inference for real-time applications
- Familiar Python interface for researchers
- Memory efficient for large state spaces
- Cross-platform compatibility

### Negative
- Additional build complexity
- Two codebases to maintain
- Potential debugging challenges across language boundaries

## Alternatives Considered
- Pure Python: Too slow for real-time applications
- Pure C++: Less accessible to AI researchers
- JAX/NumPy: Limited control over memory management

## Implementation Notes
- Use modern C++17 features
- Comprehensive Python test coverage
- Memory safety through RAII patterns
- Extensive documentation for both APIs

Date: 2025-01-28