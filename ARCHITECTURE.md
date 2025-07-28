# Active Inference Simulation Lab Architecture

## Overview

The Active Inference Simulation Lab is a hybrid Python/C++ framework implementing the Free Energy Principle for building intelligent agents. The architecture follows a layered approach with performance-critical components in C++ and user-friendly interfaces in Python.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Active Inference     │  Environment      │  Visualization  │
│  Agent Classes        │  Wrappers         │  Tools          │
├─────────────────────────────────────────────────────────────┤
│                    C++ Core Engine                         │
├─────────────────────────────────────────────────────────────┤
│  Free Energy         │  Belief           │  Planning       │
│  Computation         │  Updating         │  Algorithms     │
├─────────────────────────────────────────────────────────────┤
│                    External Dependencies                   │
│  NumPy/SciPy         │  MuJoCo          │  Gymnasium      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Free Energy Engine (C++)
- **Location**: `src/cpp/core/`
- **Purpose**: High-performance computation of free energy and its gradients
- **Key Classes**:
  - `FreeEnergyCalculator`: Core free energy computation
  - `GenerativeModel`: Probabilistic model representation
  - `BeliefState`: Efficient belief state management

### 2. Python Bindings
- **Location**: `src/python/active_inference/`
- **Purpose**: User-friendly interface to C++ core
- **Key Modules**:
  - `core.py`: Main agent classes
  - `environments.py`: Environment wrappers
  - `visualization.py`: Plotting and animation tools

### 3. Environment Integration
- **Location**: `src/python/active_inference/envs/`
- **Purpose**: Standardized interfaces to simulation environments
- **Supported**: Gymnasium, MuJoCo, custom environments

## Data Flow

1. **Perception Phase**:
   - Environment provides observations
   - C++ core updates belief states using variational inference
   - Uncertainty estimates computed and cached

2. **Planning Phase**:
   - Expected free energy computed for action sequences
   - Optimization performed in C++ for speed
   - Action selection based on minimum expected free energy

3. **Action Phase**:
   - Selected action executed in environment
   - Model parameters updated based on prediction errors
   - Learning occurs through gradient descent

## Performance Considerations

- **Memory Management**: Smart pointers in C++, automatic memory management in Python
- **Parallelization**: OpenMP for multi-core computation, vectorized operations
- **Caching**: Belief states and model parameters cached for efficiency
- **Precision**: Double precision for numerical stability in free energy computation

## Security Architecture

- **Input Validation**: All environment inputs validated and sanitized
- **Memory Safety**: RAII patterns and smart pointers prevent memory leaks
- **API Security**: Rate limiting and input validation on all public interfaces
- **Dependencies**: Regular security scanning of all dependencies

## Deployment Architecture

### Development Environment
- Docker containers for consistent development
- Pre-commit hooks for code quality
- Automated testing on multiple platforms

### Production Environment
- Containerized deployment with health checks
- Monitoring and observability integration
- Automated scaling based on computational load

## Quality Attributes

- **Performance**: Sub-millisecond inference for simple models
- **Scalability**: Supports models with 1000+ hidden states
- **Reliability**: Comprehensive test coverage with fault tolerance
- **Maintainability**: Clean interfaces, extensive documentation
- **Portability**: Cross-platform support (Linux, macOS, Windows)

## Decision Records

Architecture decisions are documented in `/docs/adr/` following the ADR format.

## Future Considerations

- GPU acceleration for large-scale models
- Distributed computing for multi-agent scenarios
- Real-time constraints for robotics applications
- Integration with hardware accelerators