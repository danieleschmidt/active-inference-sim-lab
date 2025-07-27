# ADR-002: Inference Algorithm Implementation

## Status
Accepted

## Context
Active inference requires efficient belief updating algorithms. We need to choose which inference methods to prioritize.

## Decision
We will implement three core inference algorithms:
1. **Variational Inference**: Primary method using variational message passing
2. **Particle Filtering**: For non-linear, non-Gaussian models
3. **Kalman Filtering**: For linear Gaussian special cases

All algorithms will share a common `BeliefState` interface for interoperability.

## Consequences
**Positive:**
- Flexibility to handle different model types
- Performance optimization for linear cases
- Research-friendly algorithm comparison
- Modular architecture for easy extension

**Negative:**
- Implementation complexity across multiple algorithms
- Testing burden for algorithm variants
- Memory overhead for maintaining multiple belief representations

## Alternatives Considered
- Single variational approach (rejected: limited model support)
- MCMC methods (rejected: computational cost)
- Ensemble methods (rejected: memory overhead)