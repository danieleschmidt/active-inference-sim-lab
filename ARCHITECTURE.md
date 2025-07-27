# Architecture Document
## Active Inference Simulation Lab

### System Overview
The active inference simulation lab follows a modular architecture with a high-performance C++ core and Python bindings for accessibility. The system implements the Free Energy Principle through variational inference and planning algorithms.

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API Layer                        │
├─────────────────────────────────────────────────────────────┤
│  ActiveInferenceAgent │ Environments │ Visualization │ Utils │
├─────────────────────────────────────────────────────────────┤
│                    Python Bindings (pybind11)             │
├─────────────────────────────────────────────────────────────┤
│                      C++ Core Engine                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Inference   │ │ Planning    │ │ Models      │          │
│  │ Engine      │ │ Engine      │ │ Registry    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│              External Integrations                         │
│     Gym/Gymnasium  │  MuJoCo  │  TensorBoard  │  MLflow   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Inference Engine (C++)
**Responsibility**: Belief updating and state estimation
**Key Classes**:
- `VariationalInference`: Variational message passing
- `ParticleFilter`: Monte Carlo inference
- `KalmanFilter`: Linear Gaussian inference
- `BeliefState`: Probabilistic state representation

#### 2. Planning Engine (C++)
**Responsibility**: Action selection via expected free energy minimization
**Key Classes**:
- `ActivePlanner`: Expected free energy computation
- `PolicyPrior`: Action sequence generation
- `Trajectory`: Path representation and evaluation
- `HorizonPlanner`: Multi-step lookahead

#### 3. Generative Models (C++)
**Responsibility**: World model representation and learning
**Key Classes**:
- `GenerativeModel`: Base model interface
- `LinearGaussianModel`: Simple dynamics
- `DeepGenerativeModel`: Neural network models
- `HierarchicalModel`: Multi-scale models

#### 4. Python API Layer
**Responsibility**: User-friendly interface and integrations
**Key Modules**:
- `active_inference.core`: Main agent classes
- `active_inference.envs`: Environment wrappers
- `active_inference.training`: Training loops and curriculum
- `active_inference.visualization`: Plotting and analysis

### Data Flow

```
Observation → [Inference Engine] → Belief State
                                       ↓
Belief State → [Planning Engine] → Action Policy
                                       ↓
Action Policy → [Environment] → Next Observation
                     ↓
             [Model Learning] ← Transition Data
```

1. **Perception Phase**: Observations update beliefs via variational inference
2. **Planning Phase**: Beliefs drive action selection via expected free energy
3. **Action Phase**: Selected actions execute in environment
4. **Learning Phase**: Transition data updates generative models

### Technology Stack

#### Core Technologies
- **C++ Core**: High-performance numerical computation
- **Python Bindings**: pybind11 for seamless integration
- **Linear Algebra**: Eigen library for matrix operations
- **Optimization**: Custom solvers for variational objectives

#### Development Tools
- **Build System**: CMake for C++, setuptools for Python
- **Testing**: GoogleTest (C++) + pytest (Python)
- **Documentation**: Doxygen (C++) + Sphinx (Python)
- **Packaging**: pip installable with wheels

#### Integration Layer
- **Environments**: OpenAI Gym, MuJoCo, custom environments
- **Visualization**: matplotlib, plotly, tensorboard
- **Logging**: spdlog (C++) + logging (Python)
- **Serialization**: Protocol Buffers for model persistence

### Performance Considerations

#### Memory Management
- Pool allocators for high-frequency objects
- Reference counting for shared belief states
- Lazy evaluation for expensive computations
- Memory mapping for large model parameters

#### Computational Optimization
- Vectorized operations via Eigen/BLAS
- Parallel inference for independent beliefs
- GPU acceleration for neural components
- Just-in-time compilation for custom models

#### Scalability Patterns
- Modular agent composition
- Hierarchical belief decomposition
- Distributed training support
- Streaming data processing

### Security & Safety

#### Input Validation
- Parameter bounds checking
- Model size limitations
- Memory allocation limits
- File system access restrictions

#### Numerical Stability
- Gradient clipping and regularization
- Probability distribution clamping
- Overflow/underflow detection
- Convergence monitoring

### Configuration Management

#### Model Configuration
```yaml
model:
  type: "linear_gaussian"
  state_dim: 4
  observation_dim: 2
  action_dim: 1
  noise_variance: 0.01

inference:
  method: "variational"
  iterations: 10
  convergence_threshold: 1e-6

planning:
  horizon: 5
  trajectory_samples: 100
  exploration_bonus: 0.1
```

#### Environment Configuration
```yaml
environment:
  name: "CartPole-v1"
  max_steps: 500
  observation_noise: 0.0
  reward_scale: 1.0

training:
  episodes: 1000
  evaluation_frequency: 100
  checkpoint_frequency: 500
```

### Deployment Architecture

#### Development Environment
- Local development with hot reloading
- Docker containers for consistent environments
- Jupyter notebooks for experimentation
- VS Code integration with debugging support

#### Production Deployment
- Containerized inference servers
- Model versioning and A/B testing
- Monitoring and alerting infrastructure
- Horizontal scaling for parallel environments

### Extension Points

#### Custom Models
- Plugin architecture for new generative models
- Model registry for dynamic loading
- Serialization interface for model persistence
- Validation framework for model consistency

#### Custom Environments
- Environment wrapper interface
- Observation space transformation
- Reward shaping and curriculum learning
- Multi-agent environment support