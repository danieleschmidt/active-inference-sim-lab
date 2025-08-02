# Frequently Asked Questions (FAQ)

## General Questions

### What is Active Inference?

Active Inference is a framework for understanding perception, action, and learning based on the Free Energy Principle. It provides a principled approach to building intelligent agents that minimize prediction error while exploring their environment.

### How does this differ from reinforcement learning?

Unlike traditional RL which maximizes rewards, Active Inference agents minimize "expected free energy" - a quantity that naturally balances exploration and exploitation through epistemic and pragmatic values.

### What are the main advantages?

- **Sample Efficiency**: Often requires 10-100x less data than traditional RL
- **Principled Exploration**: Natural curiosity-driven behavior
- **Interpretability**: Clear mathematical foundation
- **Uncertainty Quantification**: Built-in uncertainty estimates

## Installation & Setup

### What are the system requirements?

- **OS**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **CPU**: Multi-core processor recommended
- **Optional**: CUDA-compatible GPU for acceleration

### Why is the installation failing?

Common issues:
1. **CMake not found**: Install CMake 3.16+
2. **Compiler issues**: Ensure C++17 compatible compiler
3. **Python version**: Must be 3.9+
4. **Dependencies**: Run `pip install -r requirements.txt` first

### Can I use this without C++?

Yes! The Python-only installation works for most use cases. The C++ core is only needed for maximum performance.

## Usage Questions

### How do I create a simple agent?

```python
from active_inference import ActiveInferenceAgent

agent = ActiveInferenceAgent(
    state_dim=4,
    obs_dim=8,
    action_dim=2
)
```

### How do I integrate with Gym environments?

```python
from active_inference.wrappers import GymWrapper
import gymnasium as gym

env = gym.make("CartPole-v1")
wrapped_env = GymWrapper(env)
agent = ActiveInferenceAgent.from_env(wrapped_env)
```

### Why is training slow?

- **Check hardware**: Use multi-core CPU or GPU if available
- **Reduce model complexity**: Start with smaller state/action dimensions
- **Adjust inference iterations**: Fewer iterations for faster training
- **Use C++ core**: Install from source for maximum performance

### How do I visualize agent behavior?

```python
from active_inference.visualization import BeliefVisualizer

viz = BeliefVisualizer()
viz.animate_beliefs(agent.history)
viz.plot_uncertainty(agent.uncertainty_history)
```

## Development Questions

### How can I contribute?

1. Read our [Contributing Guide](CONTRIBUTING.md)
2. Set up the development environment
3. Find an issue labeled "good first issue"
4. Submit a pull request

### How do I add a new environment?

See our [Custom Environments Guide](guides/advanced/custom-environments.md) for detailed instructions.

### How do I add new inference methods?

1. Implement the inference algorithm in C++
2. Add Python bindings
3. Write comprehensive tests
4. Update documentation

### What's the testing strategy?

We use multiple testing levels:
- **Unit Tests**: Core algorithm correctness
- **Integration Tests**: Component interaction
- **Performance Tests**: Speed and memory benchmarks
- **End-to-End Tests**: Full agent training

## Performance Questions

### How fast should inference be?

Typical performance targets:
- **Simple models**: <1ms per inference step
- **Complex models**: <10ms per inference step
- **Real-time applications**: <100ms end-to-end

### Why is my agent not learning?

Common issues:
1. **Learning rate too high/low**: Try values between 0.001-0.1
2. **Model complexity**: Start simple and increase gradually
3. **Observation noise**: Ensure observations are properly normalized
4. **Environment rewards**: Check reward signal is informative

### How do I optimize for production?

1. **Use C++ core**: Compile from source
2. **Enable optimizations**: Use Release build mode
3. **Profile your code**: Identify bottlenecks
4. **Batch operations**: Process multiple samples together
5. **Consider GPU**: For large-scale models

## Research Questions

### How do I reproduce paper results?

Many papers include reproduction scripts:
```python
from active_inference.benchmarks import reproduce_paper
results = reproduce_paper("smith2024", environment="pong")
```

### How do I cite this work?

```bibtex
@software{active_inference_sim_lab,
  title={Active Inference Simulation Laboratory},
  author={Terragon Labs},
  year={2025},
  url={https://github.com/danieleschmidt/active-inference-sim-lab}
}
```

### Can I use this for commercial applications?

Yes! The Apache 2.0 license allows commercial use. See [LICENSE](LICENSE) for details.

## Troubleshooting

### My agent gets stuck in local minima

Try:
- Increase exploration by adjusting `complexity_weight`
- Use hierarchical agents for complex environments
- Add intrinsic motivation modules
- Adjust initial beliefs to be more uncertain

### Installation fails on Apple Silicon

Use conda for M1/M2 Macs:
```bash
conda create -n active-inference python=3.9
conda activate active-inference
pip install active-inference-sim-lab
```

### ImportError: cannot import name '...'

This usually indicates version mismatch:
```bash
pip install --upgrade active-inference-sim-lab
```

### Out of memory errors

- Reduce batch size
- Use smaller models
- Enable gradient checkpointing
- Consider using CPU instead of GPU for large models

## Getting Help

If your question isn't answered here:

1. **Search Issues**: Check if someone else had the same problem
2. **Open an Issue**: Provide minimal reproducible example
3. **Community Forum**: Join our discussions
4. **Documentation**: Check the comprehensive guides

---

**Still need help?** Open an issue with:
- Your operating system
- Python version
- Full error message
- Minimal code example