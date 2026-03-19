# Active Inference Sim Lab

A minimal, dependency-light Python toolkit implementing active inference agents based on Karl Friston's **Free Energy Principle (FEP)**.  No PyTorch, no TensorFlow — just NumPy and SciPy.

---

## Quick Start

```bash
# Run the demo (5-state grid world, 100 steps)
~/anaconda3/bin/python3 demo.py

# Run tests
~/anaconda3/bin/python3 -m pytest tests/test_fep_core.py -v
```

---

## Theory: Free Energy Principle

### 1. Generative Model

The agent maintains a generative model of its environment encoding the joint distribution:

$$P(o, s) = P(o \mid s) \cdot P(s)$$

Parameters:
- **A** `[n_obs × n_states]` — Likelihood matrix: $A_{os} = P(o \mid s)$
- **B** `[n_states × n_states × n_actions]` — Transition tensor: $B_{s's,a} = P(s' \mid s, a)$
- **D** `[n_states]` — Prior over hidden states: $D_s = P(s)$
- **C** `[n_obs]` — Log-preferences over observations: $C_o = \log \tilde{P}(o)$

### 2. Variational Free Energy

The **variational free energy** $F$ upper-bounds the surprise $-\log P(o)$:

$$F = \underbrace{D_\text{KL}[q(s) \,\|\, P(s)]}_{\text{complexity}} - \underbrace{\mathbb{E}_{q}[\log P(o \mid s)]}_{\text{accuracy}}$$

where $q(s)$ is the agent's posterior belief.  Minimising $F$ drives the agent to hold beliefs that are both accurate (explain observations) and parsimonious (close to prior).

**Variational inference** (mean-field):

$$q(s) = \sigma\!\left(\log A_{o,:} + \log D\right)$$

where $\sigma$ denotes the softmax function.

### 3. Expected Free Energy (Action Selection)

The agent selects actions by minimising **Expected Free Energy** $G(a)$:

$$G(a) = \underbrace{-\sum_o P(o \mid a)\,H[P(s \mid o, a)]}_{\text{epistemic value}} + \underbrace{D_\text{KL}[P(o \mid a) \,\|\, \tilde{P}(o)]}_{\text{pragmatic value}}$$

- **Epistemic value**: expected information gain — the agent is curious and resolves uncertainty.
- **Pragmatic value**: KL divergence from preferred outcomes — the agent is goal-directed.

Action probabilities:

$$P(a) = \sigma(-\alpha \cdot G(a))$$

where $\alpha$ is the precision (inverse temperature).

---

## Package Structure

```
active_inference/
├── __init__.py           # Public API
├── generative_model.py   # GenerativeModel — P(o,s) = P(o|s)*P(s)
├── free_energy.py        # VariationalFreeEnergy — F and G computation
├── agent.py              # ActiveInferenceAgent — infer/act/update loop
└── envs.py               # GridWorld1D — 5-state 1-D demo environment

demo.py                   # End-to-end demo (run me!)
tests/
└── test_fep_core.py      # 27 pytest tests
```

---

## Core API

### `GenerativeModel`

```python
from active_inference import GenerativeModel

model = GenerativeModel(A=A, B=B, D=D, C=C)

# P(o | s)
p = model.likelihood(obs=2, state=3)

# P(s)
p = model.prior(state=1)

# Variational posterior q(s) given observation
q_s = model.infer_states(obs=2)   # shape (n_states,)
```

### `VariationalFreeEnergy`

```python
from active_inference import VariationalFreeEnergy

# Variational free energy F
F = VariationalFreeEnergy.compute(q_s, obs=2, model=model)

# Expected free energy G(a) for action selection
G = VariationalFreeEnergy.expected_free_energy(q_s, action=2, model=model)
```

### `ActiveInferenceAgent`

```python
from active_inference import ActiveInferenceAgent

agent = ActiveInferenceAgent(model, alpha=4.0)

q_s    = agent.infer_states(obs)    # update posterior
action = agent.select_action()      # pick action minimising G
agent.update_beliefs(obs, action)   # shift prior + update posterior
```

---

## Demo: 5-State Grid World

The demo places an agent in a 1-D grid:

```
States:   [0] — [1] — [2] — [3] — [4★]
Actions:  0=← left   1=stay   2=→ right
Goal:     reach state 4
Obs:      noisy state readout (P(o=s|s) = 0.9)
```

The agent navigates from state 0 to state 4 purely by minimising Expected Free Energy — no reward signal, no gradient descent.

Sample output:
```
Step  TrueState  MAP                    Belief (q_s)        F  Action
   1          1    1  0.00 1.00 0.00 0.00 0.00    0.161  right→
   ...
   7          4    4  0.00 0.00 0.00 0.00 1.00    0.169  right→
  ✓ Goal reached at step 7!
  Mean belief at goal (last 10 steps): 0.9643
  ✓ Agent successfully converged toward goal state.
```

---

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.10 (optional, not yet used directly)

---

## References

- Friston, K. J. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11, 127–138.
- Friston, K., et al. (2017). *Active inference and epistemic value.* Cognitive Neuroscience, 8(4), 187–209.
- Da Costa, L., et al. (2020). *Active inference on discrete state-spaces: a synthesis.* Journal of Mathematical Psychology, 99, 102447.
