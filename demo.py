#!/usr/bin/env python3
"""
demo.py — Active Inference agent on a 5-state 1-D Grid World.

The agent starts at state 0 and must reach state 4 (goal) by
minimising Expected Free Energy (EFE) under Karl Friston's
Free Energy Principle.

Run:
    ~/anaconda3/bin/python3 demo.py
"""

import numpy as np

np.random.seed(42)

from active_inference import GenerativeModel, VariationalFreeEnergy, ActiveInferenceAgent
from active_inference.envs import GridWorld1D


# ── Configuration ──────────────────────────────────────────────────────────────
N_STATES   = 5
N_STEPS    = 100
GOAL_STATE = 4
OBS_NOISE  = 0.1
START      = 0


def build_model(env: GridWorld1D) -> GenerativeModel:
    A = env.build_A()
    B = env.build_B()
    D = env.build_D(start=START)
    C = env.build_C(goal=GOAL_STATE)
    return GenerativeModel(A=A, B=B, D=D, C=C)


def run_demo() -> None:
    env   = GridWorld1D(n_states=N_STATES, goal=GOAL_STATE, obs_noise=OBS_NOISE)
    model = build_model(env)
    agent = ActiveInferenceAgent(model, alpha=4.0)

    obs = env.reset(start=START)
    agent.infer_states(obs)

    print("=" * 65)
    print("  Active Inference Agent — 5-State 1-D Grid World")
    print("  Goal: reach state 4 | Start: state 0")
    print("=" * 65)
    header = f"{'Step':>4}  {'TrueState':>9}  {'MAP':>3}  {'Belief (q_s)':>30}  {'F':>7}  {'Action':>6}"
    print(header)
    print("-" * 65)

    action_names = ["←left", "stay·", "right→"]
    steps_to_goal = None

    for step in range(N_STEPS):
        action = agent.select_action()
        obs, done = env.step(action)
        agent.update_beliefs(obs, action)

        F   = agent.free_energy_history[-1]
        q_s = agent.q_s
        map_state = agent.most_likely_state()

        belief_str = " ".join(f"{v:.2f}" for v in q_s)
        print(
            f"{step+1:>4}  {env.state:>9}  {map_state:>3}  {belief_str:>30}  {F:>7.3f}  {action_names[action]:>6}"
        )

        if done and steps_to_goal is None:
            steps_to_goal = step + 1
            print(f"\n  ✓ Goal reached at step {steps_to_goal}!\n")

    print("=" * 65)
    if steps_to_goal:
        print(f"  Agent reached goal in {steps_to_goal} / {N_STEPS} steps.")
    else:
        print(f"  Agent did not reach goal in {N_STEPS} steps.")

    # Summary stats
    mean_F  = np.mean(agent.free_energy_history)
    final_F = agent.free_energy_history[-1]
    right_actions = agent.action_history.count(2)
    left_actions  = agent.action_history.count(0)
    print(f"  Mean free energy : {mean_F:.4f}")
    print(f"  Final free energy: {final_F:.4f}")
    print(f"  Action breakdown : ← {left_actions}  stay {N_STEPS - right_actions - left_actions}  → {right_actions}")
    print("=" * 65)

    # Verify convergence
    final_beliefs = np.stack(agent.belief_history[-10:])
    mean_final_belief_at_goal = float(final_beliefs[:, GOAL_STATE].mean())
    print(f"\n  Mean belief at goal (last 10 steps): {mean_final_belief_at_goal:.4f}")
    if mean_final_belief_at_goal > 0.5:
        print("  ✓ Agent successfully converged toward goal state.")
    else:
        print("  ✗ Convergence unclear — try more steps or tune alpha.")
    print()


if __name__ == "__main__":
    run_demo()
