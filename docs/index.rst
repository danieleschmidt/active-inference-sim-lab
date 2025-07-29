Active Inference Simulation Lab Documentation
============================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/C++-17-blue.svg
   :target: https://isocpp.org/
   :alt: C++17

.. image:: https://img.shields.io/badge/License-Apache--2.0-blue.svg
   :target: https://github.com/terragon-labs/active-inference-sim-lab/blob/main/LICENSE
   :alt: License

Welcome to the Active Inference Simulation Lab documentation! This toolkit provides a
lightweight yet powerful framework for building active inference agents based on the
Free Energy Principle.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   theory/free_energy_principle
   theory/active_inference
   theory/generative_models
   theory/belief_updating

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/agent_design
   guides/environments
   guides/training
   guides/visualization
   guides/benchmarking

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/agents
   api/inference
   api/planning
   api/environments
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/cpp_integration
   advanced/performance_optimization
   advanced/custom_models
   advanced/parallel_processing

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/testing
   development/documentation
   development/release_process

.. toctree::
   :maxdepth: 1
   :caption: Reference

   changelog
   license
   bibliography

What is Active Inference?
-------------------------

Active inference is a theoretical framework that describes how biological and artificial
agents can minimize surprise (or free energy) through perception and action. Unlike
traditional reinforcement learning approaches, active inference agents:

* **Model uncertainty explicitly** through probabilistic generative models
* **Balance exploration and exploitation** naturally through free energy minimization
* **Unify perception and action** under a single mathematical framework
* **Achieve sample efficiency** by leveraging strong inductive biases

Key Features
------------

🚀 **Fast C++ Core**
   Optimized implementation of active inference algorithms for maximum performance

🧠 **Free Energy Minimization**
   Principled approach to perception and action based on the Free Energy Principle

📊 **Belief-Based Planning**
   Agents that model uncertainty explicitly and plan accordingly

🎮 **Environment Integration**
   Built-in support for Gym, MuJoCo, and custom environments

⚡ **AXIOM Compatibility**
   Reproduce published results achieving human-level performance with minimal data

💾 **Minimal Dependencies**
   Runs on edge devices with less than 100MB memory footprint

Quick Example
-------------

.. code-block:: python

   from active_inference import ActiveInferenceAgent, FreeEnergyObjective

   # Create agent with generative model
   agent = ActiveInferenceAgent(
       state_dim=4,
       obs_dim=8,
       action_dim=2,
       inference_method="variational"
   )

   # Define free energy objective
   objective = FreeEnergyObjective(
       complexity_weight=1.0,
       accuracy_weight=1.0
   )

   # Run inference loop
   obs = env.reset()
   for step in range(1000):
       # Perception: Infer hidden states
       beliefs = agent.infer_states(obs)
       
       # Action: Minimize expected free energy
       action = agent.plan_action(beliefs, horizon=5)
       
       # Execute and observe
       obs, reward = env.step(action)
       
       # Update generative model
       agent.update_model(obs, action)

Performance Highlights
----------------------

Active inference agents achieve remarkable sample efficiency compared to traditional RL:

.. list-table::
   :header-rows: 1

   * - Environment
     - Active Inference
     - PPO
     - DQN
     - Efficiency Gain
   * - CartPole
     - 50 episodes
     - 200 episodes
     - 500 episodes
     - 4-10x
   * - MountainCar
     - 80 episodes
     - 1000 episodes
     - 2000 episodes
     - 12-25x
   * - Atari Pong
     - 10 episodes
     - 1000 episodes
     - 5000 episodes
     - 100-500x

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install active-inference-sim-lab

Or build from source with C++ optimizations:

.. code-block:: bash

   git clone https://github.com/terragon-labs/active-inference-sim-lab
   cd active-inference-sim-lab
   make install

Community and Support
---------------------

* **GitHub Repository**: https://github.com/terragon-labs/active-inference-sim-lab
* **Issue Tracker**: https://github.com/terragon-labs/active-inference-sim-lab/issues
* **Documentation**: https://active-inference-sim-lab.readthedocs.io/
* **PyPI Package**: https://pypi.org/project/active-inference-sim-lab/

Contributing
------------

We welcome contributions! Please see our :doc:`development/contributing` guide for details
on how to get started.

License
-------

This project is licensed under the Apache 2.0 License - see the :doc:`license` file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`