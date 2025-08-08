"""
Command Line Interface for Active Inference Simulation Laboratory.

This module provides a comprehensive CLI for running experiments, training agents,
analyzing results, and managing active inference research workflows.
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rich_print
import numpy as np
import logging

# Import active inference components
from . import ActiveInferenceAgent, MockEnvironment
from .environments import (
    ActiveInferenceGridWorld, 
    ForagingEnvironment,
    SocialDilemmaEnvironment
)
from .utils.logging import setup_logging
from .utils.validation import validate_config

# CLI App
app = typer.Typer(
    name="active-inference",
    help="🧠 Active Inference Simulation Laboratory - Command Line Interface",
    rich_markup_mode="rich"
)
console = Console()

# Global state
current_config = {}
current_agent = None
current_env = None


@app.command()
def info():
    """Display information about Active Inference Simulation Laboratory."""
    console.print("\n🧠 [bold blue]Active Inference Simulation Laboratory[/bold blue]")
    console.print("=" * 60)
    console.print()
    console.print("📋 [bold]Key Features:[/bold]")
    console.print("  • Fast C++ Core with Python bindings")
    console.print("  • Free Energy Principle implementation")
    console.print("  • Belief-based planning and perception")
    console.print("  • Multiple inference methods (Variational, Kalman, Particle)")
    console.print("  • Sophisticated planning algorithms")
    console.print("  • Research-focused environments")
    console.print("  • Multi-agent social cognition")
    console.print()
    console.print("🚀 [bold]Quick Start:[/bold]")
    console.print("  [cyan]active-inference experiment run --config examples/basic.yaml[/cyan]")
    console.print("  [cyan]active-inference agent train --env grid-world --episodes 1000[/cyan]")
    console.print("  [cyan]active-inference benchmark run --comparative[/cyan]")
    console.print()


@app.command()
def create_config(
    output: str = typer.Option("config.yaml", "--output", "-o", help="Output configuration file"),
    template: str = typer.Option("basic", "--template", "-t", help="Configuration template"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive configuration")
):
    """Create a new configuration file for experiments."""
    
    templates = {
        "basic": {
            "experiment": {
                "name": "basic_active_inference",
                "description": "Basic active inference experiment",
                "episodes": 100,
                "max_steps": 200,
                "seed": 42
            },
            "agent": {
                "type": "ActiveInferenceAgent",
                "state_dim": 4,
                "obs_dim": 4,
                "action_dim": 2,
                "inference_method": "variational",
                "planning_horizon": 5,
                "learning_rate": 0.01,
                "temperature": 0.5
            },
            "environment": {
                "type": "MockEnvironment",
                "obs_dim": 4,
                "action_dim": 2,
                "episode_length": 200
            },
            "logging": {
                "level": "INFO",
                "save_trajectories": True,
                "save_beliefs": True,
                "tensorboard": True
            }
        },
        "grid_world": {
            "experiment": {
                "name": "grid_world_exploration",
                "description": "Grid world exploration with active inference",
                "episodes": 500,
                "max_steps": 300,
                "seed": 42
            },
            "agent": {
                "type": "ActiveInferenceAgent",
                "state_dim": 8,
                "obs_dim": 50,  # Flattened grid observation
                "action_dim": 2,
                "inference_method": "variational",
                "planning_horizon": 3,
                "learning_rate": 0.005,
                "temperature": 0.3
            },
            "environment": {
                "type": "ActiveInferenceGridWorld",
                "size": 10,
                "vision_range": 2,
                "observation_noise": 0.1,
                "uncertainty_regions": [[3, 3], [7, 7]],
                "goal_locations": [[8, 8]],
                "dynamic_goals": False
            },
            "logging": {
                "level": "INFO",
                "save_trajectories": True,
                "save_beliefs": True,
                "save_visualizations": True,
                "tensorboard": True
            }
        },
        "social": {
            "experiment": {
                "name": "social_dilemma",
                "description": "Multi-agent social dilemma with theory of mind",
                "episodes": 1000,
                "max_steps": 50,
                "seed": 42
            },
            "agent": {
                "type": "ActiveInferenceAgent",
                "n_agents": 2,
                "state_dim": 6,
                "obs_dim": 12,
                "action_dim": 1,
                "inference_method": "variational",
                "planning_horizon": 3,
                "learning_rate": 0.01,
                "temperature": 0.8
            },
            "environment": {
                "type": "SocialDilemmaEnvironment",
                "n_agents": 2,
                "game_type": "prisoners_dilemma",
                "max_rounds": 20,
                "reputation_system": True,
                "communication": False
            },
            "logging": {
                "level": "INFO",
                "save_trajectories": True,
                "save_beliefs": True,
                "save_social_dynamics": True,
                "tensorboard": True
            }
        }
    }
    
    if template not in templates:
        console.print(f"❌ Unknown template: {template}")
        console.print(f"Available templates: {list(templates.keys())}")
        raise typer.Exit(1)
    
    config = templates[template]
    
    if interactive:
        config = _interactive_config_creation(config)
    
    # Save configuration
    output_path = Path(output)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    console.print(f"✅ Configuration saved to [bold]{output}[/bold]")
    console.print(f"📋 Template: [cyan]{template}[/cyan]")
    
    # Display configuration summary
    _display_config_summary(config)


def _interactive_config_creation(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive configuration creation."""
    console.print("\n🔧 [bold]Interactive Configuration Creation[/bold]")
    
    config = base_config.copy()
    
    # Experiment settings
    console.print("\n📊 [bold]Experiment Settings:[/bold]")
    config["experiment"]["name"] = typer.prompt(
        "Experiment name", default=config["experiment"]["name"]
    )
    config["experiment"]["episodes"] = typer.prompt(
        "Number of episodes", default=config["experiment"]["episodes"], type=int
    )
    config["experiment"]["max_steps"] = typer.prompt(
        "Max steps per episode", default=config["experiment"]["max_steps"], type=int
    )
    
    # Agent settings
    console.print("\n🤖 [bold]Agent Settings:[/bold]")
    config["agent"]["inference_method"] = typer.prompt(
        "Inference method (variational/kalman/particle)", 
        default=config["agent"]["inference_method"]
    )
    config["agent"]["planning_horizon"] = typer.prompt(
        "Planning horizon", default=config["agent"]["planning_horizon"], type=int
    )
    config["agent"]["learning_rate"] = typer.prompt(
        "Learning rate", default=config["agent"]["learning_rate"], type=float
    )
    
    return config


def _display_config_summary(config: Dict[str, Any]):
    """Display configuration summary."""
    table = Table(title="Configuration Summary")
    table.add_column("Section", style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                table.add_row(section, key, str(value))
    
    console.print(table)


# Experiment Commands
experiment_app = typer.Typer(name="experiment")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def run_experiment(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    output_dir: str = typer.Option("results", "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint")
):
    """Run an active inference experiment."""
    
    console.print(f"\n🚀 [bold]Running Active Inference Experiment[/bold]")
    console.print("=" * 50)
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)
    
    with open(config_path) as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            exp_config = yaml.safe_load(f)
        else:
            exp_config = json.load(f)
    
    # Validate configuration
    validation_result = validate_config(exp_config)
    if not validation_result.is_valid:
        console.print("❌ [bold red]Configuration validation failed:[/bold red]")
        for error in validation_result.errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)
    
    # Setup logging
    logger = setup_logging(
        level=exp_config.get("logging", {}).get("level", "INFO"),
        output_dir=output_dir
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to output directory
    config_output = output_path / "experiment_config.yaml"
    with open(config_output, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False, indent=2)
    
    try:
        # Create environment
        env = _create_environment(exp_config["environment"])
        
        # Create agent(s)
        agents = _create_agents(exp_config["agent"], env)
        
        # Run experiment
        results = _run_experiment_loop(
            agents, env, exp_config["experiment"], 
            output_path, verbose, resume
        )
        
        # Save results
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n✅ [bold green]Experiment completed successfully![/bold green]")
        console.print(f"📊 Results saved to: [cyan]{output_path}[/cyan]")
        
        # Display summary
        _display_experiment_summary(results)
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Experiment failed:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _create_environment(env_config: Dict[str, Any]):
    """Create environment from configuration."""
    env_type = env_config["type"]
    
    if env_type == "MockEnvironment":
        return MockEnvironment(**{k: v for k, v in env_config.items() if k != "type"})
    elif env_type == "ActiveInferenceGridWorld":
        return ActiveInferenceGridWorld(**{k: v for k, v in env_config.items() if k != "type"})
    elif env_type == "ForagingEnvironment":
        return ForagingEnvironment(**{k: v for k, v in env_config.items() if k != "type"})
    elif env_type == "SocialDilemmaEnvironment":
        return SocialDilemmaEnvironment(**{k: v for k, v in env_config.items() if k != "type"})
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def _create_agents(agent_config: Dict[str, Any], env):
    """Create agent(s) from configuration."""
    agent_type = agent_config["type"]
    
    if agent_type == "ActiveInferenceAgent":
        # Single agent
        agent_params = {k: v for k, v in agent_config.items() if k not in ["type", "n_agents"]}
        return [ActiveInferenceAgent(**agent_params)]
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def _run_experiment_loop(agents, env, exp_config, output_path, verbose, resume):
    """Main experiment loop."""
    episodes = exp_config["episodes"]
    max_steps = exp_config["max_steps"]
    
    # Initialize results tracking
    results = {
        "episodes": [],
        "statistics": {
            "total_episodes": 0,
            "total_steps": 0,
            "average_reward": 0.0,
            "average_episode_length": 0.0,
            "success_rate": 0.0
        }
    }
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running experiment...", total=episodes)
        
        for episode in range(episodes):
            # Reset environment
            if len(agents) == 1:
                obs = env.reset()
                agents[0].reset(obs)
            else:
                obs_list = env.reset()
                for agent, obs in zip(agents, obs_list):
                    agent.reset(obs)
            
            episode_reward = 0.0
            episode_length = 0
            episode_data = {
                "episode": episode,
                "steps": [],
                "total_reward": 0.0,
                "success": False
            }
            
            # Episode loop
            for step in range(max_steps):
                # Agent action(s)
                if len(agents) == 1:
                    action = agents[0].act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    agents[0].update_model(next_obs, action, reward)
                    
                    episode_reward += reward
                    obs = next_obs
                else:
                    # Multi-agent case
                    actions = [agent.act(obs) for agent, obs in zip(agents, obs_list)]
                    next_obs_list, rewards, terminated, info = env.step(actions)
                    
                    for agent, next_obs, action, reward in zip(agents, next_obs_list, actions, rewards):
                        agent.update_model(next_obs, action, reward)
                    
                    episode_reward += sum(rewards)
                    obs_list = next_obs_list
                
                episode_length += 1
                
                # Save step data
                step_data = {
                    "step": step,
                    "reward": reward if len(agents) == 1 else sum(rewards),
                    "terminated": terminated,
                    "info": info
                }
                episode_data["steps"].append(step_data)
                
                if terminated or truncated:
                    episode_data["success"] = terminated
                    break
            
            episode_data["total_reward"] = episode_reward
            episode_data["length"] = episode_length
            results["episodes"].append(episode_data)
            
            # Update progress
            progress.update(task, advance=1)
            
            if verbose and episode % 100 == 0:
                console.print(f"Episode {episode}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Compute final statistics
    total_episodes = len(results["episodes"])
    total_reward = sum(ep["total_reward"] for ep in results["episodes"])
    total_length = sum(ep["length"] for ep in results["episodes"])
    success_count = sum(1 for ep in results["episodes"] if ep["success"])
    
    results["statistics"].update({
        "total_episodes": total_episodes,
        "total_steps": total_length,
        "average_reward": total_reward / max(1, total_episodes),
        "average_episode_length": total_length / max(1, total_episodes),
        "success_rate": success_count / max(1, total_episodes)
    })
    
    return results


def _display_experiment_summary(results: Dict[str, Any]):
    """Display experiment summary."""
    stats = results["statistics"]
    
    console.print("\n📊 [bold]Experiment Summary[/bold]")
    
    table = Table()
    table.add_column("Metric", style="bold blue")
    table.add_column("Value", style="green")
    
    table.add_row("Total Episodes", str(stats["total_episodes"]))
    table.add_row("Total Steps", str(stats["total_steps"]))
    table.add_row("Average Reward", f"{stats['average_reward']:.3f}")
    table.add_row("Average Episode Length", f"{stats['average_episode_length']:.1f}")
    table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
    
    console.print(table)


# Agent Commands
agent_app = typer.Typer(name="agent")
app.add_typer(agent_app, name="agent")


@agent_app.command("train")
def train_agent(
    env_type: str = typer.Option("mock", "--env", "-e", help="Environment type"),
    episodes: int = typer.Option(1000, "--episodes", "-n", help="Number of training episodes"),
    save_path: str = typer.Option("agent_checkpoint.json", "--save", "-s", help="Save checkpoint path"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Agent configuration file")
):
    """Train an active inference agent."""
    
    console.print(f"\n🎓 [bold]Training Active Inference Agent[/bold]")
    console.print("=" * 45)
    
    # Load agent configuration
    if config:
        with open(config) as f:
            agent_config = yaml.safe_load(f)["agent"]
    else:
        agent_config = {
            "state_dim": 4,
            "obs_dim": 4,
            "action_dim": 2,
            "inference_method": "variational",
            "planning_horizon": 5,
            "learning_rate": 0.01,
            "temperature": 0.5
        }
    
    # Create environment
    if env_type == "mock":
        env = MockEnvironment()
    elif env_type == "grid-world":
        env = ActiveInferenceGridWorld(size=8)
    elif env_type == "foraging":
        env = ForagingEnvironment(size=8, resource_density=0.1)
    else:
        console.print(f"❌ Unknown environment type: {env_type}")
        raise typer.Exit(1)
    
    # Create agent
    agent = ActiveInferenceAgent(**agent_config)
    
    console.print(f"🤖 Agent: {agent}")
    console.print(f"🌍 Environment: {env_type}")
    console.print(f"📚 Episodes: {episodes}")
    
    # Training loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Training agent...", total=episodes)
        
        for episode in range(episodes):
            obs = env.reset()
            agent.reset(obs)
            
            episode_reward = 0.0
            
            while True:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update_model(next_obs, action, reward)
                
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            progress.update(task, advance=1)
            
            if episode % 100 == 0:
                stats = agent.get_statistics()
                console.print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                            f"Free Energy={stats.get('current_free_energy', 0):.3f}")
    
    # Save trained agent
    agent.save_checkpoint(save_path)
    console.print(f"\n✅ [bold green]Training completed![/bold green]")
    console.print(f"💾 Agent saved to: [cyan]{save_path}[/cyan]")
    
    # Display final statistics
    final_stats = agent.get_statistics()
    
    table = Table(title="Training Results")
    table.add_column("Metric", style="bold blue")
    table.add_column("Value", style="green")
    
    for key, value in final_stats.items():
        table.add_row(key, f"{value:.3f}" if isinstance(value, float) else str(value))
    
    console.print(table)


# Benchmark Commands
benchmark_app = typer.Typer(name="benchmark")
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("run")
def run_benchmark(
    comparative: bool = typer.Option(False, "--comparative", "-c", help="Run comparative benchmark"),
    environments: List[str] = typer.Option(["mock"], "--env", "-e", help="Environments to test"),
    output: str = typer.Option("benchmark_results.json", "--output", "-o", help="Output file")
):
    """Run performance benchmarks."""
    
    console.print(f"\n⚡ [bold]Running Active Inference Benchmarks[/bold]")
    console.print("=" * 50)
    
    results = {
        "benchmark_type": "comparative" if comparative else "standard",
        "environments": environments,
        "results": {}
    }
    
    for env_name in environments:
        console.print(f"\n🌍 Testing environment: [cyan]{env_name}[/cyan]")
        
        # Create environment
        if env_name == "mock":
            env = MockEnvironment()
        elif env_name == "grid-world":
            env = ActiveInferenceGridWorld(size=6)
        else:
            console.print(f"⚠️  Unknown environment: {env_name}")
            continue
        
        # Test agent
        agent = ActiveInferenceAgent(
            state_dim=4, obs_dim=env.obs_dim if hasattr(env, 'obs_dim') else 4,
            action_dim=2, planning_horizon=3
        )
        
        # Run benchmark episodes
        episode_rewards = []
        episode_lengths = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"Benchmarking {env_name}..."),
            console=console
        ) as progress:
            
            task = progress.add_task("Running episodes...", total=50)
            
            for episode in range(50):
                obs = env.reset()
                agent.reset(obs)
                
                episode_reward = 0.0
                episode_length = 0
                
                for step in range(200):  # Max 200 steps
                    action = agent.act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    agent.update_model(next_obs, action, reward)
                    
                    episode_reward += reward
                    episode_length += 1
                    obs = next_obs
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                progress.update(task, advance=1)
        
        # Compute statistics
        env_results = {
            "episodes": 50,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "max_reward": float(np.max(episode_rewards)),
            "min_reward": float(np.min(episode_rewards))
        }
        
        results["results"][env_name] = env_results
        
        console.print(f"  📊 Mean Reward: {env_results['mean_reward']:.3f} ± {env_results['std_reward']:.3f}")
        console.print(f"  📏 Mean Length: {env_results['mean_length']:.1f} ± {env_results['std_length']:.1f}")
    
    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n✅ [bold green]Benchmark completed![/bold green]")
    console.print(f"📊 Results saved to: [cyan]{output}[/cyan]")


# Main CLI entry point
def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()