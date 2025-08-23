"""
Multi-agent orchestration system for scalable Active Inference.
"""

import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import logging
from enum import Enum


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    LEARNING = "learning"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentMetrics:
    """Metrics for an individual agent."""
    agent_id: str
    status: AgentStatus
    total_steps: int
    total_episodes: int
    avg_reward: float
    avg_free_energy: float
    last_update: float
    error_count: int
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass
class CoordinationMessage:
    """Message for inter-agent coordination."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    priority: int = 0


class MultiAgentOrchestrator:
    """
    Advanced multi-agent orchestration system for scalable Active Inference.
    
    Features:
    - Dynamic agent lifecycle management
    - Load balancing and resource allocation
    - Inter-agent communication and coordination
    - Hierarchical multi-agent structures
    - Performance monitoring and optimization
    """
    
    def __init__(self,
                 max_agents: int = 100,
                 enable_communication: bool = True,
                 enable_load_balancing: bool = True,
                 resource_monitoring_interval: float = 5.0):
        """
        Initialize multi-agent orchestrator.
        
        Args:
            max_agents: Maximum number of concurrent agents
            enable_communication: Enable inter-agent communication
            enable_load_balancing: Enable automatic load balancing
            resource_monitoring_interval: Seconds between resource monitoring
        """
        self.max_agents = max_agents
        self.enable_communication = enable_communication
        self.enable_load_balancing = enable_load_balancing
        self.resource_monitoring_interval = resource_monitoring_interval
        
        # Agent management
        self.agents: Dict[str, Any] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.agent_executors: Dict[str, ThreadPoolExecutor] = {}
        
        # Communication system
        self.message_queue: deque = deque()
        self.communication_handlers: Dict[str, Callable] = {}
        
        # Load balancing
        self.load_balancer_active = False
        self.resource_usage: Dict[str, float] = defaultdict(float)
        
        # Coordination and hierarchy
        self.agent_groups: Dict[str, Set[str]] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
        
        # Threading and monitoring
        self.orchestrator_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        
        # Performance tracking
        self.total_agents_created = 0
        self.total_messages_processed = 0
        self.coordination_events = 0
        
        # Logging
        self.logger = logging.getLogger("MultiAgentOrchestrator")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"MultiAgentOrchestrator initialized: max_agents={max_agents}")
    
    def register_agent(self,
                      agent_id: str,
                      agent: Any,
                      group: Optional[str] = None,
                      max_workers: int = 1) -> bool:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: The agent instance
            group: Optional group assignment
            max_workers: Maximum worker threads for this agent
            
        Returns:
            True if registration successful
        """
        with self.lock:
            if agent_id in self.agents:
                self.logger.warning(f"Agent {agent_id} already registered")
                return False
            
            if len(self.agents) >= self.max_agents:
                self.logger.error(f"Maximum agents limit reached: {self.max_agents}")
                return False
            
            # Register agent
            self.agents[agent_id] = agent
            self.agent_executors[agent_id] = ThreadPoolExecutor(max_workers=max_workers)
            
            # Initialize metrics
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                status=AgentStatus.IDLE,
                total_steps=0,
                total_episodes=0,
                avg_reward=0.0,
                avg_free_energy=0.0,
                last_update=time.time(),
                error_count=0
            )
            
            # Add to group if specified
            if group:
                if group not in self.agent_groups:
                    self.agent_groups[group] = set()
                self.agent_groups[group].add(agent_id)
            
            self.total_agents_created += 1
            self.logger.info(f"Agent {agent_id} registered successfully (group: {group})")
            
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the orchestrator.
        
        Args:
            agent_id: Agent to unregister
            
        Returns:
            True if unregistration successful
        """
        with self.lock:
            if agent_id not in self.agents:
                self.logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
            
            # Shutdown executor
            if agent_id in self.agent_executors:
                self.agent_executors[agent_id].shutdown(wait=True)
                del self.agent_executors[agent_id]
            
            # Remove from groups
            for group, agent_set in self.agent_groups.items():
                agent_set.discard(agent_id)
            
            # Clean up
            del self.agents[agent_id]
            del self.agent_metrics[agent_id]
            
            self.logger.info(f"Agent {agent_id} unregistered successfully")
            return True
    
    def start_orchestration(self):
        """Start the orchestration system."""
        if self.orchestrator_thread and self.orchestrator_thread.is_alive():
            self.logger.warning("Orchestration already running")
            return
        
        self.stop_event.clear()
        
        # Start orchestrator thread
        self.orchestrator_thread = threading.Thread(
            target=self._orchestration_loop,
            daemon=True,
            name="Orchestrator"
        )
        self.orchestrator_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="AgentMonitoring"
        )
        self.monitoring_thread.start()
        
        self.logger.info("Multi-agent orchestration started")
    
    def stop_orchestration(self):
        """Stop the orchestration system."""
        self.stop_event.set()
        
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=10.0)
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        # Shutdown all agent executors
        with self.lock:
            for executor in self.agent_executors.values():
                executor.shutdown(wait=True)
        
        self.logger.info("Multi-agent orchestration stopped")
    
    def execute_parallel_episodes(self,
                                 environment_factory: Callable,
                                 num_episodes: int,
                                 episode_length: int = 1000) -> Dict[str, Any]:
        """
        Execute episodes in parallel across multiple agents.
        
        Args:
            environment_factory: Factory function to create environment instances
            num_episodes: Number of episodes per agent
            episode_length: Maximum steps per episode
            
        Returns:
            Aggregated results across all agents
        """
        if not self.agents:
            raise ValueError("No agents registered")
        
        start_time = time.time()
        all_results = {}
        
        # Submit episodes to agents
        future_to_agent = {}
        with self.lock:
            for agent_id, agent in self.agents.items():
                if self.agent_metrics[agent_id].status == AgentStatus.FAILED:
                    continue
                
                executor = self.agent_executors[agent_id]
                future = executor.submit(
                    self._run_agent_episodes,
                    agent_id, agent, environment_factory, num_episodes, episode_length
                )
                future_to_agent[future] = agent_id
        
        # Collect results
        completed_agents = 0
        for future in as_completed(future_to_agent.keys()):
            agent_id = future_to_agent[future]
            try:
                agent_results = future.result()
                all_results[agent_id] = agent_results
                completed_agents += 1
                
                # Update metrics
                self._update_agent_metrics_from_results(agent_id, agent_results)
                
            except Exception as e:
                self.logger.error(f"Agent {agent_id} episode execution failed: {e}")
                with self.lock:
                    self.agent_metrics[agent_id].status = AgentStatus.FAILED
                    self.agent_metrics[agent_id].error_count += 1
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        aggregated_results.update({
            'execution_time': total_time,
            'completed_agents': completed_agents,
            'total_agents': len(self.agents),
            'success_rate': completed_agents / len(self.agents) * 100 if self.agents else 0
        })
        
        self.logger.info(f"Parallel episode execution completed: {completed_agents}/{len(self.agents)} "
                        f"agents succeeded in {total_time:.2f}s")
        
        return aggregated_results
    
    def coordinate_multi_agent_learning(self,
                                      learning_task: str,
                                      coordination_frequency: int = 100) -> Dict[str, Any]:
        """
        Coordinate learning across multiple agents with information sharing.
        
        Args:
            learning_task: Type of learning coordination
            coordination_frequency: Steps between coordination events
            
        Returns:
            Coordination results and metrics
        """
        if not self.enable_communication:
            raise ValueError("Communication must be enabled for coordination")
        
        coordination_results = {
            'learning_task': learning_task,
            'coordination_events': 0,
            'messages_exchanged': 0,
            'consensus_reached': False,
            'performance_improvement': {}
        }
        
        # Implement coordination logic based on task type
        if learning_task == "shared_experience":
            coordination_results.update(
                self._coordinate_experience_sharing(coordination_frequency)
            )
        elif learning_task == "consensus_building":
            coordination_results.update(
                self._coordinate_consensus_building()
            )
        elif learning_task == "hierarchical_learning":
            coordination_results.update(
                self._coordinate_hierarchical_learning()
            )
        
        return coordination_results
    
    def send_message(self, message: CoordinationMessage) -> bool:
        """Send a message through the communication system."""
        if not self.enable_communication:
            return False
        
        self.message_queue.append(message)
        self.total_messages_processed += 1
        
        self.logger.debug(f"Message queued: {message.sender_id} -> {message.receiver_id}")
        return True
    
    def broadcast_message(self,
                         sender_id: str,
                         message_type: str,
                         content: Dict[str, Any],
                         exclude_sender: bool = True) -> int:
        """
        Broadcast a message to all agents.
        
        Args:
            sender_id: ID of sending agent
            message_type: Type of message
            content: Message content
            exclude_sender: Whether to exclude sender from broadcast
            
        Returns:
            Number of agents that received the message
        """
        if not self.enable_communication:
            return 0
        
        recipients = 0
        with self.lock:
            for agent_id in self.agents:
                if exclude_sender and agent_id == sender_id:
                    continue
                
                message = CoordinationMessage(
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    message_type=message_type,
                    content=content,
                    timestamp=time.time()
                )
                
                self.message_queue.append(message)
                recipients += 1
        
        self.total_messages_processed += recipients
        self.logger.debug(f"Broadcast from {sender_id}: {recipients} recipients")
        
        return recipients
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        with self.lock:
            # Agent status breakdown
            status_counts = defaultdict(int)
            for metrics in self.agent_metrics.values():
                status_counts[metrics.status.value] += 1
            
            # Resource utilization
            total_cpu = sum(self.resource_usage.get(f"{aid}_cpu", 0) 
                          for aid in self.agents)
            total_memory = sum(self.resource_usage.get(f"{aid}_memory", 0) 
                             for aid in self.agents)
            
            # Group information
            group_info = {
                group: len(agent_set) 
                for group, agent_set in self.agent_groups.items()
            }
            
            return {
                'total_agents': len(self.agents),
                'max_agents': self.max_agents,
                'agents_by_status': dict(status_counts),
                'total_agents_created': self.total_agents_created,
                'total_messages_processed': self.total_messages_processed,
                'coordination_events': self.coordination_events,
                'pending_messages': len(self.message_queue),
                'resource_usage': {
                    'total_cpu_percent': min(100.0, total_cpu),
                    'total_memory_percent': min(100.0, total_memory),
                },
                'groups': group_info,
                'communication_enabled': self.enable_communication,
                'load_balancing_enabled': self.enable_load_balancing,
                'orchestration_active': (self.orchestrator_thread and 
                                       self.orchestrator_thread.is_alive())
            }
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        while not self.stop_event.is_set():
            try:
                # Process messages
                if self.enable_communication:
                    self._process_messages()
                
                # Load balancing
                if self.enable_load_balancing:
                    self._balance_load()
                
                # Coordination events
                self._process_coordination_events()
                
                # Sleep until next cycle
                self.stop_event.wait(1.0)
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                self.stop_event.wait(1.0)
    
    def _monitoring_loop(self):
        """Resource and performance monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self._update_resource_usage()
                self._check_agent_health()
                self.stop_event.wait(self.resource_monitoring_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self.stop_event.wait(1.0)
    
    def _process_messages(self):
        """Process pending communication messages."""
        processed = 0
        while self.message_queue and processed < 100:  # Limit to prevent blocking
            try:
                message = self.message_queue.popleft()
                self._handle_message(message)
                processed += 1
            except IndexError:
                break
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    def _handle_message(self, message: CoordinationMessage):
        """Handle individual coordination message."""
        # Route message to appropriate handler
        handler = self.communication_handlers.get(message.message_type)
        if handler:
            handler(message)
        else:
            # Default handling
            self.logger.debug(f"Unhandled message type: {message.message_type}")
    
    def _run_agent_episodes(self,
                           agent_id: str,
                           agent: Any,
                           environment_factory: Callable,
                           num_episodes: int,
                           episode_length: int) -> Dict[str, Any]:
        """Run episodes for a specific agent."""
        results = {
            'agent_id': agent_id,
            'episodes_completed': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'episode_rewards': [],
            'avg_free_energy': 0.0
        }
        
        # Update status
        with self.lock:
            self.agent_metrics[agent_id].status = AgentStatus.ACTIVE
        
        try:
            env = environment_factory()
            
            for episode in range(num_episodes):
                episode_reward = 0.0
                episode_steps = 0
                free_energies = []
                
                obs = env.reset()
                agent.reset(obs)
                
                for step in range(episode_length):
                    # Agent acts
                    action = agent.act(obs)
                    
                    # Environment step (handle different return formats)
                    env_result = env.step(action)
                    if len(env_result) == 4:
                        next_obs, reward, done, info = env_result
                    else:
                        next_obs, reward, done = env_result[:3]
                        info = {}
                    
                    # Update agent
                    agent.update_model(next_obs, action, reward)
                    
                    episode_reward += reward
                    episode_steps += 1
                    
                    # Record free energy if available
                    if hasattr(agent, 'history') and agent.history['free_energy']:
                        fe = agent.history['free_energy'][-1]
                        if hasattr(fe, 'total'):
                            free_energies.append(fe.total)
                        else:
                            free_energies.append(float(fe))
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                # Record episode results
                results['episodes_completed'] += 1
                results['total_steps'] += episode_steps
                results['total_reward'] += episode_reward
                results['episode_rewards'].append(episode_reward)
                
                if free_energies:
                    results['avg_free_energy'] = np.mean(free_energies)
            
        except Exception as e:
            self.logger.error(f"Agent {agent_id} episode execution failed: {e}")
            raise
        
        finally:
            # Update status
            with self.lock:
                self.agent_metrics[agent_id].status = AgentStatus.IDLE
        
        return results
    
    def _update_agent_metrics_from_results(self, agent_id: str, results: Dict[str, Any]):
        """Update agent metrics based on execution results."""
        with self.lock:
            metrics = self.agent_metrics[agent_id]
            metrics.total_episodes += results['episodes_completed']
            metrics.total_steps += results['total_steps']
            
            if results['episode_rewards']:
                metrics.avg_reward = np.mean(results['episode_rewards'])
            
            if 'avg_free_energy' in results:
                metrics.avg_free_energy = results['avg_free_energy']
            
            metrics.last_update = time.time()
    
    def _aggregate_results(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        if not all_results:
            return {}
        
        total_episodes = sum(r['episodes_completed'] for r in all_results.values())
        total_steps = sum(r['total_steps'] for r in all_results.values())
        total_reward = sum(r['total_reward'] for r in all_results.values())
        
        all_episode_rewards = []
        all_free_energies = []
        
        for results in all_results.values():
            all_episode_rewards.extend(results['episode_rewards'])
            if 'avg_free_energy' in results:
                all_free_energies.append(results['avg_free_energy'])
        
        return {
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'avg_reward_per_episode': total_reward / max(1, total_episodes),
            'avg_steps_per_episode': total_steps / max(1, total_episodes),
            'avg_free_energy': np.mean(all_free_energies) if all_free_energies else 0.0,
            'best_episode_reward': max(all_episode_rewards) if all_episode_rewards else 0.0,
            'worst_episode_reward': min(all_episode_rewards) if all_episode_rewards else 0.0
        }
    
    def _balance_load(self):
        """Simple load balancing across agents."""
        # Simplified load balancing - could be much more sophisticated
        with self.lock:
            active_agents = [
                aid for aid, metrics in self.agent_metrics.items()
                if metrics.status == AgentStatus.ACTIVE
            ]
            
            if len(active_agents) > self.max_agents * 0.8:
                # Too many active agents, pause some
                for agent_id in active_agents[int(self.max_agents * 0.8):]:
                    self.agent_metrics[agent_id].status = AgentStatus.PAUSED
    
    def _update_resource_usage(self):
        """Update resource usage metrics (simplified)."""
        # In a real implementation, this would use psutil or similar
        with self.lock:
            for agent_id in self.agents:
                # Simulate resource usage based on agent activity
                metrics = self.agent_metrics[agent_id]
                if metrics.status == AgentStatus.ACTIVE:
                    metrics.cpu_usage = np.random.uniform(10, 50)
                    metrics.memory_usage = np.random.uniform(5, 25)
                else:
                    metrics.cpu_usage = np.random.uniform(1, 5)
                    metrics.memory_usage = np.random.uniform(1, 10)
                
                self.resource_usage[f"{agent_id}_cpu"] = metrics.cpu_usage
                self.resource_usage[f"{agent_id}_memory"] = metrics.memory_usage
    
    def _check_agent_health(self):
        """Check health of all registered agents."""
        with self.lock:
            current_time = time.time()
            for agent_id, metrics in self.agent_metrics.items():
                # Check for stale agents
                if current_time - metrics.last_update > 300:  # 5 minutes
                    if metrics.status == AgentStatus.ACTIVE:
                        metrics.status = AgentStatus.FAILED
                        self.logger.warning(f"Agent {agent_id} marked as failed (stale)")
    
    def _process_coordination_events(self):
        """Process coordination events."""
        # This would implement specific coordination algorithms
        self.coordination_events += 1
    
    def _coordinate_experience_sharing(self, frequency: int) -> Dict[str, Any]:
        """Coordinate experience sharing between agents."""
        return {
            'shared_experiences': 0,
            'participating_agents': len(self.agents)
        }
    
    def _coordinate_consensus_building(self) -> Dict[str, Any]:
        """Coordinate consensus building across agents."""
        return {
            'consensus_rounds': 0,
            'final_consensus': None
        }
    
    def _coordinate_hierarchical_learning(self) -> Dict[str, Any]:
        """Coordinate hierarchical learning structures."""
        return {
            'hierarchy_levels': 0,
            'coordination_efficiency': 0.0
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_orchestration()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_orchestration()