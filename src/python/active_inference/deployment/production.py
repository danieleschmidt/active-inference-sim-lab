"""
Production-ready Active Inference deployment components.

This module provides production-hardened agent implementations with
monitoring, auto-scaling, and enterprise features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import threading
import queue
import json
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
import os

from ..core.agent import ActiveInferenceAgent
from ..performance.optimization import OptimizedActiveInferenceAgent, OptimizationConfig
from ..utils.validation import ValidationError


@dataclass
class ProductionConfig:
    """Configuration for production deployment."""
    # Performance settings
    optimization_level: str = "production"  # "development", "staging", "production"
    max_memory_mb: int = 2048
    max_cpu_cores: int = 4
    enable_gpu: bool = False
    
    # Reliability settings
    health_check_interval: float = 30.0
    max_error_rate: float = 0.05
    circuit_breaker_threshold: int = 10
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"
    enable_tracing: bool = True
    
    # Scaling settings
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    
    # Security settings
    enable_authentication: bool = True
    api_key_required: bool = True
    rate_limit_requests_per_minute: int = 1000
    
    # Persistence settings
    enable_persistence: bool = True
    checkpoint_interval: float = 300.0  # 5 minutes
    backup_retention_days: int = 7


class ProductionAgent:
    """
    Production-ready Active Inference agent with enterprise features.
    
    Includes monitoring, health checks, circuit breakers, and auto-recovery.
    """
    
    def __init__(self,
                 agent_config: Dict[str, Any],
                 production_config: ProductionConfig = None):
        """
        Initialize production agent.
        
        Args:
            agent_config: Configuration for underlying agent
            production_config: Production deployment configuration
        """
        self.prod_config = production_config or ProductionConfig()
        self.agent_config = agent_config
        
        # Setup logging
        self._setup_logging()
        
        # Initialize core agent
        self._initialize_agent()
        
        # Production features
        self._setup_health_monitoring()
        self._setup_circuit_breaker()
        self._setup_metrics()
        self._setup_persistence()
        
        # Runtime state
        self.is_running = False
        self.is_healthy = True
        self.error_count = 0
        self.request_count = 0
        self.last_health_check = time.time()
        
        # Graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("Production agent initialized successfully")
    
    def _setup_logging(self):
        """Setup production logging."""
        log_level = getattr(logging, self.prod_config.log_level.upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - '
                   'pid:%(process)d - thread:%(thread)d',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('agent.log') if self.prod_config.enable_metrics else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger("ProductionAgent")
    
    def _initialize_agent(self):
        """Initialize optimized agent based on production config."""
        
        # Create optimization config based on production settings
        opt_config = OptimizationConfig(
            use_gpu=self.prod_config.enable_gpu,
            enable_caching=True,
            parallel_belief_updates=True,
            vectorized_planning=True,
            memory_limit_mb=self.prod_config.max_memory_mb,
            optimization_level="speed" if self.prod_config.optimization_level == "production" else "balanced"
        )
        
        # Initialize optimized agent
        self.agent = OptimizedActiveInferenceAgent(
            optimization_config=opt_config,
            **self.agent_config
        )
        
        self.logger.info(f"Agent initialized with optimization level: {self.prod_config.optimization_level}")
    
    def _setup_health_monitoring(self):
        """Setup health monitoring system."""
        self.health_metrics = {
            'last_successful_action': time.time(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0
        }
        
        # Start health check thread
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()
    
    def _setup_circuit_breaker(self):
        """Setup circuit breaker for fault tolerance."""
        self.circuit_breaker = {
            'state': 'closed',  # closed, open, half-open
            'failure_count': 0,
            'last_failure_time': 0,
            'success_count': 0,
            'timeout': 60.0  # Reset timeout in seconds
        }
    
    def _setup_metrics(self):
        """Setup metrics collection."""
        if self.prod_config.enable_metrics:
            self.metrics = {
                'requests_total': 0,
                'requests_successful': 0,
                'requests_failed': 0,
                'response_time_total': 0.0,
                'agent_actions_total': 0,
                'belief_updates_total': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
        else:
            self.metrics = {}
    
    def _setup_persistence(self):
        """Setup persistence and checkpointing."""
        if self.prod_config.enable_persistence:
            self.checkpoint_dir = Path("checkpoints")
            self.checkpoint_dir.mkdir(exist_ok=True)
            
            # Start checkpoint thread
            self.checkpoint_thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
            self.checkpoint_thread.start()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def start(self):
        """Start the production agent."""
        self.is_running = True
        self.logger.info("Production agent started")
    
    def shutdown(self):
        """Shutdown the production agent gracefully."""
        self.logger.info("Shutting down production agent...")
        
        self.is_running = False
        
        # Save final checkpoint
        if self.prod_config.enable_persistence:
            self._save_checkpoint()
        
        # Log final metrics
        if self.prod_config.enable_metrics:
            self._log_final_metrics()
        
        self.logger.info("Production agent shutdown complete")
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Production-hardened action method with monitoring and error handling.
        
        Args:
            observation: Environment observation
            
        Returns:
            Selected action
            
        Raises:
            ValidationError: If circuit breaker is open or validation fails
        """
        start_time = time.time()
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise ValidationError("Circuit breaker is open - service temporarily unavailable")
        
        # Validate input
        if not self._validate_observation(observation):
            self._record_failure()
            raise ValidationError("Invalid observation provided")
        
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                self._record_failure()
                raise ValidationError("Rate limit exceeded")
            
            # Execute action with timeout
            action = self._execute_with_timeout(
                lambda: self.agent.act(observation),
                timeout=self.prod_config.timeout_seconds
            )
            
            # Record success
            self._record_success(time.time() - start_time)
            
            return action
            
        except Exception as e:
            self._record_failure()
            self.logger.error(f"Action execution failed: {e}")
            raise
    
    def update_model(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        """
        Production-hardened model update with error handling.
        
        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
        """
        try:
            # Validate inputs
            if not (self._validate_observation(observation) and 
                   self._validate_action(action) and
                   self._validate_reward(reward)):
                self.logger.warning("Invalid inputs for model update")
                return
            
            # Execute update
            self.agent.update_model(observation, action, reward)
            
            # Update metrics
            if self.prod_config.enable_metrics:
                self.metrics['belief_updates_total'] += 1
            
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            self._record_failure()
    
    def _validate_observation(self, observation: np.ndarray) -> bool:
        """Validate observation input."""
        if not isinstance(observation, np.ndarray):
            return False
        
        if observation.shape != (self.agent.obs_dim,):
            return False
        
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            return False
        
        return True
    
    def _validate_action(self, action: np.ndarray) -> bool:
        """Validate action input."""
        if not isinstance(action, np.ndarray):
            return False
        
        if action.shape != (self.agent.action_dim,):
            return False
        
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            return False
        
        return True
    
    def _validate_reward(self, reward: float) -> bool:
        """Validate reward input."""
        return isinstance(reward, (int, float)) and not (np.isnan(reward) or np.isinf(reward))
    
    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state."""
        cb = self.circuit_breaker
        current_time = time.time()
        
        if cb['state'] == 'open':
            # Check if timeout has elapsed
            if current_time - cb['last_failure_time'] > cb['timeout']:
                cb['state'] = 'half-open'
                cb['success_count'] = 0
                self.logger.info("Circuit breaker moved to half-open state")
            else:
                return False
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check rate limiting."""
        # Simple rate limiting implementation
        current_time = time.time()
        
        # Reset counter every minute
        if not hasattr(self, '_rate_limit_window_start'):
            self._rate_limit_window_start = current_time
            self._rate_limit_count = 0
        
        if current_time - self._rate_limit_window_start > 60.0:
            self._rate_limit_window_start = current_time
            self._rate_limit_count = 0
        
        if self._rate_limit_count >= self.prod_config.rate_limit_requests_per_minute:
            return False
        
        self._rate_limit_count += 1
        return True
    
    def _execute_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Execute function with timeout."""
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def worker():
            try:
                result = func()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Return result
        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("No result returned from operation")
    
    def _record_success(self, response_time: float):
        """Record successful operation."""
        
        # Update circuit breaker
        cb = self.circuit_breaker
        if cb['state'] == 'half-open':
            cb['success_count'] += 1
            if cb['success_count'] >= 5:  # Require 5 successes to close
                cb['state'] = 'closed'
                cb['failure_count'] = 0
                self.logger.info("Circuit breaker closed")
        
        # Update metrics
        self.request_count += 1
        self.health_metrics['successful_requests'] += 1
        self.health_metrics['total_requests'] += 1
        self.health_metrics['last_successful_action'] = time.time()
        
        # Update average response time
        total_time = self.health_metrics['average_response_time'] * (self.request_count - 1)
        self.health_metrics['average_response_time'] = (total_time + response_time) / self.request_count
        
        if self.prod_config.enable_metrics:
            self.metrics['requests_successful'] += 1
            self.metrics['requests_total'] += 1
            self.metrics['response_time_total'] += response_time
    
    def _record_failure(self):
        """Record failed operation."""
        
        self.error_count += 1
        
        # Update circuit breaker
        cb = self.circuit_breaker
        cb['failure_count'] += 1
        cb['last_failure_time'] = time.time()
        
        if cb['failure_count'] >= self.prod_config.circuit_breaker_threshold:
            cb['state'] = 'open'
            self.logger.warning("Circuit breaker opened due to high failure rate")
        
        # Update metrics
        self.health_metrics['failed_requests'] += 1
        self.health_metrics['total_requests'] += 1
        
        if self.prod_config.enable_metrics:
            self.metrics['requests_failed'] += 1
            self.metrics['requests_total'] += 1
    
    def _health_check_loop(self):
        """Health check monitoring loop."""
        
        while self.is_running:
            try:
                self._perform_health_check()
                time.sleep(self.prod_config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
    
    def _perform_health_check(self):
        """Perform health check."""
        
        current_time = time.time()
        
        # Check error rate
        if self.health_metrics['total_requests'] > 0:
            error_rate = self.health_metrics['failed_requests'] / self.health_metrics['total_requests']
            if error_rate > self.prod_config.max_error_rate:
                self.is_healthy = False
                self.logger.warning(f"High error rate detected: {error_rate:.3f}")
            else:
                self.is_healthy = True
        
        # Check if agent is responsive
        time_since_last_action = current_time - self.health_metrics['last_successful_action']
        if time_since_last_action > 300:  # 5 minutes
            self.logger.warning(f"No successful actions in {time_since_last_action:.1f} seconds")
        
        # Update memory usage (simplified)
        try:
            import psutil
            process = psutil.Process()
            self.health_metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            self.health_metrics['cpu_usage_percent'] = process.cpu_percent()
        except ImportError:
            pass  # psutil not available
        
        self.last_health_check = current_time
    
    def _checkpoint_loop(self):
        """Checkpoint saving loop."""
        
        while self.is_running:
            try:
                time.sleep(self.prod_config.checkpoint_interval)
                if self.is_running:  # Check again after sleep
                    self._save_checkpoint()
            except Exception as e:
                self.logger.error(f"Checkpoint save failed: {e}")
    
    def _save_checkpoint(self):
        """Save agent checkpoint."""
        
        if not self.prod_config.enable_persistence:
            return
        
        try:
            timestamp = int(time.time())
            checkpoint_path = self.checkpoint_dir / f"agent_checkpoint_{timestamp}.json"
            
            self.agent.save_checkpoint(str(checkpoint_path))
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files."""
        
        current_time = time.time()
        max_age = self.prod_config.backup_retention_days * 24 * 3600
        
        for checkpoint_file in self.checkpoint_dir.glob("agent_checkpoint_*.json"):
            try:
                file_age = current_time - checkpoint_file.stat().st_mtime
                if file_age > max_age:
                    checkpoint_file.unlink()
                    self.logger.debug(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {checkpoint_file}: {e}")
    
    def _log_final_metrics(self):
        """Log final metrics before shutdown."""
        
        self.logger.info("=== FINAL METRICS ===")
        self.logger.info(f"Total requests: {self.health_metrics['total_requests']}")
        self.logger.info(f"Successful requests: {self.health_metrics['successful_requests']}")
        self.logger.info(f"Failed requests: {self.health_metrics['failed_requests']}")
        
        if self.health_metrics['total_requests'] > 0:
            success_rate = self.health_metrics['successful_requests'] / self.health_metrics['total_requests']
            self.logger.info(f"Success rate: {success_rate:.3f}")
        
        self.logger.info(f"Average response time: {self.health_metrics['average_response_time']:.3f}s")
        self.logger.info(f"Circuit breaker state: {self.circuit_breaker['state']}")
        
        if self.prod_config.enable_metrics:
            agent_stats = self.agent.get_performance_stats()
            self.logger.info(f"Cache hit rate: {agent_stats.get('cache_hit_rate', 0):.3f}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        
        return {
            'is_healthy': self.is_healthy,
            'is_running': self.is_running,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'error_count': self.error_count,
            'request_count': self.request_count,
            'last_health_check': self.last_health_check,
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            'metrics': self.health_metrics.copy()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get production metrics."""
        
        health_status = self.get_health_status()
        agent_stats = self.agent.get_performance_stats()
        
        return {
            'health': health_status,
            'agent_performance': agent_stats,
            'production_metrics': self.metrics.copy() if self.prod_config.enable_metrics else {},
            'configuration': {
                'optimization_level': self.prod_config.optimization_level,
                'max_memory_mb': self.prod_config.max_memory_mb,
                'enable_gpu': self.prod_config.enable_gpu,
                'circuit_breaker_threshold': self.prod_config.circuit_breaker_threshold
            }
        }


class LoadBalancer:
    """Load balancer for multiple agent instances."""
    
    def __init__(self, agents: List[ProductionAgent]):
        """
        Initialize load balancer.
        
        Args:
            agents: List of production agent instances
        """
        self.agents = agents
        self.current_index = 0
        self.logger = logging.getLogger("LoadBalancer")
    
    def get_next_agent(self) -> ProductionAgent:
        """Get next agent using round-robin scheduling."""
        
        # Filter healthy agents
        healthy_agents = [agent for agent in self.agents if agent.is_healthy]
        
        if not healthy_agents:
            raise RuntimeError("No healthy agents available")
        
        # Round-robin selection
        agent = healthy_agents[self.current_index % len(healthy_agents)]
        self.current_index += 1
        
        return agent
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Route action request to available agent."""
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                agent = self.get_next_agent()
                return agent.act(observation)
            except Exception as e:
                self.logger.warning(f"Agent request failed (attempt {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt
                    raise
        
        raise RuntimeError("All agent requests failed")


class AutoScaler:
    """Auto-scaling for agent instances based on load."""
    
    def __init__(self,
                 agent_factory: Callable[[], ProductionAgent],
                 min_instances: int = 1,
                 max_instances: int = 10):
        """
        Initialize auto-scaler.
        
        Args:
            agent_factory: Function to create new agent instances
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
        """
        self.agent_factory = agent_factory
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.instances: List[ProductionAgent] = []
        self.logger = logging.getLogger("AutoScaler")
        
        # Initialize minimum instances
        for _ in range(min_instances):
            self.scale_up()
    
    def scale_up(self) -> bool:
        """Add new agent instance."""
        
        if len(self.instances) >= self.max_instances:
            self.logger.warning("Cannot scale up: maximum instances reached")
            return False
        
        try:
            new_agent = self.agent_factory()
            new_agent.start()
            self.instances.append(new_agent)
            
            self.logger.info(f"Scaled up: {len(self.instances)} instances")
            return True
            
        except Exception as e:
            self.logger.error(f"Scale up failed: {e}")
            return False
    
    def scale_down(self) -> bool:
        """Remove agent instance."""
        
        if len(self.instances) <= self.min_instances:
            self.logger.warning("Cannot scale down: minimum instances reached")
            return False
        
        try:
            # Remove least healthy instance
            agent_to_remove = min(self.instances, 
                                key=lambda a: a.health_metrics['successful_requests'])
            
            agent_to_remove.shutdown()
            self.instances.remove(agent_to_remove)
            
            self.logger.info(f"Scaled down: {len(self.instances)} instances")
            return True
            
        except Exception as e:
            self.logger.error(f"Scale down failed: {e}")
            return False
    
    def check_scaling_conditions(self):
        """Check if scaling is needed based on metrics."""
        
        if not self.instances:
            return
        
        # Calculate average CPU utilization
        total_cpu = sum(agent.health_metrics.get('cpu_usage_percent', 0) 
                       for agent in self.instances)
        avg_cpu = total_cpu / len(self.instances)
        
        # Scale up if high utilization
        if avg_cpu > 80 and len(self.instances) < self.max_instances:
            self.scale_up()
        
        # Scale down if low utilization
        elif avg_cpu < 30 and len(self.instances) > self.min_instances:
            self.scale_down()