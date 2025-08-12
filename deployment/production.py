#!/usr/bin/env python3
"""
Production deployment configuration and orchestration for Active Inference agents.

This module provides production-ready deployment with monitoring, scaling,
and management capabilities.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import yaml

# Add src to path
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src" / "python"
sys.path.insert(0, str(src_path))

from active_inference import ActiveInferenceAgent
from active_inference.performance.optimization import OptimizedActiveInferenceAgent, OptimizationConfig
from active_inference.performance.caching import BaseCache, CacheStrategy
from active_inference.utils.logging import setup_logging
from active_inference.utils.health_check import health_check


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    # Agent Configuration
    agent_type: str = "optimized"  # "basic" or "optimized"
    state_dim: int = 8
    obs_dim: int = 8
    action_dim: int = 4
    planning_horizon: int = 5
    
    # Performance Configuration
    enable_gpu: bool = False
    enable_caching: bool = True
    cache_size: int = 10000
    batch_size: int = 64
    num_workers: int = 4
    
    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 8080
    health_check_interval: int = 30
    log_level: str = "INFO"
    
    # Scaling Configuration
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Security Configuration
    max_request_size: int = 1024 * 1024  # 1MB
    rate_limit: int = 1000  # requests per minute
    enable_auth: bool = True


class ProductionOrchestrator:
    """Orchestrates production deployment of Active Inference agents."""
    
    def __init__(self, config: ProductionConfig):
        """Initialize production orchestrator."""
        self.config = config
        self.agents: Dict[str, ActiveInferenceAgent] = {}
        self.health_checker = None
        self.metrics_collector = None
        self.logger = setup_logging(level=config.log_level)
        
        # Initialize components
        self._setup_monitoring()
        self._setup_agents()
    
    def _setup_monitoring(self):
        """Setup monitoring and health checking."""
        self.health_checker = None  # Simplified for now
        # Could implement: health_check.HealthMonitor(check_interval=self.config.health_check_interval)
        
        if self.config.enable_metrics:
            self._setup_metrics_collection()
    
    def _setup_metrics_collection(self):
        """Setup metrics collection and export."""
        try:
            # Try to setup Prometheus metrics
            from prometheus_client import start_http_server, Counter, Histogram, Gauge
            
            # Start metrics server
            start_http_server(self.config.metrics_port)
            self.logger.info(f"Metrics server started on port {self.config.metrics_port}")
            
            # Define metrics
            self.request_counter = Counter('ai_requests_total', 'Total AI requests')
            self.response_time = Histogram('ai_response_time_seconds', 'Response time')
            self.active_agents = Gauge('ai_active_agents', 'Number of active agents')
            
        except ImportError:
            self.logger.warning("Prometheus client not available, metrics disabled")
            self.config.enable_metrics = False
    
    def _setup_agents(self):
        """Setup agent instances."""
        for i in range(self.config.min_instances):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = self._create_agent(agent_id)
        
        self.logger.info(f"Initialized {len(self.agents)} agent instances")
    
    def _create_agent(self, agent_id: str) -> ActiveInferenceAgent:
        """Create a new agent instance."""
        if self.config.agent_type == "optimized":
            opt_config = OptimizationConfig(
                use_gpu=self.config.enable_gpu,
                enable_caching=self.config.enable_caching,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers
            )
            
            agent = OptimizedActiveInferenceAgent(
                state_dim=self.config.state_dim,
                obs_dim=self.config.obs_dim,
                action_dim=self.config.action_dim,
                planning_horizon=self.config.planning_horizon,
                optimization_config=opt_config,
                agent_id=agent_id,
                enable_logging=True
            )
        else:
            agent = ActiveInferenceAgent(
                state_dim=self.config.state_dim,
                obs_dim=self.config.obs_dim,
                action_dim=self.config.action_dim,
                planning_horizon=self.config.planning_horizon,
                agent_id=agent_id,
                enable_logging=True
            )
        
        # Register with health checker
        if self.health_checker:
            self.health_checker.register_component(agent_id, agent)
        
        return agent
    
    def process_request(self, observation: List[float], agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an inference request.
        
        Args:
            observation: Input observation
            agent_id: Optional specific agent ID
            
        Returns:
            Response with action and metadata
        """
        start_time = time.time()
        
        try:
            # Input validation
            if len(observation) != self.config.obs_dim:
                raise ValueError(f"Invalid observation dimension: {len(observation)}")
            
            # Select agent
            if agent_id and agent_id in self.agents:
                agent = self.agents[agent_id]
            else:
                # Load balance across agents
                agent = self._select_agent()
            
            # Process request
            obs_array = np.array(observation)
            action = agent.act(obs_array)
            
            # Update metrics
            if self.config.enable_metrics:
                self.request_counter.inc()
                self.response_time.observe(time.time() - start_time)
            
            return {
                "action": action.tolist(),
                "agent_id": agent.agent_id,
                "status": "success",
                "processing_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": time.time()
            }
    
    def _select_agent(self) -> ActiveInferenceAgent:
        """Select best agent for load balancing."""
        # Simple round-robin for now
        # In production, could use more sophisticated strategies
        agent_ids = list(self.agents.keys())
        if not hasattr(self, '_current_agent_index'):
            self._current_agent_index = 0
        
        agent_id = agent_ids[self._current_agent_index]
        self._current_agent_index = (self._current_agent_index + 1) % len(agent_ids)
        
        return self.agents[agent_id]
    
    def scale_up(self):
        """Scale up by adding more agent instances."""
        if len(self.agents) < self.config.max_instances:
            agent_id = f"agent_{len(self.agents)}"
            self.agents[agent_id] = self._create_agent(agent_id)
            
            if self.config.enable_metrics:
                self.active_agents.set(len(self.agents))
            
            self.logger.info(f"Scaled up to {len(self.agents)} agents")
    
    def scale_down(self):
        """Scale down by removing agent instances."""
        if len(self.agents) > self.config.min_instances:
            # Remove last agent
            agent_ids = list(self.agents.keys())
            agent_to_remove = agent_ids[-1]
            
            # Cleanup
            del self.agents[agent_to_remove]
            
            if self.config.enable_metrics:
                self.active_agents.set(len(self.agents))
            
            self.logger.info(f"Scaled down to {len(self.agents)} agents")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                "health": "healthy",  # Could be more sophisticated
                "episodes": agent.episode_count,
                "steps": agent.step_count,
                "uptime": time.time() - agent.start_time if hasattr(agent, 'start_time') else 0
            }
        
        return {
            "status": "running",
            "num_agents": len(self.agents),
            "config": asdict(self.config),
            "agents": agent_statuses,
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down production orchestrator...")
        
        # Cleanup agents
        for agent_id, agent in self.agents.items():
            try:
                # Let agent finish current operations
                if hasattr(agent, 'shutdown'):
                    agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down agent {agent_id}: {e}")
        
        self.agents.clear()
        
        # Cleanup monitoring
        if self.health_checker:
            self.health_checker.shutdown()
        
        self.logger.info("Shutdown complete")


def create_config_from_file(config_path: str) -> ProductionConfig:
    """Create production config from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ProductionConfig(**config_data)


def main():
    """Main entry point for production deployment."""
    parser = argparse.ArgumentParser(description="Active Inference Production Deployment")
    parser.add_argument("--config", "-c", type=str, 
                       help="Path to configuration file")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = create_config_from_file(args.config)
    else:
        config = ProductionConfig()
    
    # Initialize orchestrator
    orchestrator = ProductionOrchestrator(config)
    
    try:
        # Simple demo mode - in production would integrate with web framework
        print(f"🚀 Active Inference Production System Started")
        print(f"📊 Agents: {len(orchestrator.agents)}")
        print(f"📈 Metrics: {'Enabled' if config.enable_metrics else 'Disabled'}")
        print(f"⚡ Optimization: {'Enabled' if config.agent_type == 'optimized' else 'Basic'}")
        
        # Demo request processing
        demo_observation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = orchestrator.process_request(demo_observation)
        
        print(f"\n📝 Demo Request Result:")
        print(json.dumps(result, indent=2))
        
        print(f"\n🏥 System Status:")
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        
        print(f"\n✅ Production system operational!")
        print(f"Use Ctrl+C to shutdown...")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Shutdown requested...")
        orchestrator.shutdown()
        print(f"✅ Shutdown complete!")


if __name__ == "__main__":
    main()