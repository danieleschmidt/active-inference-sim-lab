"""
Container and deployment management for Active Inference agents.

This module provides containerization, orchestration, and deployment
management for production Active Inference systems.
"""

import docker
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
import threading


logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for a single container instance."""
    name: str
    image: str
    cpu_limit: float = 1.0
    memory_limit: str = "512m"
    environment: Dict[str, str] = None
    ports: Dict[str, str] = None
    volumes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.ports is None:
            self.ports = {}
        if self.volumes is None:
            self.volumes = {}


class ContainerManager:
    """
    Manages Docker containers for Active Inference agents.
    
    Provides container lifecycle management, health monitoring,
    and automatic scaling based on system load.
    """
    
    def __init__(self, max_containers: int = 10):
        """Initialize container manager."""
        self.max_containers = max_containers
        self.containers: Dict[str, Any] = {}
        self.client = None
        self.health_monitor_thread = None
        self.monitoring_active = False
        
        # Try to connect to Docker
        try:
            self.client = docker.from_env()
            logger.info("Connected to Docker daemon")
        except Exception as e:
            logger.warning(f"Could not connect to Docker: {e}")
            self.client = None
    
    def start_container(self, config: ContainerConfig) -> Optional[str]:
        """Start a new container with given configuration."""
        if not self.client:
            logger.error("Docker client not available")
            return None
            
        try:
            container = self.client.containers.run(
                config.image,
                name=config.name,
                detach=True,
                environment=config.environment,
                ports=config.ports,
                volumes=config.volumes,
                mem_limit=config.memory_limit,
                nano_cpus=int(config.cpu_limit * 1e9)
            )
            
            self.containers[config.name] = {
                'container': container,
                'config': config,
                'started_at': time.time(),
                'status': 'running'
            }
            
            logger.info(f"Started container {config.name}")
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to start container {config.name}: {e}")
            return None
    
    def stop_container(self, name: str) -> bool:
        """Stop a container by name."""
        if name not in self.containers:
            logger.warning(f"Container {name} not found")
            return False
            
        try:
            container_info = self.containers[name]
            container_info['container'].stop()
            container_info['status'] = 'stopped'
            logger.info(f"Stopped container {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop container {name}: {e}")
            return False
    
    def get_container_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get resource usage stats for a container."""
        if name not in self.containers:
            return None
            
        try:
            container = self.containers[name]['container']
            stats = container.stats(stream=False)
            
            # Calculate CPU and memory usage
            cpu_stats = stats.get('cpu_stats', {})
            memory_stats = stats.get('memory_stats', {})
            
            cpu_usage = 0
            if 'system_cpu_usage' in cpu_stats and 'cpu_usage' in cpu_stats:
                cpu_usage = (cpu_stats['cpu_usage']['total_usage'] / 
                            cpu_stats['system_cpu_usage']) * 100
            
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            return {
                'name': name,
                'cpu_percent': cpu_usage,
                'memory_usage': memory_usage,
                'memory_limit': memory_limit,
                'memory_percent': memory_percent,
                'status': self.containers[name]['status']
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for container {name}: {e}")
            return None
    
    def scale_containers(self, target_count: int, base_config: ContainerConfig) -> List[str]:
        """Scale containers to target count."""
        current_count = len([c for c in self.containers.values() 
                           if c['status'] == 'running'])
        
        container_ids = []
        
        if target_count > current_count:
            # Scale up
            for i in range(current_count, min(target_count, self.max_containers)):
                config = ContainerConfig(
                    name=f"{base_config.name}_{i}",
                    image=base_config.image,
                    cpu_limit=base_config.cpu_limit,
                    memory_limit=base_config.memory_limit,
                    environment=base_config.environment.copy(),
                    ports={f"{8000+i}": "8000"} if base_config.ports else {},
                    volumes=base_config.volumes.copy()
                )
                
                container_id = self.start_container(config)
                if container_id:
                    container_ids.append(container_id)
        
        elif target_count < current_count:
            # Scale down
            running_containers = [(name, info) for name, info in self.containers.items()
                                if info['status'] == 'running']
            
            for i in range(target_count, current_count):
                if i < len(running_containers):
                    name = running_containers[i][0]
                    self.stop_container(name)
        
        logger.info(f"Scaled containers to {target_count}")
        return container_ids
    
    def start_health_monitoring(self, check_interval: int = 30):
        """Start health monitoring thread."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            args=(check_interval,),
            daemon=True
        )
        self.health_monitor_thread.start()
        logger.info("Started health monitoring")
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join()
        logger.info("Stopped health monitoring")
    
    def _health_monitor_loop(self, check_interval: int):
        """Health monitoring loop."""
        while self.monitoring_active:
            try:
                # Check container health
                for name, info in list(self.containers.items()):
                    try:
                        container = info['container']
                        container.reload()
                        
                        if container.status != 'running':
                            logger.warning(f"Container {name} not running: {container.status}")
                            info['status'] = container.status
                            
                            # Attempt restart if exited
                            if container.status == 'exited':
                                logger.info(f"Attempting to restart container {name}")
                                container.restart()
                                info['status'] = 'running'
                                
                    except Exception as e:
                        logger.error(f"Health check failed for container {name}: {e}")
                
                # Check system resources
                system_stats = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                }
                
                # Auto-scaling based on load
                if system_stats['cpu_percent'] > 80:
                    logger.info("High CPU load detected, consider scaling up")
                elif system_stats['cpu_percent'] < 20:
                    logger.info("Low CPU load detected, consider scaling down")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(check_interval)
    
    def cleanup(self):
        """Clean up all containers and resources."""
        self.stop_health_monitoring()
        
        for name in list(self.containers.keys()):
            self.stop_container(name)
            
        logger.info("Container manager cleanup complete")


class ContainerOrchestrator:
    """
    High-level orchestration for multi-container deployments.
    
    Manages complex deployments with load balancing, service discovery,
    and automated failover capabilities.
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.manager = ContainerManager()
        self.services: Dict[str, Dict[str, Any]] = {}
        self.load_balancer = None
    
    def deploy_service(self, service_name: str, config: ContainerConfig, 
                      replicas: int = 1) -> bool:
        """Deploy a service with specified number of replicas."""
        try:
            container_ids = []
            
            for i in range(replicas):
                replica_config = ContainerConfig(
                    name=f"{service_name}_replica_{i}",
                    image=config.image,
                    cpu_limit=config.cpu_limit,
                    memory_limit=config.memory_limit,
                    environment=config.environment.copy(),
                    ports={f"{8000+i}": "8000"} if config.ports else {},
                    volumes=config.volumes.copy()
                )
                
                container_id = self.manager.start_container(replica_config)
                if container_id:
                    container_ids.append(container_id)
            
            self.services[service_name] = {
                'config': config,
                'replicas': replicas,
                'container_ids': container_ids,
                'deployed_at': time.time()
            }
            
            logger.info(f"Deployed service {service_name} with {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    def scale_service(self, service_name: str, new_replicas: int) -> bool:
        """Scale a service to new replica count."""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        try:
            service = self.services[service_name]
            config = service['config']
            
            # Use container manager scaling
            container_ids = self.manager.scale_containers(new_replicas, config)
            
            service['replicas'] = new_replicas
            service['container_ids'] = container_ids
            
            logger.info(f"Scaled service {service_name} to {new_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale service {service_name}: {e}")
            return False
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status of all service replicas."""
        if service_name not in self.services:
            return {'status': 'not_found'}
        
        service = self.services[service_name]
        replica_stats = []
        
        for i in range(service['replicas']):
            replica_name = f"{service_name}_replica_{i}"
            stats = self.manager.get_container_stats(replica_name)
            if stats:
                replica_stats.append(stats)
        
        healthy_replicas = len([s for s in replica_stats if s.get('status') == 'running'])
        
        return {
            'service_name': service_name,
            'total_replicas': service['replicas'],
            'healthy_replicas': healthy_replicas,
            'health_ratio': healthy_replicas / service['replicas'],
            'replica_stats': replica_stats,
            'deployed_at': service['deployed_at']
        }
    
    def start_monitoring(self):
        """Start orchestrator monitoring."""
        self.manager.start_health_monitoring()
        logger.info("Started orchestrator monitoring")
    
    def stop_monitoring(self):
        """Stop orchestrator monitoring."""
        self.manager.stop_health_monitoring()
        logger.info("Stopped orchestrator monitoring")
    
    def cleanup(self):
        """Clean up all services and resources."""
        for service_name in list(self.services.keys()):
            logger.info(f"Cleaning up service {service_name}")
        
        self.manager.cleanup()
        self.services.clear()
        logger.info("Orchestrator cleanup complete")


# Factory functions for easy usage
def create_agent_container(agent_id: str, image: str = "active-inference:latest") -> ContainerConfig:
    """Create a container configuration for an Active Inference agent."""
    return ContainerConfig(
        name=f"active_inference_agent_{agent_id}",
        image=image,
        cpu_limit=1.0,
        memory_limit="1g",
        environment={
            'AGENT_ID': agent_id,
            'LOG_LEVEL': 'INFO',
            'PYTHONPATH': '/app/src/python'
        },
        ports={'8000': '8000'},
        volumes={'/tmp/active_inference_logs': '/app/logs'}
    )


def create_development_cluster(num_agents: int = 3) -> ContainerOrchestrator:
    """Create a development cluster with multiple agents."""
    orchestrator = ContainerOrchestrator()
    
    # Deploy multiple agent services
    for i in range(num_agents):
        config = create_agent_container(f"dev_agent_{i}")
        orchestrator.deploy_service(f"agent_service_{i}", config, replicas=1)
    
    orchestrator.start_monitoring()
    return orchestrator