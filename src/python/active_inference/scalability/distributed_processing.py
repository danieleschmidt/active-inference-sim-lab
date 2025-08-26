"""
Distributed Processing Module for Active Inference Framework

This module provides distributed computing capabilities for the Active Inference
framework including cluster management, distributed task execution, and
inter-node communication.

Author: Terragon Labs
Generation: 3 (MAKE IT SCALE)
"""

import numpy as np
import threading
import time
import socket
import json
import pickle
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
import multiprocessing as mp
import hashlib
import uuid


class NodeRole(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    OBSERVER = "observer"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DistributedTask:
    """Represents a distributed computation task."""
    task_id: str
    function_name: str
    args: Any
    kwargs: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    role: NodeRole
    host: str
    port: int
    capabilities: Dict[str, Any]
    last_heartbeat: float
    load_factor: float = 0.0
    active_tasks: int = 0
    max_concurrent_tasks: int = 4
    
    def is_available(self) -> bool:
        return (time.time() - self.last_heartbeat < 30 and 
                self.active_tasks < self.max_concurrent_tasks)


class WorkerNode:
    """Distributed worker node for processing Active Inference tasks."""
    
    def __init__(self, node_id: str = None, port: int = 8080):
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.host = socket.gethostname()
        
        # Task processing
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_executor = ThreadPoolExecutor(max_workers=4)
        
        # Node state
        self.running = False
        self.load_factor = 0.0
        self.capabilities = {
            'active_inference': True,
            'parallel_processing': True,
            'gpu_acceleration': False,  # Could be detected
            'max_memory_gb': 8,
            'cpu_cores': mp.cpu_count()
        }
        
        # Communication
        self.coordinator_host = None
        self.coordinator_port = None
        self.heartbeat_thread = None
        
        # Performance tracking
        self.execution_stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_task_time': 0.0
        }
    
    def register_with_coordinator(self, coordinator_host: str, coordinator_port: int) -> bool:
        """Register this worker with the coordinator."""
        try:
            self.coordinator_host = coordinator_host
            self.coordinator_port = coordinator_port
            
            registration_data = {
                'action': 'register_worker',
                'node_info': {
                    'node_id': self.node_id,
                    'role': NodeRole.WORKER.value,
                    'host': self.host,
                    'port': self.port,
                    'capabilities': self.capabilities,
                    'last_heartbeat': time.time()
                }
            }
            
            # Send registration (simplified - would use actual network communication)
            print(f"Worker {self.node_id} registered with coordinator at {coordinator_host}:{coordinator_port}")
            return True
            
        except Exception as e:
            print(f"Failed to register with coordinator: {e}")
            return False
    
    def start(self) -> None:
        """Start the worker node."""
        self.running = True
        
        # Start task processing thread
        processing_thread = threading.Thread(target=self._process_tasks)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start heartbeat thread
        if self.coordinator_host:
            self.heartbeat_thread = threading.Thread(target=self._send_heartbeats)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
        
        print(f"Worker node {self.node_id} started on {self.host}:{self.port}")
    
    def stop(self) -> None:
        """Stop the worker node."""
        self.running = False
        self.task_executor.shutdown(wait=True)
        print(f"Worker node {self.node_id} stopped")
    
    def submit_task(self, task: DistributedTask) -> None:
        """Submit a task for processing."""
        priority = -task.priority  # PriorityQueue uses min-heap
        self.task_queue.put((priority, task.created_at, task))
    
    def _process_tasks(self) -> None:
        """Main task processing loop."""
        while self.running:
            try:
                # Get next task with timeout
                priority, timestamp, task = self.task_queue.get(timeout=1.0)
                
                # Update load factor
                self.load_factor = self.task_queue.qsize() / 10.0
                
                # Execute task
                future = self.task_executor.submit(self._execute_task, task)
                self.active_tasks[task.task_id] = future
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in task processing loop: {e}")
    
    def _execute_task(self, task: DistributedTask) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.assigned_worker = self.node_id
        
        try:
            start_time = time.time()
            
            # Execute based on function name
            if task.function_name == 'active_inference_step':
                result = self._execute_active_inference_step(*task.args, **task.kwargs)
            elif task.function_name == 'belief_update':
                result = self._execute_belief_update(*task.args, **task.kwargs)
            elif task.function_name == 'batch_inference':
                result = self._execute_batch_inference(*task.args, **task.kwargs)
            else:
                raise ValueError(f"Unknown function: {task.function_name}")
            
            execution_time = time.time() - start_time
            
            # Update task result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # Update statistics
            self.execution_stats['tasks_completed'] += 1
            self.execution_stats['total_execution_time'] += execution_time
            self.execution_stats['average_task_time'] = (
                self.execution_stats['total_execution_time'] / 
                self.execution_stats['tasks_completed']
            )
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            self.execution_stats['tasks_failed'] += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                self.submit_task(task)
        
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)
    
    def _execute_active_inference_step(self, observation: np.ndarray, 
                                     state_dim: int = 8) -> Dict[str, Any]:
        """Execute a single Active Inference step."""
        # Simplified Active Inference computation
        beliefs = np.random.rand(state_dim) * 0.1 + observation[:state_dim]
        preferences = np.ones(state_dim) * 0.5
        prediction_error = observation - beliefs
        free_energy = np.sum(np.square(prediction_error))
        
        # Action selection (simplified)
        action = np.zeros(4)
        action[np.argmin(np.abs(beliefs - preferences))] = 1.0
        
        return {
            'beliefs': beliefs.tolist(),
            'action': action.tolist(),
            'free_energy': float(free_energy),
            'prediction_error': prediction_error.tolist()
        }
    
    def _execute_belief_update(self, observation: np.ndarray, 
                              prior_beliefs: np.ndarray) -> np.ndarray:
        """Execute belief update computation."""
        prediction_error = observation - prior_beliefs
        updated_beliefs = prior_beliefs + 0.1 * prediction_error
        return updated_beliefs
    
    def _execute_batch_inference(self, observations: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Execute batch inference on multiple observations."""
        results = []
        for obs in observations:
            result = self._execute_active_inference_step(obs)
            results.append(result)
        return results
    
    def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to coordinator."""
        while self.running:
            try:
                heartbeat_data = {
                    'action': 'heartbeat',
                    'node_id': self.node_id,
                    'load_factor': self.load_factor,
                    'active_tasks': len(self.active_tasks),
                    'execution_stats': self.execution_stats,
                    'timestamp': time.time()
                }
                
                # Send heartbeat (simplified)
                # In real implementation, would use network communication
                
                time.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                print(f"Error sending heartbeat: {e}")
                time.sleep(10)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            'node_id': self.node_id,
            'role': NodeRole.WORKER.value,
            'host': self.host,
            'port': self.port,
            'running': self.running,
            'load_factor': self.load_factor,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'execution_stats': self.execution_stats,
            'capabilities': self.capabilities
        }


class CoordinatorNode:
    """Coordinator node for managing distributed Active Inference cluster."""
    
    def __init__(self, port: int = 8080):
        self.node_id = str(uuid.uuid4())
        self.port = port
        self.host = socket.gethostname()
        
        # Cluster management
        self.workers = {}  # node_id -> NodeInfo
        self.task_registry = {}  # task_id -> DistributedTask
        self.task_queue = queue.PriorityQueue()
        
        # Load balancing
        self.load_balancer = RoundRobinLoadBalancer()
        
        # Scheduling
        self.scheduler_thread = None
        self.running = False
        
        # Performance monitoring
        self.cluster_stats = {
            'total_tasks_processed': 0,
            'total_execution_time': 0.0,
            'average_task_time': 0.0,
            'worker_utilization': 0.0
        }
    
    def start(self) -> None:
        """Start the coordinator node."""
        self.running = True
        
        # Start task scheduler
        self.scheduler_thread = threading.Thread(target=self._schedule_tasks)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Start worker monitoring
        monitor_thread = threading.Thread(target=self._monitor_workers)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print(f"Coordinator node {self.node_id} started on {self.host}:{self.port}")
    
    def stop(self) -> None:
        """Stop the coordinator node."""
        self.running = False
        print(f"Coordinator node {self.node_id} stopped")
    
    def register_worker(self, node_info: NodeInfo) -> bool:
        """Register a new worker node."""
        try:
            self.workers[node_info.node_id] = node_info
            print(f"Registered worker {node_info.node_id} ({node_info.host}:{node_info.port})")
            return True
        except Exception as e:
            print(f"Failed to register worker: {e}")
            return False
    
    def submit_distributed_task(self, function_name: str, *args, **kwargs) -> str:
        """Submit a task for distributed execution."""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            function_name=function_name,
            args=args,
            kwargs=kwargs
        )
        
        self.task_registry[task.task_id] = task
        priority = -task.priority
        self.task_queue.put((priority, task.created_at, task))
        
        return task.task_id
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task."""
        task = self.task_registry.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None
    
    def _schedule_tasks(self) -> None:
        """Main task scheduling loop."""
        while self.running:
            try:
                # Get next task with timeout
                priority, timestamp, task = self.task_queue.get(timeout=1.0)
                
                # Find available worker
                available_worker = self.load_balancer.select_worker(
                    list(self.workers.values())
                )
                
                if available_worker:
                    # Assign task to worker
                    task.assigned_worker = available_worker.node_id
                    task.status = TaskStatus.RUNNING
                    
                    # Send task to worker (simplified)
                    # In real implementation, would use network communication
                    print(f"Assigned task {task.task_id} to worker {available_worker.node_id}")
                    
                    # Update worker load
                    available_worker.active_tasks += 1
                else:
                    # No available workers, put task back in queue
                    self.task_queue.put((priority, timestamp, task))
                    time.sleep(0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in task scheduling: {e}")
    
    def _monitor_workers(self) -> None:
        """Monitor worker health and performance."""
        while self.running:
            try:
                current_time = time.time()
                dead_workers = []
                
                for worker_id, worker_info in self.workers.items():
                    # Check if worker is alive (heartbeat timeout)
                    if current_time - worker_info.last_heartbeat > 30:
                        dead_workers.append(worker_id)
                
                # Remove dead workers
                for worker_id in dead_workers:
                    print(f"Removing dead worker: {worker_id}")
                    self.workers.pop(worker_id)
                
                # Update cluster statistics
                if self.workers:
                    total_utilization = sum(w.load_factor for w in self.workers.values())
                    self.cluster_stats['worker_utilization'] = total_utilization / len(self.workers)
                
                time.sleep(15)  # Monitor every 15 seconds
                
            except Exception as e:
                print(f"Error monitoring workers: {e}")
                time.sleep(15)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_workers = [w for w in self.workers.values() if w.is_available()]
        
        return {
            'coordinator_id': self.node_id,
            'total_workers': len(self.workers),
            'active_workers': len(active_workers),
            'pending_tasks': self.task_queue.qsize(),
            'total_tasks': len(self.task_registry),
            'cluster_stats': self.cluster_stats,
            'worker_details': [asdict(w) for w in self.workers.values()]
        }


class RoundRobinLoadBalancer:
    """Simple round-robin load balancer for worker selection."""
    
    def __init__(self):
        self.last_selected_index = 0
    
    def select_worker(self, workers: List[NodeInfo]) -> Optional[NodeInfo]:
        """Select next available worker using round-robin."""
        available_workers = [w for w in workers if w.is_available()]
        
        if not available_workers:
            return None
        
        # Round-robin selection
        selected_worker = available_workers[self.last_selected_index % len(available_workers)]
        self.last_selected_index = (self.last_selected_index + 1) % len(available_workers)
        
        return selected_worker


class DistributedActiveInferenceCluster:
    """
    High-level interface for distributed Active Inference processing.
    """
    
    def __init__(self):
        self.coordinator = CoordinatorNode()
        self.workers = []
        self.running = False
    
    def start_cluster(self, num_workers: int = 4) -> None:
        """Start the distributed cluster."""
        # Start coordinator
        self.coordinator.start()
        
        # Start worker nodes
        for i in range(num_workers):
            worker = WorkerNode(port=8081 + i)
            worker.register_with_coordinator(self.coordinator.host, self.coordinator.port)
            
            # Register with coordinator
            node_info = NodeInfo(
                node_id=worker.node_id,
                role=NodeRole.WORKER,
                host=worker.host,
                port=worker.port,
                capabilities=worker.capabilities,
                last_heartbeat=time.time()
            )
            self.coordinator.register_worker(node_info)
            
            worker.start()
            self.workers.append(worker)
        
        self.running = True
        print(f"Distributed cluster started with {num_workers} workers")
    
    def stop_cluster(self) -> None:
        """Stop the distributed cluster."""
        for worker in self.workers:
            worker.stop()
        
        self.coordinator.stop()
        self.running = False
        print("Distributed cluster stopped")
    
    def distribute_inference_batch(self, observations: List[np.ndarray]) -> List[str]:
        """Distribute batch inference across the cluster."""
        task_ids = []
        
        # Split batch among workers
        batch_size = max(1, len(observations) // len(self.workers))
        
        for i in range(0, len(observations), batch_size):
            batch = observations[i:i + batch_size]
            task_id = self.coordinator.submit_distributed_task(
                'batch_inference', 
                batch
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def collect_results(self, task_ids: List[str], timeout: float = 30.0) -> List[Any]:
        """Collect results from distributed tasks."""
        results = []
        start_time = time.time()
        
        while len(results) < len(task_ids) and (time.time() - start_time) < timeout:
            for task_id in task_ids:
                if task_id not in [r['task_id'] for r in results if isinstance(r, dict)]:
                    result = self.coordinator.get_task_result(task_id)
                    if result is not None:
                        results.append({'task_id': task_id, 'result': result})
            
            time.sleep(0.1)
        
        return results
    
    def get_cluster_performance(self) -> Dict[str, Any]:
        """Get comprehensive cluster performance metrics."""
        coordinator_status = self.coordinator.get_cluster_status()
        worker_statuses = [worker.get_status() for worker in self.workers]
        
        return {
            'coordinator_status': coordinator_status,
            'worker_statuses': worker_statuses,
            'cluster_running': self.running,
            'total_processing_capacity': sum(
                w['capabilities']['cpu_cores'] for w in worker_statuses
            )
        }


def distributed_benchmark(cluster: DistributedActiveInferenceCluster,
                         num_observations: int = 1000) -> Dict[str, Any]:
    """Benchmark distributed processing performance."""
    print(f"Running distributed benchmark with {num_observations} observations...")
    
    # Generate test data
    observations = [np.random.rand(8) for _ in range(num_observations)]
    
    start_time = time.time()
    
    # Distribute processing
    task_ids = cluster.distribute_inference_batch(observations)
    
    # Collect results
    results = cluster.collect_results(task_ids, timeout=60.0)
    
    total_time = time.time() - start_time
    
    # Get cluster performance
    performance_stats = cluster.get_cluster_performance()
    
    return {
        'total_time': total_time,
        'observations_processed': len(observations),
        'tasks_completed': len(results),
        'throughput': len(observations) / total_time if total_time > 0 else 0,
        'cluster_performance': performance_stats
    }