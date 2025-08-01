# Load Testing Configuration for Active Inference Simulation Lab
# Tests system performance under various load conditions

from locust import HttpUser, task, between
import json
import random
import numpy as np

class ActiveInferenceUser(HttpUser):
    """Simulates users interacting with Active Inference API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session with agent creation."""
        self.agent_id = None
        self.create_agent()
    
    def create_agent(self):
        """Create an active inference agent."""
        agent_config = {
            "state_dim": random.choice([2, 4, 8]),
            "obs_dim": random.choice([4, 8, 16]),
            "action_dim": random.choice([1, 2, 4]),
            "inference_method": random.choice(["variational", "particle", "kalman"])
        }
        
        response = self.client.post(
            "/api/v1/agents",
            json=agent_config,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            self.agent_id = response.json().get("agent_id")
    
    @task(3)
    def inference_request(self):
        """Perform state inference - most common operation."""
        if not self.agent_id:
            self.create_agent()
            return
        
        # Generate random observations
        obs_dim = random.choice([4, 8, 16])
        observations = np.random.normal(0, 1, obs_dim).tolist()
        
        inference_data = {
            "agent_id": self.agent_id,
            "observations": observations,
            "prior_beliefs": {
                "mean": [0.0, 0.0],
                "variance": [1.0, 1.0]
            }
        }
        
        self.client.post(
            "/api/v1/inference",
            json=inference_data,
            headers={"Content-Type": "application/json"},
            name="State Inference"
        )
    
    @task(2) 
    def planning_request(self):
        """Perform action planning."""
        if not self.agent_id:
            self.create_agent()
            return
        
        planning_data = {
            "agent_id": self.agent_id,
            "current_beliefs": {
                "mean": np.random.normal(0, 1, 2).tolist(),
                "variance": np.random.uniform(0.1, 2.0, 2).tolist()
            },
            "horizon": random.choice([3, 5, 10]),
            "objective": "expected_free_energy"
        }
        
        self.client.post(
            "/api/v1/planning",
            json=planning_data,
            headers={"Content-Type": "application/json"},
            name="Action Planning"
        )
    
    @task(1)
    def model_update(self):
        """Update generative model - less frequent operation."""
        if not self.agent_id:
            self.create_agent()
            return
        
        update_data = {
            "agent_id": self.agent_id,
            "observations": np.random.normal(0, 1, 8).tolist(),
            "actions": np.random.normal(0, 0.5, 2).tolist(),
            "learning_rate": 0.01
        }
        
        self.client.post(
            "/api/v1/model/update",
            json=update_data,
            headers={"Content-Type": "application/json"},
            name="Model Update"
        )
    
    @task(1)
    def get_agent_status(self):
        """Get agent status and metrics."""
        if not self.agent_id:
            self.create_agent()
            return
        
        self.client.get(
            f"/api/v1/agents/{self.agent_id}/status",
            name="Agent Status"
        )

class HighLoadUser(ActiveInferenceUser):
    """Simulates high-frequency users for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Aggressive load pattern
    
    @task(5)
    def rapid_inference(self):
        """Rapid inference requests."""
        self.inference_request()

class BatchProcessingUser(HttpUser):
    """Simulates batch processing workloads."""
    
    wait_time = between(5, 10)  # Longer wait between batches
    
    @task
    def batch_inference(self):
        """Process batch of inference requests."""
        batch_size = random.choice([10, 25, 50])
        
        batch_data = {
            "requests": []
        }
        
        for _ in range(batch_size):
            batch_data["requests"].append({
                "observations": np.random.normal(0, 1, 8).tolist(),
                "agent_config": {
                    "state_dim": 4,
                    "obs_dim": 8,
                    "action_dim": 2
                }
            })
        
        self.client.post(
            "/api/v1/batch/inference",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            name="Batch Inference"
        )

# Custom load shapes for different testing scenarios
from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """Step-wise load increase for gradual stress testing."""
    
    step_time = 60  # 60 seconds per step
    step_load = 10  # 10 users per step
    spawn_rate = 2  # Spawn 2 users per second
    time_limit = 600  # 10 minutes total
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)

class SpikeLoadShape(LoadTestShape):
    """Spike load pattern for testing system resilience."""
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (10, 2)  # Baseline load
        elif run_time < 120:
            return (100, 10)  # Spike
        elif run_time < 180:
            return (10, 2)  # Return to baseline
        elif run_time < 240:
            return (200, 20)  # Larger spike
        elif run_time < 300:
            return (10, 2)  # Final baseline
        else:
            return None  # Stop test