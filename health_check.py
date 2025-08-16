
import time
import json
from datetime import datetime
from active_inference import ActiveInferenceAgent
from active_inference.environments import MockEnvironment

def health_check():
    """Comprehensive health check endpoint."""
    start_time = time.perf_counter()
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'response_time_ms': 0
    }
    
    try:
        # Component health checks
        checks = [
            ('agent_creation', test_agent_creation),
            ('environment_interaction', test_environment),
            ('inference_pipeline', test_inference),
            ('memory_usage', test_memory)
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                health_status['checks'][check_name] = {
                    'status': 'pass' if result else 'fail',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                health_status['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Overall status
        failed_checks = sum(1 for check in health_status['checks'].values() 
                           if check['status'] != 'pass')
        
        if failed_checks == 0:
            health_status['status'] = 'healthy'
        elif failed_checks <= 1:
            health_status['status'] = 'degraded'
        else:
            health_status['status'] = 'unhealthy'
        
        health_status['response_time_ms'] = (time.perf_counter() - start_time) * 1000
        
    except Exception as e:
        health_status['status'] = 'error'
        health_status['error'] = str(e)
    
    return health_status

def test_agent_creation():
    """Test agent creation."""
    agent = ActiveInferenceAgent(
        state_dim=2, obs_dim=4, action_dim=2,
        agent_id="health_check_agent"
    )
    return agent is not None

def test_environment():
    """Test environment interaction."""
    env = MockEnvironment(obs_dim=4, action_dim=2)
    obs = env.reset()
    return obs is not None and len(obs) == 4

def test_inference():
    """Test inference pipeline."""
    agent = ActiveInferenceAgent(
        state_dim=2, obs_dim=4, action_dim=2,
        agent_id="inference_test"
    )
    obs = np.random.randn(4)
    action = agent.act(obs)
    return action is not None and len(action) == 2

def test_memory():
    """Test memory usage."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb < 500  # Less than 500MB

if __name__ == "__main__":
    import numpy as np
    result = health_check()
    print(json.dumps(result, indent=2))
