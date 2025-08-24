"""
Comprehensive unit tests for Circuit Breaker functionality.
Generation 4: Quality Gates - Robust Testing Suite
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from typing import List

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src" / "python"))

from active_inference.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    CircuitBreakerError, circuit_registry, circuit_breaker
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig data class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 30.0
        assert config.recovery_timeout == 60.0
        assert config.max_retries == 3
        assert config.exponential_backoff is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=15.0,
            recovery_timeout=30.0,
            max_retries=5,
            exponential_backoff=False
        )
        
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout == 15.0
        assert config.recovery_timeout == 30.0
        assert config.max_retries == 5
        assert config.exponential_backoff is False


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.1,  # Fast timeout for testing
            recovery_timeout=0.2
        )
        self.circuit_breaker = CircuitBreaker("test_circuit", self.config)
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
    
    def test_successful_execution(self):
        """Test successful function execution."""
        def successful_function():
            return "success"
        
        result = self.circuit_breaker.call(successful_function)
        
        assert result == "success"
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.success_count == 1
        assert self.circuit_breaker.failure_count == 0
    
    def test_failed_execution(self):
        """Test failed function execution."""
        def failing_function():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            self.circuit_breaker.call(failing_function)
        
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 1
        assert self.circuit_breaker.success_count == 0
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after failure threshold reached."""
        def failing_function():
            raise RuntimeError("failure")
        
        # Execute failures up to threshold
        for i in range(self.config.failure_threshold):
            with pytest.raises(RuntimeError):
                self.circuit_breaker.call(failing_function)
            
            if i < self.config.failure_threshold - 1:
                assert self.circuit_breaker.state == CircuitState.CLOSED
            else:
                assert self.circuit_breaker.state == CircuitState.OPEN
    
    def test_circuit_fails_fast_when_open(self):
        """Test circuit fails fast when in OPEN state."""
        # Force circuit to OPEN state
        self.circuit_breaker.force_open("test")
        
        def any_function():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerError, match="Circuit 'test_circuit' is open"):
            self.circuit_breaker.call(any_function)
    
    def test_fallback_function_when_open(self):
        """Test fallback function is called when circuit is open."""
        def fallback():
            return "fallback result"
        
        circuit_with_fallback = CircuitBreaker("test_with_fallback", self.config, fallback)
        circuit_with_fallback.force_open("test")
        
        def main_function():
            return "main result"
        
        result = circuit_with_fallback.call(main_function)
        assert result == "fallback result"
    
    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        # Force circuit to OPEN state
        self.circuit_breaker.force_open("test")
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(self.config.timeout + 0.05)
        
        def test_function():
            return "test"
        
        # This should transition to HALF_OPEN
        result = self.circuit_breaker.call(test_function)
        
        assert result == "test"
        assert self.circuit_breaker.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_after_successful_half_open(self):
        """Test circuit closes after successful executions in HALF_OPEN state."""
        # Put circuit in HALF_OPEN state
        self.circuit_breaker._transition_to_half_open()
        
        def successful_function():
            return "success"
        
        # Execute successful calls up to success threshold
        for i in range(self.config.success_threshold):
            result = self.circuit_breaker.call(successful_function)
            assert result == "success"
        
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.success_count == 0  # Reset after closing
        assert self.circuit_breaker.failure_count == 0  # Reset after closing
    
    def test_circuit_opens_on_half_open_failure(self):
        """Test circuit opens again on failure in HALF_OPEN state."""
        # Put circuit in HALF_OPEN state
        self.circuit_breaker._transition_to_half_open()
        
        def failing_function():
            raise Exception("half-open failure")
        
        with pytest.raises(Exception, match="half-open failure"):
            self.circuit_breaker.call(failing_function)
        
        assert self.circuit_breaker.state == CircuitState.OPEN
    
    def test_decorator_interface(self):
        """Test circuit breaker as a decorator."""
        @self.circuit_breaker
        def decorated_function(x, y):
            if x < 0:
                raise ValueError("negative input")
            return x + y
        
        # Test successful call
        result = decorated_function(1, 2)
        assert result == 3
        
        # Test failed call
        with pytest.raises(ValueError, match="negative input"):
            decorated_function(-1, 2)
    
    def test_statistics(self):
        """Test circuit breaker statistics."""
        def test_function():
            return "ok"
        
        def failing_function():
            raise Exception("fail")
        
        # Execute some calls
        self.circuit_breaker.call(test_function)
        self.circuit_breaker.call(test_function)
        
        try:
            self.circuit_breaker.call(failing_function)
        except Exception:
            pass
        
        stats = self.circuit_breaker.get_statistics()
        
        assert stats['name'] == "test_circuit"
        assert stats['state'] == CircuitState.CLOSED.value
        assert stats['call_count'] == 3
        assert stats['total_successes'] == 2
        assert stats['total_failures'] == 1
        assert stats['success_rate'] == 2/3
        assert stats['failure_rate'] == 1/3
        assert stats['current_success_count'] == 2
        assert stats['current_failure_count'] == 1
        assert 'config' in stats
        assert 'time_since_last_failure' in stats
    
    def test_health_status(self):
        """Test health status reporting."""
        # Test healthy state
        health = self.circuit_breaker.get_health_status()
        assert health['health_status'] == "healthy"
        assert health['circuit_state'] == CircuitState.CLOSED.value
        assert health['is_functional'] is True
        
        # Test open state
        self.circuit_breaker.force_open("test")
        health = self.circuit_breaker.get_health_status()
        assert health['health_status'] == "critical"
        assert health['circuit_state'] == CircuitState.OPEN.value
        assert health['is_functional'] is False  # No fallback
        
        # Test half-open state
        self.circuit_breaker._transition_to_half_open()
        health = self.circuit_breaker.get_health_status()
        assert health['health_status'] == "warning"
        assert health['circuit_state'] == CircuitState.HALF_OPEN.value
    
    def test_manual_state_control(self):
        """Test manual circuit state control."""
        # Test force open
        self.circuit_breaker.force_open("manual test")
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        # Test force closed
        self.circuit_breaker.force_closed("manual reset")
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
    
    def test_reset(self):
        """Test circuit breaker reset functionality."""
        # Generate some activity
        self.circuit_breaker.call(lambda: "ok")
        try:
            self.circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass
        
        # Reset
        self.circuit_breaker.reset()
        
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.call_count == 0
        assert self.circuit_breaker.total_failures == 0
        assert self.circuit_breaker.total_successes == 0
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
    
    def test_thread_safety(self):
        """Test thread safety of circuit breaker."""
        results = []
        errors = []
        
        def concurrent_function(thread_id):
            try:
                if thread_id % 3 == 0:  # Every 3rd thread fails
                    raise Exception(f"fail-{thread_id}")
                else:
                    return f"success-{thread_id}"
            except Exception as e:
                raise
        
        def worker(thread_id):
            try:
                result = self.circuit_breaker.call(concurrent_function, thread_id)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(30)]
            for future in futures:
                future.result()  # Wait for completion
        
        # Verify results
        assert len(results) + len(errors) == 30
        assert len(results) > 0  # Some should succeed
        assert len(errors) > 0   # Some should fail
        
        # Verify statistics are consistent
        stats = self.circuit_breaker.get_statistics()
        assert stats['call_count'] == 30
        assert stats['total_successes'] + stats['total_failures'] == 30


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        circuit_registry._breakers.clear()
    
    def test_register_and_get_breaker(self):
        """Test registering and retrieving circuit breakers."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test_breaker", config)
        
        circuit_registry.register("test_breaker", breaker)
        
        retrieved = circuit_registry.get("test_breaker")
        assert retrieved is breaker
        assert retrieved.config.failure_threshold == 2
    
    def test_get_nonexistent_breaker(self):
        """Test getting non-existent circuit breaker."""
        result = circuit_registry.get("nonexistent")
        assert result is None
    
    def test_get_all_statistics(self):
        """Test getting statistics for all registered breakers."""
        # Register multiple breakers
        for i in range(3):
            config = CircuitBreakerConfig(failure_threshold=i+1)
            breaker = CircuitBreaker(f"breaker_{i}", config)
            circuit_registry.register(f"breaker_{i}", breaker)
        
        all_stats = circuit_registry.get_all_statistics()
        
        assert len(all_stats) == 3
        assert "breaker_0" in all_stats
        assert "breaker_1" in all_stats
        assert "breaker_2" in all_stats
        
        for name, stats in all_stats.items():
            assert 'name' in stats
            assert 'state' in stats
            assert 'call_count' in stats
    
    def test_global_health_status(self):
        """Test global health status across all breakers."""
        # Test with no breakers
        health = circuit_registry.get_global_health_status()
        assert health['health_status'] == 'healthy'
        assert health['circuit_count'] == 0
        
        # Add healthy breakers
        for i in range(3):
            breaker = CircuitBreaker(f"healthy_{i}")
            circuit_registry.register(f"healthy_{i}", breaker)
        
        health = circuit_registry.get_global_health_status()
        assert health['health_status'] == 'healthy'
        assert health['circuit_count'] == 3
        assert health['open_circuits'] == 0
        
        # Force one breaker open
        circuit_registry.get("healthy_0").force_open("test")
        
        health = circuit_registry.get_global_health_status()
        assert health['health_status'] == 'degraded'  # >0 open circuits
        assert health['open_circuits'] == 1
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        # Register and activate some breakers
        for i in range(2):
            breaker = CircuitBreaker(f"reset_test_{i}")
            circuit_registry.register(f"reset_test_{i}", breaker)
            
            # Generate some activity
            breaker.call(lambda: "ok")
        
        # Reset all
        circuit_registry.reset_all()
        
        # Verify all are reset
        for name, breaker in circuit_registry._breakers.items():
            stats = breaker.get_statistics()
            assert stats['call_count'] == 0
            assert stats['total_failures'] == 0
            assert stats['total_successes'] == 0


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""
    
    def test_basic_decorator(self):
        """Test basic decorator usage."""
        call_count = 0
        
        @circuit_breaker(name="decorator_test", register_globally=False)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            if x < 0:
                raise ValueError("negative")
            return x * 2
        
        # Test successful calls
        assert test_function(5) == 10
        assert test_function(3) == 6
        assert call_count == 2
        
        # Test failed call
        with pytest.raises(ValueError, match="negative"):
            test_function(-1)
        assert call_count == 3
    
    def test_decorator_with_config(self):
        """Test decorator with custom configuration."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        
        @circuit_breaker(name="config_test", config=config, register_globally=False)
        def failing_function():
            raise RuntimeError("always fails")
        
        # Fail enough times to open circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                failing_function()
        
        # Should now fail fast
        with pytest.raises(CircuitBreakerError, match="config_test"):
            failing_function()
    
    def test_decorator_with_fallback(self):
        """Test decorator with fallback function."""
        def fallback_func(x):
            return f"fallback: {x}"
        
        @circuit_breaker(
            name="fallback_test", 
            fallback=fallback_func, 
            register_globally=False
        )
        def main_function(x):
            if x == "fail":
                raise Exception("forced failure")
            return f"main: {x}"
        
        # Test normal operation
        assert main_function("normal") == "main: normal"
        
        # Force circuit open
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        
        @circuit_breaker(
            name="fallback_test2", 
            config=config,
            fallback=fallback_func, 
            register_globally=False
        )
        def failing_function(x):
            raise Exception("always fails")
        
        # Fail to open circuit
        with pytest.raises(Exception):
            failing_function("test")
        
        # Should use fallback
        result = failing_function("test")
        assert result == "fallback: test"
    
    def test_global_registration(self):
        """Test global registration of decorated functions."""
        circuit_registry._breakers.clear()
        
        @circuit_breaker(name="global_test", register_globally=True)
        def global_function():
            return "global"
        
        # Verify it's registered globally
        registered_breaker = circuit_registry.get("global_test")
        assert registered_breaker is not None
        
        # Verify function works
        assert global_function() == "global"
        
        # Verify statistics are tracked
        stats = registered_breaker.get_statistics()
        assert stats['call_count'] == 1
        assert stats['total_successes'] == 1


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker in realistic scenarios."""
    
    def test_database_connection_simulation(self):
        """Simulate database connection with circuit breaker."""
        connection_failures = 0
        max_failures = 3
        
        def simulate_db_query():
            nonlocal connection_failures
            if connection_failures < max_failures:
                connection_failures += 1
                raise ConnectionError(f"DB connection failed #{connection_failures}")
            return "query result"
        
        def db_fallback():
            return "cached result"
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.05
        )
        
        db_circuit = CircuitBreaker("db_circuit", config, db_fallback)
        
        # First two calls should fail and open circuit
        with pytest.raises(ConnectionError):
            db_circuit.call(simulate_db_query)
        
        with pytest.raises(ConnectionError):
            db_circuit.call(simulate_db_query)
        
        assert db_circuit.state == CircuitState.OPEN
        
        # Next call should use fallback
        result = db_circuit.call(simulate_db_query)
        assert result == "cached result"
        
        # Wait for timeout and recovery
        time.sleep(0.1)
        
        # Should transition to half-open and eventually succeed
        result = db_circuit.call(simulate_db_query)
        assert result == "query result"
        assert db_circuit.state == CircuitState.CLOSED
    
    def test_api_rate_limiting_scenario(self):
        """Test circuit breaker with API rate limiting scenario."""
        api_calls = 0
        rate_limit = 5
        
        class RateLimitError(Exception):
            pass
        
        def api_call():
            nonlocal api_calls
            api_calls += 1
            if api_calls <= rate_limit:
                return f"API response #{api_calls}"
            else:
                raise RateLimitError("Rate limit exceeded")
        
        def api_fallback():
            return "Rate limited - using cached data"
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=0.1
        )
        
        api_circuit = CircuitBreaker("api_circuit", config, api_fallback)
        
        # Make successful calls up to rate limit
        for i in range(rate_limit):
            result = api_circuit.call(api_call)
            assert f"API response #{i+1}" in result
        
        # Next calls should fail until circuit opens
        for _ in range(3):
            with pytest.raises(RateLimitError):
                api_circuit.call(api_call)
        
        assert api_circuit.state == CircuitState.OPEN
        
        # Should use fallback
        result = api_circuit.call(api_call)
        assert result == "Rate limited - using cached data"
    
    def test_microservice_communication(self):
        """Test circuit breaker for microservice communication."""
        service_health = {"healthy": True}
        
        def call_microservice():
            if not service_health["healthy"]:
                raise ConnectionError("Service unavailable")
            return {"data": "service response"}
        
        def service_fallback():
            return {"data": "default response", "source": "fallback"}
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.05,
            recovery_timeout=0.1
        )
        
        service_circuit = CircuitBreaker("service_circuit", config, service_fallback)
        
        # Test normal operation
        result = service_circuit.call(call_microservice)
        assert result["data"] == "service response"
        
        # Simulate service going down
        service_health["healthy"] = False
        
        # Fail enough times to open circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                service_circuit.call(call_microservice)
        
        assert service_circuit.state == CircuitState.OPEN
        
        # Should use fallback
        result = service_circuit.call(call_microservice)
        assert result["source"] == "fallback"
        
        # Simulate service recovery
        service_health["healthy"] = True
        
        # Wait for circuit to attempt recovery
        time.sleep(0.1)
        
        # Should recover to normal operation
        result = service_circuit.call(call_microservice)
        assert result["data"] == "service response"
        assert "source" not in result
        assert service_circuit.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__])