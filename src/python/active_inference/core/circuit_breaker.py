"""
Circuit Breaker Pattern Implementation for Active Inference Systems
Generation 2: MAKE IT ROBUST - Fault Tolerance & Resilience
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures to trigger open
    success_threshold: int = 2           # Successes to close from half-open
    timeout: float = 30.0                # Seconds before trying half-open
    recovery_timeout: float = 60.0       # Seconds to wait before full recovery
    max_retries: int = 3                 # Max retries before giving up
    exponential_backoff: bool = True     # Use exponential backoff


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation providing fault tolerance and graceful degradation.
    
    Protects against cascading failures by monitoring failure rates and temporarily
    disabling failing operations while allowing the system to recover.
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None,
                 fallback_function: Optional[Callable[..., T]] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Configuration settings
            fallback_function: Optional fallback function to call when circuit is open
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_function = fallback_function
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.call_count = 0
        self.total_failures = 0
        self.total_successes = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._state_change_history = []
        self._recovery_attempts = 0
        
        logger.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator interface for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open and no fallback available
            Exception: Original function exceptions when circuit is closed
        """
        with self._lock:
            self.call_count += 1
            
            # Check if circuit should transition states
            self._check_state_transitions()
            
            # Handle open circuit
            if self.state == CircuitState.OPEN:
                if self.fallback_function:
                    logger.debug(f"Circuit '{self.name}' open, using fallback")
                    try:
                        return self.fallback_function(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Fallback function failed: {e}")
                        raise CircuitBreakerError(f"Circuit '{self.name}' open and fallback failed: {e}")
                else:
                    raise CircuitBreakerError(f"Circuit '{self.name}' is open")
            
            # Execute function (CLOSED or HALF_OPEN states)
            try:
                result = func(*args, **kwargs)
                self._record_success()
                # Check state transitions after success
                self._check_state_transitions()
                return result
                
            except Exception as e:
                self._record_failure(e)
                # Check state transitions after failure
                self._check_state_transitions()
                raise
    
    def _check_state_transitions(self):
        """Check and handle state transitions."""
        current_time = time.time()
        
        # Check if we should open (too many failures) - check first for immediate response
        if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self._transition_to_open("Failure threshold exceeded")
            return
            
        if self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if (current_time - self.last_failure_time) >= self.config.timeout:
                self._transition_to_half_open()
                
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close (enough successes)
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
            # Check if failure occurred - transition back to open immediately
            elif self.failure_count > 0:
                self._transition_to_open("Half-open failure")
            # Check recovery timeout
            elif (current_time - self.last_failure_time) >= self.config.recovery_timeout:
                self._transition_to_open("Recovery timeout exceeded")
    
    def _record_success(self):
        """Record successful function call."""
        self.success_count += 1
        self.total_successes += 1
        self.last_success_time = time.time()
        
        # Reset failure count on success in CLOSED state
        if self.state == CircuitState.CLOSED:
            self.failure_count = 0
        
        logger.debug(f"Circuit '{self.name}' success recorded (count: {self.success_count})")
    
    def _record_failure(self, exception: Exception):
        """Record failed function call."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        # Only reset success count in half-open state to prevent close->open loop
        if self.state == CircuitState.HALF_OPEN:
            self.success_count = 0
        
        logger.warning(f"Circuit '{self.name}' failure recorded: {exception} (count: {self.failure_count})")
    
    def _transition_to_open(self, reason: str = ""):
        """Transition circuit to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0
        
        self._record_state_change(old_state, self.state, reason)
        logger.warning(f"Circuit '{self.name}' transitioned to OPEN state: {reason}")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self._recovery_attempts += 1
        
        self._record_state_change(old_state, self.state, "Testing recovery")
        logger.info(f"Circuit '{self.name}' transitioned to HALF_OPEN state (attempt {self._recovery_attempts})")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        self._record_state_change(old_state, self.state, "Service recovered")
        logger.info(f"Circuit '{self.name}' transitioned to CLOSED state - service recovered")
    
    def _record_state_change(self, old_state: CircuitState, new_state: CircuitState, reason: str):
        """Record state change for monitoring."""
        change_record = {
            'timestamp': time.time(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'reason': reason,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        }
        
        self._state_change_history.append(change_record)
        
        # Keep only recent history (last 100 changes)
        if len(self._state_change_history) > 100:
            self._state_change_history = self._state_change_history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            current_time = time.time()
            
            # Calculate rates
            if self.call_count > 0:
                failure_rate = self.total_failures / self.call_count
                success_rate = self.total_successes / self.call_count
            else:
                failure_rate = 0.0
                success_rate = 0.0
            
            return {
                'name': self.name,
                'state': self.state.value,
                'call_count': self.call_count,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'current_failure_count': self.failure_count,
                'current_success_count': self.success_count,
                'failure_rate': failure_rate,
                'success_rate': success_rate,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'recovery_attempts': self._recovery_attempts,
                'time_since_last_failure': current_time - self.last_failure_time if self.last_failure_time > 0 else 0,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout,
                    'recovery_timeout': self.config.recovery_timeout,
                    'max_retries': self.config.max_retries,
                    'exponential_backoff': self.config.exponential_backoff
                },
                'state_changes': len(self._state_change_history)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring integration."""
        stats = self.get_statistics()
        
        # Determine health status
        if self.state == CircuitState.CLOSED:
            health_status = "healthy"
        elif self.state == CircuitState.HALF_OPEN:
            health_status = "warning"
        else:  # OPEN
            health_status = "critical"
        
        return {
            'health_status': health_status,
            'circuit_state': self.state.value,
            'failure_rate': stats['failure_rate'],
            'is_functional': self.state != CircuitState.OPEN or self.fallback_function is not None,
            'last_check': time.time()
        }
    
    def force_open(self, reason: str = "Manual override"):
        """Manually force circuit to OPEN state."""
        with self._lock:
            self._transition_to_open(reason)
            # Set last_failure_time to current time to prevent immediate half-open transition
            self.last_failure_time = time.time()
    
    def force_closed(self, reason: str = "Manual override"):
        """Manually force circuit to CLOSED state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self._record_state_change(old_state, self.state, reason)
            logger.info(f"Circuit '{self.name}' manually forced to CLOSED state: {reason}")
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.call_count = 0
            self.total_failures = 0
            self.total_successes = 0
            self.last_failure_time = 0.0
            self.last_success_time = 0.0
            self._recovery_attempts = 0
            self._state_change_history.clear()
            
            logger.info(f"Circuit '{self.name}' reset from {old_state.value} to CLOSED")
    
    def __repr__(self) -> str:
        """String representation of circuit breaker."""
        return (f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
                f"failures={self.failure_count}/{self.config.failure_threshold})")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        with self._lock:
            self._breakers[name] = breaker
        logger.info(f"Circuit breaker '{name}' registered")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_statistics() 
                   for name, breaker in self._breakers.items()}
    
    def get_global_health_status(self) -> Dict[str, Any]:
        """Get global health status across all circuit breakers."""
        with self._lock:
            if not self._breakers:
                return {'health_status': 'healthy', 'circuit_count': 0}
            
            all_stats = self.get_all_statistics()
            
            # Aggregate statistics
            total_calls = sum(stats['call_count'] for stats in all_stats.values())
            total_failures = sum(stats['total_failures'] for stats in all_stats.values())
            open_circuits = sum(1 for stats in all_stats.values() if stats['state'] == 'open')
            half_open_circuits = sum(1 for stats in all_stats.values() if stats['state'] == 'half_open')
            
            # Determine overall health
            if open_circuits > len(self._breakers) * 0.5:
                overall_health = 'critical'
            elif open_circuits > 0 or half_open_circuits > len(self._breakers) * 0.3:
                overall_health = 'degraded'
            elif half_open_circuits > 0:
                overall_health = 'warning'
            else:
                overall_health = 'healthy'
            
            return {
                'health_status': overall_health,
                'circuit_count': len(self._breakers),
                'open_circuits': open_circuits,
                'half_open_circuits': half_open_circuits,
                'total_calls': total_calls,
                'total_failures': total_failures,
                'global_failure_rate': total_failures / max(1, total_calls),
                'individual_stats': all_stats
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
        logger.info("All circuit breakers reset")


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None,
                   fallback: Optional[Callable] = None,
                   register_globally: bool = True):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        name: Unique name for the circuit breaker
        config: Circuit breaker configuration
        fallback: Fallback function to use when circuit is open
        register_globally: Whether to register in global registry
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func):
        breaker = CircuitBreaker(name=name, config=config, fallback_function=fallback)
        
        if register_globally:
            circuit_registry.register(name, breaker)
        
        return breaker(func)
    
    return decorator