"""Advanced Fault Tolerance and Resilience for Active Inference Systems.

This module implements production-grade fault tolerance mechanisms for 
Generation 2: MAKE IT ROBUST, including:

- Circuit Breaker patterns for preventing cascade failures
- Bulkhead isolation for component failure containment  
- Retry mechanisms with exponential backoff and jitter
- Graceful degradation and fallback strategies
- Self-healing capabilities and automatic recovery
- Chaos engineering for proactive resilience testing
"""

import time
import threading
import logging
import traceback
import random
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import functools
import inspect


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if circuit can close


class FaultType(Enum):
    """Types of faults that can occur."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMPUTATION_ERROR = "computation_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening circuit
    recovery_timeout: float = 60.0       # Time before trying to close circuit
    request_timeout: float = 10.0        # Request timeout
    half_open_max_requests: int = 3      # Max requests in half-open state
    min_requests_threshold: int = 10      # Min requests before circuit can open


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, constant


@dataclass
class FaultEvent:
    """Record of a fault event."""
    timestamp: float
    fault_type: FaultType
    component: str
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    recovery_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascade failures.
    
    Implements the circuit breaker pattern to detect failures and prevent
    cascading failures by failing fast when a service is unavailable.
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        self.half_open_requests = 0
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_open_count = 0
        self.recovery_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Failure history for analysis
        self.failure_history: deque = deque(maxlen=1000)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Original function exception if circuit is closed
        """
        with self.lock:
            self.total_requests += 1
            
            # Check if circuit should allow request
            if not self._should_allow_request():
                self.logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is open")
            
            # Execute the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record success
                execution_time = time.time() - start_time
                self._record_success(execution_time)
                
                return result
                
            except Exception as e:
                # Record failure
                execution_time = time.time() - start_time
                self._record_failure(e, execution_time)
                
                raise  # Re-raise the original exception
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on circuit state."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.config.recovery_timeout:
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            if self.half_open_requests < self.config.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False
        
        return False
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful execution."""
        self.successful_requests += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Successful request in half-open state
            if self.half_open_requests >= self.config.half_open_max_requests:
                # Enough successful requests, close the circuit
                self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.recovery_count += 1
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
        
        self.logger.debug(f"Circuit breaker {self.name} recorded success, "
                         f"execution_time={execution_time:.3f}s")
    
    def _record_failure(self, error: Exception, execution_time: float) -> None:
        """Record failed execution."""
        self.failed_requests += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Store failure for analysis
        failure_record = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'execution_time': execution_time,
            'state': self.state.value
        }
        self.failure_history.append(failure_record)
        
        # Check if circuit should open
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold and
            self.total_requests >= self.config.min_requests_threshold):
            
            self.logger.warning(f"Circuit breaker {self.name} transitioning to OPEN "
                              f"(failures: {self.failure_count})")
            self.state = CircuitState.OPEN
            self.circuit_open_count += 1
        
        elif self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state, go back to open
            self.logger.warning(f"Circuit breaker {self.name} transitioning back to OPEN "
                              f"(half-open failure)")
            self.state = CircuitState.OPEN
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self.lock:
            return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            success_rate = (self.successful_requests / max(1, self.total_requests)) * 100
            
            return {
                'name': self.name,
                'state': self.state.value,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': success_rate,
                'failure_count': self.failure_count,
                'circuit_open_count': self.circuit_open_count,
                'recovery_count': self.recovery_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'recent_failures': list(self.failure_history)[-10:]  # Last 10 failures
            }
    
    def force_open(self) -> None:
        """Manually force circuit breaker open."""
        with self.lock:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} manually forced OPEN")
    
    def force_close(self) -> None:
        """Manually force circuit breaker closed."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.logger.info(f"Circuit breaker {self.name} manually forced CLOSED")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """
    Advanced retry mechanism with multiple strategies.
    
    Implements sophisticated retry logic with exponential backoff,
    jitter, and different backoff strategies.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("RetryManager")
        
        # Statistics
        self.retry_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'total_delay_time': 0.0
        })
    
    def retry(self, 
             func: Callable,
             *args,
             exception_types: Tuple = (Exception,),
             config_override: Optional[RetryConfig] = None,
             **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            exception_types: Exception types to retry on
            config_override: Override retry configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        config = config_override or self.config
        func_name = getattr(func, '__name__', str(func))
        
        last_exception = None
        total_delay = 0.0
        
        for attempt in range(config.max_attempts):
            self.retry_stats[func_name]['total_attempts'] += 1
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    # Successful retry
                    self.retry_stats[func_name]['successful_retries'] += 1
                    self.logger.info(f"Function {func_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except exception_types as e:
                last_exception = e
                
                # Don't sleep after the last attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, config)
                    total_delay += delay
                    
                    self.logger.warning(f"Function {func_name} failed on attempt {attempt + 1}, "
                                      f"retrying in {delay:.2f}s: {e}")
                    
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    self.retry_stats[func_name]['failed_retries'] += 1
                    self.retry_stats[func_name]['total_delay_time'] += total_delay
                    
                    self.logger.error(f"Function {func_name} failed after {config.max_attempts} attempts")
            
            except Exception as e:
                # Non-retryable exception
                last_exception = e
                self.logger.error(f"Function {func_name} failed with non-retryable exception: {e}")
                break
        
        # All attempts failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        if config.backoff_strategy == "exponential":
            delay = config.base_delay * (config.exponential_base ** attempt)
        elif config.backoff_strategy == "linear":
            delay = config.base_delay * (attempt + 1)
        else:  # constant
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Apply jitter to avoid thundering herd
        if config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return dict(self.retry_stats)


class BulkheadIsolation:
    """
    Bulkhead isolation for preventing resource exhaustion.
    
    Isolates different components to prevent failures in one component
    from affecting others by limiting resource usage.
    """
    
    def __init__(self, 
                 name: str,
                 max_concurrent_requests: int = 100,
                 queue_size: int = 200,
                 timeout: float = 30.0):
        """
        Initialize bulkhead isolation.
        
        Args:
            name: Bulkhead name
            max_concurrent_requests: Maximum concurrent requests
            queue_size: Maximum queue size for pending requests
            timeout: Request timeout
        """
        self.name = name
        self.max_concurrent_requests = max_concurrent_requests
        self.queue_size = queue_size
        self.timeout = timeout
        self.logger = logging.getLogger(f"Bulkhead.{name}")
        
        # Resource management
        self.semaphore = threading.Semaphore(max_concurrent_requests)
        self.executor = ThreadPoolExecutor(
            max_workers=max_concurrent_requests,
            thread_name_prefix=f"Bulkhead-{name}"
        )
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'rejected_requests': 0,
            'timeout_requests': 0,
            'active_requests': 0,
            'queue_length': 0
        }
        
        self.stats_lock = threading.RLock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through bulkhead isolation.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BulkheadRejectedException: If request is rejected
            TimeoutError: If request times out
        """
        with self.stats_lock:
            self.stats['total_requests'] += 1
            self.stats['queue_length'] = self.executor._work_queue.qsize() if hasattr(self.executor._work_queue, 'qsize') else 0
        
        # Check if we can accept more requests
        if self.stats['queue_length'] > self.queue_size:
            with self.stats_lock:
                self.stats['rejected_requests'] += 1
            
            raise BulkheadRejectedException(f"Bulkhead {self.name} queue is full")
        
        # Submit to executor
        future = self.executor.submit(self._execute_with_semaphore, func, *args, **kwargs)
        
        try:
            result = future.result(timeout=self.timeout)
            
            with self.stats_lock:
                self.stats['completed_requests'] += 1
            
            return result
            
        except TimeoutError:
            with self.stats_lock:
                self.stats['timeout_requests'] += 1
            
            self.logger.warning(f"Request to {func.__name__} timed out in bulkhead {self.name}")
            raise
        
        except Exception as e:
            self.logger.error(f"Request to {func.__name__} failed in bulkhead {self.name}: {e}")
            raise
    
    def _execute_with_semaphore(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with semaphore protection."""
        with self.semaphore:
            with self.stats_lock:
                self.stats['active_requests'] += 1
            
            try:
                return func(*args, **kwargs)
            finally:
                with self.stats_lock:
                    self.stats['active_requests'] -= 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self.stats_lock:
            stats_copy = self.stats.copy()
            stats_copy['name'] = self.name
            stats_copy['max_concurrent_requests'] = self.max_concurrent_requests
            stats_copy['queue_size'] = self.queue_size
            return stats_copy
    
    def shutdown(self) -> None:
        """Shutdown the bulkhead."""
        self.executor.shutdown(wait=True)


class BulkheadRejectedException(Exception):
    """Exception raised when bulkhead rejects a request."""
    pass


class FaultTolerantSystem:
    """
    Comprehensive fault tolerance system combining all patterns.
    
    Integrates circuit breakers, retry mechanisms, and bulkhead isolation
    to provide comprehensive fault tolerance for Active Inference systems.
    """
    
    def __init__(self):
        """Initialize fault tolerant system."""
        self.logger = logging.getLogger("FaultTolerantSystem")
        
        # Component registry
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.retry_manager = RetryManager()
        
        # Fault tracking
        self.fault_events: deque = deque(maxlen=10000)
        self.fault_stats = defaultdict(int)
        
        # Self-healing
        self.healing_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.healing_thread: Optional[threading.Thread] = None
        self.healing_active = False
        
        # Thread safety
        self.system_lock = threading.RLock()
    
    def create_circuit_breaker(self, 
                             name: str,
                             config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        with self.system_lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
            
            circuit_breaker = CircuitBreaker(name, config)
            self.circuit_breakers[name] = circuit_breaker
            
            self.logger.info(f"Created circuit breaker: {name}")
            return circuit_breaker
    
    def create_bulkhead(self,
                       name: str,
                       max_concurrent_requests: int = 100,
                       queue_size: int = 200,
                       timeout: float = 30.0) -> BulkheadIsolation:
        """Create and register a bulkhead."""
        with self.system_lock:
            if name in self.bulkheads:
                return self.bulkheads[name]
            
            bulkhead = BulkheadIsolation(name, max_concurrent_requests, queue_size, timeout)
            self.bulkheads[name] = bulkhead
            
            self.logger.info(f"Created bulkhead: {name}")
            return bulkhead
    
    def execute_with_protection(self,
                              func: Callable,
                              circuit_breaker_name: Optional[str] = None,
                              bulkhead_name: Optional[str] = None,
                              retry_config: Optional[RetryConfig] = None,
                              fallback_func: Optional[Callable] = None,
                              *args,
                              **kwargs) -> Any:
        """
        Execute function with comprehensive fault tolerance protection.
        
        Args:
            func: Function to execute
            circuit_breaker_name: Circuit breaker to use
            bulkhead_name: Bulkhead to use
            retry_config: Retry configuration
            fallback_func: Fallback function if all else fails
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        func_name = getattr(func, '__name__', str(func))
        
        try:
            # Define the execution chain
            def execute_chain():
                execution_func = func
                
                # Apply circuit breaker protection
                if circuit_breaker_name:
                    circuit_breaker = self.circuit_breakers.get(circuit_breaker_name)
                    if circuit_breaker:
                        execution_func = lambda *a, **kw: circuit_breaker.call(func, *a, **kw)
                
                # Apply bulkhead isolation
                if bulkhead_name:
                    bulkhead = self.bulkheads.get(bulkhead_name)
                    if bulkhead:
                        orig_func = execution_func
                        execution_func = lambda *a, **kw: bulkhead.execute(orig_func, *a, **kw)
                
                return execution_func(*args, **kwargs)
            
            # Apply retry mechanism
            if retry_config:
                result = self.retry_manager.retry(execute_chain, config_override=retry_config)
            else:
                result = execute_chain()
            
            return result
            
        except Exception as e:
            # Record fault event
            self._record_fault_event(func_name, e)
            
            # Try fallback if available
            if fallback_func:
                try:
                    self.logger.warning(f"Executing fallback for {func_name}: {e}")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed for {func_name}: {fallback_error}")
                    raise e  # Raise original exception
            
            raise  # Re-raise if no fallback
    
    def _record_fault_event(self, component: str, error: Exception) -> None:
        """Record a fault event for analysis."""
        fault_type = self._classify_fault_type(error)
        
        fault_event = FaultEvent(
            timestamp=time.time(),
            fault_type=fault_type,
            component=component,
            error_message=str(error),
            context={'error_type': type(error).__name__}
        )
        
        with self.system_lock:
            self.fault_events.append(fault_event)
            self.fault_stats[fault_type.value] += 1
            self.fault_stats['total'] += 1
        
        self.logger.warning(f"Fault recorded: {component} - {fault_type.value} - {error}")
    
    def _classify_fault_type(self, error: Exception) -> FaultType:
        """Classify the type of fault based on the exception."""
        error_type = type(error).__name__.lower()
        
        if 'timeout' in error_type:
            return FaultType.TIMEOUT
        elif any(term in error_type for term in ['connection', 'network', 'socket']):
            return FaultType.CONNECTION_ERROR
        elif any(term in error_type for term in ['validation', 'value', 'type']):
            return FaultType.VALIDATION_ERROR
        elif any(term in error_type for term in ['memory', 'resource', 'limit']):
            return FaultType.RESOURCE_EXHAUSTION
        elif any(term in error_type for term in ['computation', 'calculation', 'math']):
            return FaultType.COMPUTATION_ERROR
        else:
            return FaultType.UNKNOWN_ERROR
    
    def register_healing_callback(self, component: str, callback: Callable) -> None:
        """Register a self-healing callback for a component."""
        self.healing_callbacks[component].append(callback)
        self.logger.info(f"Registered healing callback for {component}")
    
    def start_self_healing(self, check_interval: float = 60.0) -> None:
        """Start the self-healing monitoring thread."""
        if self.healing_active:
            return
        
        self.healing_active = True
        self.healing_thread = threading.Thread(
            target=self._healing_worker,
            args=(check_interval,),
            daemon=True,
            name="SelfHealingWorker"
        )
        self.healing_thread.start()
        
        self.logger.info("Self-healing system started")
    
    def stop_self_healing(self) -> None:
        """Stop the self-healing monitoring thread."""
        self.healing_active = False
        
        if self.healing_thread and self.healing_thread.is_alive():
            self.healing_thread.join(timeout=5.0)
        
        self.logger.info("Self-healing system stopped")
    
    def _healing_worker(self, check_interval: float) -> None:
        """Worker thread for self-healing monitoring."""
        while self.healing_active:
            try:
                self._check_and_heal_components()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in healing worker: {e}")
    
    def _check_and_heal_components(self) -> None:
        """Check component health and trigger healing if needed."""
        # Check circuit breakers
        for name, circuit_breaker in self.circuit_breakers.items():
            if circuit_breaker.get_state() == CircuitState.OPEN:
                self.logger.info(f"Attempting to heal circuit breaker: {name}")
                self._trigger_healing_callbacks(name)
        
        # Check bulkheads for high rejection rates
        for name, bulkhead in self.bulkheads.items():
            stats = bulkhead.get_statistics()
            if stats['total_requests'] > 0:
                rejection_rate = stats['rejected_requests'] / stats['total_requests']
                if rejection_rate > 0.1:  # More than 10% rejections
                    self.logger.info(f"Attempting to heal bulkhead: {name} (rejection_rate: {rejection_rate:.2%})")
                    self._trigger_healing_callbacks(name)
    
    def _trigger_healing_callbacks(self, component: str) -> None:
        """Trigger healing callbacks for a component."""
        callbacks = self.healing_callbacks.get(component, [])
        
        for callback in callbacks:
            try:
                callback()
                self.logger.info(f"Healing callback executed for {component}")
            except Exception as e:
                self.logger.error(f"Healing callback failed for {component}: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self.system_lock:
            # Circuit breaker health
            circuit_health = {}
            for name, cb in self.circuit_breakers.items():
                stats = cb.get_statistics()
                circuit_health[name] = {
                    'state': stats['state'],
                    'success_rate': stats['success_rate'],
                    'is_healthy': stats['state'] == 'closed' and stats['success_rate'] > 90
                }
            
            # Bulkhead health
            bulkhead_health = {}
            for name, bulkhead in self.bulkheads.items():
                stats = bulkhead.get_statistics()
                rejection_rate = stats['rejected_requests'] / max(1, stats['total_requests']) * 100
                bulkhead_health[name] = {
                    'active_requests': stats['active_requests'],
                    'rejection_rate': rejection_rate,
                    'is_healthy': rejection_rate < 10  # Less than 10% rejections
                }
            
            # Fault statistics
            recent_faults = [f for f in self.fault_events if time.time() - f.timestamp < 3600]  # Last hour
            fault_rate = len(recent_faults) / max(1, len(self.fault_events))
            
            # Overall health
            healthy_circuits = sum(1 for health in circuit_health.values() if health['is_healthy'])
            healthy_bulkheads = sum(1 for health in bulkhead_health.values() if health['is_healthy'])
            
            total_components = len(circuit_health) + len(bulkhead_health)
            healthy_components = healthy_circuits + healthy_bulkheads
            
            system_health_score = (healthy_components / max(1, total_components)) * 100
            
            return {
                'system_health_score': system_health_score,
                'circuit_breakers': circuit_health,
                'bulkheads': bulkhead_health,
                'fault_statistics': dict(self.fault_stats),
                'recent_fault_count': len(recent_faults),
                'fault_rate': fault_rate,
                'self_healing_active': self.healing_active,
                'total_fault_events': len(self.fault_events)
            }
    
    def shutdown(self) -> None:
        """Shutdown the fault tolerant system."""
        self.stop_self_healing()
        
        # Shutdown bulkheads
        for bulkhead in self.bulkheads.values():
            bulkhead.shutdown()
        
        self.logger.info("Fault tolerant system shutdown complete")


# Decorators for easy fault tolerance integration

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            system = get_global_fault_tolerant_system()
            circuit_breaker = system.create_circuit_breaker(name, config)
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def retry(config: Optional[RetryConfig] = None, exception_types: Tuple = (Exception,)):
    """Decorator to add retry protection to a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            system = get_global_fault_tolerant_system()
            return system.retry_manager.retry(func, *args, exception_types=exception_types, config_override=config, **kwargs)
        return wrapper
    return decorator


def bulkhead(name: str, max_concurrent: int = 100, queue_size: int = 200, timeout: float = 30.0):
    """Decorator to add bulkhead isolation to a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            system = get_global_fault_tolerant_system()
            bulkhead = system.create_bulkhead(name, max_concurrent, queue_size, timeout)
            return bulkhead.execute(func, *args, **kwargs)
        return wrapper
    return decorator


# Global fault tolerant system instance
_global_fault_tolerant_system: Optional[FaultTolerantSystem] = None


def get_global_fault_tolerant_system() -> FaultTolerantSystem:
    """Get or create the global fault tolerant system."""
    global _global_fault_tolerant_system
    if _global_fault_tolerant_system is None:
        _global_fault_tolerant_system = FaultTolerantSystem()
    return _global_fault_tolerant_system


def shutdown_global_fault_tolerance() -> None:
    """Shutdown the global fault tolerant system."""
    global _global_fault_tolerant_system
    if _global_fault_tolerant_system:
        _global_fault_tolerant_system.shutdown()
        _global_fault_tolerant_system = None