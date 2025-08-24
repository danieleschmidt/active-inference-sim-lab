"""
Advanced Retry Mechanisms for Robust Active Inference Systems
Generation 2: MAKE IT ROBUST - Resilient Operation Patterns
"""

import time
import random
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union, Type
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"                    # Fixed delay
    LINEAR = "linear"                  # Linear backoff
    EXPONENTIAL = "exponential"        # Exponential backoff
    JITTERED_EXPONENTIAL = "jittered_exponential"  # Exponential with jitter
    FIBONACCI = "fibonacci"            # Fibonacci backoff


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 300.0
    jitter: bool = True
    exponential_base: float = 2.0
    timeout_per_attempt: Optional[float] = None
    exceptions_to_retry: tuple = (Exception,)
    exceptions_to_ignore: tuple = ()
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_final_failure: Optional[Callable[[Exception, int], None]] = None


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, attempts: int, last_exception: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class RetryMechanism:
    """
    Advanced retry mechanism with multiple strategies and comprehensive error handling.
    """
    
    def __init__(self, config: RetryConfig):
        """
        Initialize retry mechanism.
        
        Args:
            config: Retry configuration
        """
        self.config = config
        self._attempt_count = 0
        self._total_attempts = 0
        self._success_count = 0
        self._failure_count = 0
        self._last_attempt_time = 0.0
        self._fibonacci_cache = [1, 1]
        
        # Statistics
        self._retry_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_retry_attempts': 0,
            'avg_attempts_per_call': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetryStrategy}
        }
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        self._retry_stats['total_calls'] += 1
        self._retry_stats['strategy_usage'][self.config.strategy.value] += 1
        
        last_exception = None
        attempt = 0
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self._attempt_count = attempt
                self._total_attempts += 1
                self._last_attempt_time = time.time()
                
                # Execute with timeout if specified
                if self.config.timeout_per_attempt:
                    result = self._execute_with_timeout(func, self.config.timeout_per_attempt, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                self._success_count += 1
                self._retry_stats['successful_calls'] += 1
                self._retry_stats['total_retry_attempts'] += (attempt - 1)
                self._update_average_attempts()
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry_exception(e):
                    logger.debug(f"Not retrying exception: {e}")
                    raise
                
                # Check if we have more attempts
                if attempt >= self.config.max_attempts:
                    break
                
                # Calculate and apply delay
                delay = self._calculate_delay(attempt)
                
                # Call retry callback if provided
                if self.config.on_retry:
                    try:
                        self.config.on_retry(attempt, e, delay)
                    except Exception as callback_error:
                        logger.error(f"Retry callback failed: {callback_error}")
                
                logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        # All attempts failed
        self._failure_count += 1
        self._retry_stats['failed_calls'] += 1
        self._retry_stats['total_retry_attempts'] += (attempt - 1)
        self._update_average_attempts()
        
        # Call final failure callback if provided
        if self.config.on_final_failure:
            try:
                self.config.on_final_failure(last_exception, attempt)
            except Exception as callback_error:
                logger.error(f"Final failure callback failed: {callback_error}")
        
        error_msg = f"All {self.config.max_attempts} retry attempts failed"
        raise RetryExhaustedError(error_msg, attempt, last_exception)
    
    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        self._retry_stats['total_calls'] += 1
        self._retry_stats['strategy_usage'][self.config.strategy.value] += 1
        
        last_exception = None
        attempt = 0
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self._attempt_count = attempt
                self._total_attempts += 1
                self._last_attempt_time = time.time()
                
                # Execute with timeout if specified
                if self.config.timeout_per_attempt:
                    result = await asyncio.wait_for(func(*args, **kwargs), self.config.timeout_per_attempt)
                else:
                    result = await func(*args, **kwargs)
                
                # Success
                self._success_count += 1
                self._retry_stats['successful_calls'] += 1
                self._retry_stats['total_retry_attempts'] += (attempt - 1)
                self._update_average_attempts()
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry_exception(e):
                    logger.debug(f"Not retrying exception: {e}")
                    raise
                
                # Check if we have more attempts
                if attempt >= self.config.max_attempts:
                    break
                
                # Calculate and apply delay
                delay = self._calculate_delay(attempt)
                
                # Call retry callback if provided
                if self.config.on_retry:
                    try:
                        self.config.on_retry(attempt, e, delay)
                    except Exception as callback_error:
                        logger.error(f"Retry callback failed: {callback_error}")
                
                logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        # All attempts failed
        self._failure_count += 1
        self._retry_stats['failed_calls'] += 1
        self._retry_stats['total_retry_attempts'] += (attempt - 1)
        self._update_average_attempts()
        
        # Call final failure callback if provided
        if self.config.on_final_failure:
            try:
                self.config.on_final_failure(last_exception, attempt)
            except Exception as callback_error:
                logger.error(f"Final failure callback failed: {callback_error}")
        
        error_msg = f"All {self.config.max_attempts} retry attempts failed"
        raise RetryExhaustedError(error_msg, attempt, last_exception)
    
    def _execute_with_timeout(self, func: Callable[..., T], timeout: float, *args, **kwargs) -> T:
        """Execute function with timeout using thread pool."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Function execution timed out after {timeout}s")
    
    def _should_retry_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        # Don't retry if exception is in ignore list
        if isinstance(exception, self.config.exceptions_to_ignore):
            return False
        
        # Retry if exception is in retry list
        return isinstance(exception, self.config.exceptions_to_retry)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
            jitter = random.uniform(0.5, 1.5) if self.config.jitter else 1.0
            delay = base_delay * jitter
            
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            fib_value = self._get_fibonacci(attempt)
            delay = self.config.base_delay * fib_value
            
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled (except for jittered exponential which has its own)
        if self.config.jitter and self.config.strategy != RetryStrategy.JITTERED_EXPONENTIAL:
            jitter = random.uniform(0.1, 0.3)
            delay *= (1 + jitter)
        
        return delay
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number with caching."""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(
                self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            )
        return self._fibonacci_cache[n - 1] if n > 0 else 1
    
    def _update_average_attempts(self):
        """Update average attempts per call statistic."""
        total_calls = self._retry_stats['total_calls']
        if total_calls > 0:
            self._retry_stats['avg_attempts_per_call'] = (
                (self._retry_stats['successful_calls'] + self._retry_stats['failed_calls'] + 
                 self._retry_stats['total_retry_attempts']) / total_calls
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry mechanism statistics."""
        success_rate = (self._success_count / max(1, self._total_attempts)) * 100
        
        return {
            'total_executions': self._retry_stats['total_calls'],
            'successful_executions': self._retry_stats['successful_calls'],
            'failed_executions': self._retry_stats['failed_calls'],
            'total_attempts': self._total_attempts,
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'success_rate': success_rate,
            'avg_attempts_per_call': self._retry_stats['avg_attempts_per_call'],
            'total_retry_attempts': self._retry_stats['total_retry_attempts'],
            'last_attempt_time': self._last_attempt_time,
            'current_attempt': self._attempt_count,
            'strategy_usage': self._retry_stats['strategy_usage'].copy(),
            'config': {
                'max_attempts': self.config.max_attempts,
                'strategy': self.config.strategy.value,
                'base_delay': self.config.base_delay,
                'max_delay': self.config.max_delay,
                'jitter': self.config.jitter,
                'timeout_per_attempt': self.config.timeout_per_attempt
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self._attempt_count = 0
        self._total_attempts = 0
        self._success_count = 0
        self._failure_count = 0
        self._last_attempt_time = 0.0
        
        self._retry_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_retry_attempts': 0,
            'avg_attempts_per_call': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetryStrategy}
        }
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator interface."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        return wrapper


def retry(max_attempts: int = 3,
          strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
          base_delay: float = 1.0,
          max_delay: float = 300.0,
          jitter: bool = True,
          exponential_base: float = 2.0,
          timeout_per_attempt: Optional[float] = None,
          exceptions_to_retry: tuple = (Exception,),
          exceptions_to_ignore: tuple = (),
          on_retry: Optional[Callable[[int, Exception, float], None]] = None,
          on_final_failure: Optional[Callable[[Exception, int], None]] = None):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        strategy: Retry strategy to use
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        jitter: Whether to add jitter to delays
        exponential_base: Base for exponential backoff
        timeout_per_attempt: Timeout for each attempt
        exceptions_to_retry: Exceptions that should trigger retry
        exceptions_to_ignore: Exceptions to not retry
        on_retry: Callback called on each retry
        on_final_failure: Callback called when all retries fail
        
    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        strategy=strategy,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        exponential_base=exponential_base,
        timeout_per_attempt=timeout_per_attempt,
        exceptions_to_retry=exceptions_to_retry,
        exceptions_to_ignore=exceptions_to_ignore,
        on_retry=on_retry,
        on_final_failure=on_final_failure
    )
    
    def decorator(func):
        retry_mechanism = RetryMechanism(config)
        return retry_mechanism(func)
    
    return decorator


class GlobalRetryRegistry:
    """
    Global registry for retry mechanisms with monitoring and management.
    """
    
    def __init__(self):
        self._mechanisms: Dict[str, RetryMechanism] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, mechanism: RetryMechanism) -> None:
        """Register a retry mechanism."""
        with self._lock:
            self._mechanisms[name] = mechanism
        logger.info(f"Retry mechanism registered: {name}")
    
    def get(self, name: str) -> Optional[RetryMechanism]:
        """Get retry mechanism by name."""
        with self._lock:
            return self._mechanisms.get(name)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all registered mechanisms."""
        with self._lock:
            return {name: mechanism.get_statistics() 
                   for name, mechanism in self._mechanisms.items()}
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get aggregated global statistics."""
        all_stats = self.get_all_statistics()
        
        if not all_stats:
            return {'total_mechanisms': 0}
        
        # Aggregate statistics
        total_executions = sum(stats['total_executions'] for stats in all_stats.values())
        total_successes = sum(stats['successful_executions'] for stats in all_stats.values())
        total_failures = sum(stats['failed_executions'] for stats in all_stats.values())
        total_attempts = sum(stats['total_attempts'] for stats in all_stats.values())
        total_retries = sum(stats['total_retry_attempts'] for stats in all_stats.values())
        
        # Calculate rates
        global_success_rate = (total_successes / max(1, total_executions)) * 100
        global_retry_rate = (total_retries / max(1, total_attempts)) * 100
        
        # Strategy usage
        strategy_totals = {}
        for stats in all_stats.values():
            for strategy, count in stats['strategy_usage'].items():
                strategy_totals[strategy] = strategy_totals.get(strategy, 0) + count
        
        return {
            'total_mechanisms': len(self._mechanisms),
            'total_executions': total_executions,
            'successful_executions': total_successes,
            'failed_executions': total_failures,
            'total_attempts': total_attempts,
            'total_retry_attempts': total_retries,
            'global_success_rate': global_success_rate,
            'global_retry_rate': global_retry_rate,
            'strategy_distribution': strategy_totals,
            'individual_stats': all_stats
        }
    
    def reset_all_statistics(self):
        """Reset statistics for all mechanisms."""
        with self._lock:
            for mechanism in self._mechanisms.values():
                mechanism.reset_statistics()
        logger.info("All retry mechanism statistics reset")


# Global retry registry instance
retry_registry = GlobalRetryRegistry()


def smart_retry(name: str, 
                config: Optional[RetryConfig] = None,
                register_globally: bool = True):
    """
    Smart retry decorator with automatic registration and monitoring.
    
    Args:
        name: Unique name for the retry mechanism
        config: Retry configuration (uses defaults if None)
        register_globally: Whether to register in global registry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        if config is None:
            retry_config = RetryConfig()
        else:
            retry_config = config
        
        mechanism = RetryMechanism(retry_config)
        
        if register_globally:
            retry_registry.register(name, mechanism)
        
        return mechanism(func)
    
    return decorator