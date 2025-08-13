"""Advanced Validation Framework for Robust Active Inference Systems.

This module implements comprehensive validation, error handling, and monitoring
capabilities for production-grade Active Inference deployments:

- Multi-level input validation with semantic checking
- Real-time health monitoring and anomaly detection
- Graceful error recovery and fallback mechanisms
- Performance monitoring and resource management
- Security validation and threat detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
import logging
import time
import threading
from queue import Queue, Empty
from abc import ABC, abstractmethod
import psutil
import hashlib
import json
from pathlib import Path
from functools import wraps
import traceback
from collections import deque, defaultdict
import warnings
from contextlib import contextmanager


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    confidence_score: float = 1.0


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    error_rate: float
    response_time: float
    throughput: float
    timestamp: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security-related event."""
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    source: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationException(Exception):
    """Base exception for validation errors."""
    def __init__(self, message: str, validation_result: ValidationResult = None):
        super().__init__(message)
        self.validation_result = validation_result


class SecurityException(Exception):
    """Exception for security-related issues."""
    def __init__(self, message: str, security_event: SecurityEvent = None):
        super().__init__(message)
        self.security_event = security_event


class PerformanceException(Exception):
    """Exception for performance-related issues."""
    pass


class AdvancedValidator:
    """Advanced validation system with multi-level checks."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.logger = logging.getLogger("AdvancedValidator")
        self.validation_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(int)
        
        # Validation rules
        self.rules = {
            'array_validation': self._validate_array_advanced,
            'model_validation': self._validate_model_parameters,
            'agent_validation': self._validate_agent_state,
            'environment_validation': self._validate_environment_compatibility,
            'performance_validation': self._validate_performance_requirements
        }
    
    def validate_comprehensive(self, 
                             data: Any, 
                             validation_type: str,
                             context: Dict[str, Any] = None) -> ValidationResult:
        """Perform comprehensive validation with multiple checks."""
        start_time = time.time()
        context = context or {}
        
        result = ValidationResult(is_valid=True)
        
        try:
            # Type-specific validation
            if validation_type in self.rules:
                type_result = self.rules[validation_type](data, context)
                self._merge_validation_results(result, type_result)
            
            # Semantic validation
            semantic_result = self._validate_semantic_consistency(data, validation_type, context)
            self._merge_validation_results(result, semantic_result)
            
            # Security validation
            security_result = self._validate_security_constraints(data, context)
            self._merge_validation_results(result, security_result)
            
            # Performance validation
            perf_result = self._validate_performance_impact(data, context)
            self._merge_validation_results(result, perf_result)
            
            # Compute overall confidence
            result.confidence_score = self._compute_validation_confidence(result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation system error: {str(e)}")
            result.confidence_score = 0.0
            self.logger.error(f"Validation system error: {e}")
        
        result.validation_time = time.time() - start_time
        
        # Record validation history
        self.validation_history.append({
            'timestamp': time.time(),
            'validation_type': validation_type,
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'validation_time': result.validation_time
        })
        
        # Update error patterns
        for error in result.errors:
            self.error_patterns[error[:50]] += 1  # First 50 chars
        
        # Raise exception in strict mode
        if self.strict_mode and not result.is_valid:
            raise ValidationException(f"Validation failed: {result.errors}", result)
        
        return result
    
    def _validate_array_advanced(self, array: np.ndarray, context: Dict) -> ValidationResult:
        """Advanced array validation with statistical checks."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic array checks
            if not isinstance(array, np.ndarray):
                result.is_valid = False
                result.errors.append(f"Expected numpy array, got {type(array)}")
                return result
            
            # Shape validation
            expected_shape = context.get('expected_shape')
            if expected_shape and array.shape != expected_shape:
                result.is_valid = False
                result.errors.append(f"Shape mismatch: expected {expected_shape}, got {array.shape}")
            
            # Data type validation
            expected_dtype = context.get('expected_dtype')
            if expected_dtype and array.dtype != expected_dtype:
                result.warnings.append(f"Dtype mismatch: expected {expected_dtype}, got {array.dtype}")
            
            # Value range validation
            min_val = context.get('min_value')
            max_val = context.get('max_value')
            
            if min_val is not None and np.any(array < min_val):
                result.errors.append(f"Values below minimum {min_val}: {np.sum(array < min_val)} elements")
                result.is_valid = False
            
            if max_val is not None and np.any(array > max_val):
                result.errors.append(f"Values above maximum {max_val}: {np.sum(array > max_val)} elements")
                result.is_valid = False
            
            # Statistical validation
            if array.size > 0:
                # Check for infinite or NaN values
                if not np.isfinite(array).all():
                    nan_count = np.isnan(array).sum()
                    inf_count = np.isinf(array).sum()
                    result.errors.append(f"Non-finite values: {nan_count} NaN, {inf_count} infinite")
                    result.is_valid = False
                
                # Statistical anomaly detection
                if array.size > 3:
                    mean = np.mean(array)
                    std = np.std(array)
                    
                    # Z-score outlier detection
                    if std > 0:
                        z_scores = np.abs((array - mean) / std)
                        outliers = np.sum(z_scores > 3)
                        if outliers > array.size * 0.1:  # More than 10% outliers
                            result.warnings.append(f"High outlier rate: {outliers}/{array.size} ({outliers/array.size:.1%})")
                    
                    # Distribution checks
                    skewness = self._compute_skewness(array)
                    if abs(skewness) > 2:
                        result.warnings.append(f"High skewness: {skewness:.2f} (may indicate data issues)")
                    
                    # Entropy check for information content
                    if array.size > 10:
                        entropy = self._compute_entropy(array)
                        result.metadata['entropy'] = entropy
                        if entropy < 0.1:
                            result.warnings.append(f"Low entropy: {entropy:.3f} (data may be too uniform)")
            
            # Memory usage validation
            memory_mb = array.nbytes / (1024 * 1024)
            max_memory = context.get('max_memory_mb', 1000)  # 1GB default
            if memory_mb > max_memory:
                result.warnings.append(f"Large memory usage: {memory_mb:.1f}MB > {max_memory}MB")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Array validation error: {str(e)}")
        
        return result
    
    def _validate_model_parameters(self, params: Dict, context: Dict) -> ValidationResult:
        """Validate model parameters for consistency and safety."""
        result = ValidationResult(is_valid=True)
        
        try:
            if not isinstance(params, dict):
                result.is_valid = False
                result.errors.append(f"Expected dictionary, got {type(params)}")
                return result
            
            # Check required parameters
            required_params = context.get('required_params', [])
            for param in required_params:
                if param not in params:
                    result.errors.append(f"Missing required parameter: {param}")
                    result.is_valid = False
            
            # Validate parameter ranges
            param_ranges = context.get('param_ranges', {})
            for param, value in params.items():
                if param in param_ranges:
                    min_val, max_val = param_ranges[param]
                    if not (min_val <= value <= max_val):
                        result.errors.append(f"Parameter {param}={value} outside range [{min_val}, {max_val}]")
                        result.is_valid = False
            
            # Check for parameter interactions
            interactions = context.get('parameter_interactions', [])
            for interaction in interactions:
                if not self._check_parameter_interaction(params, interaction):
                    result.warnings.append(f"Parameter interaction warning: {interaction['description']}")
            
            # Validate parameter stability
            if 'learning_rate' in params and 'temperature' in params:
                lr = params['learning_rate']
                temp = params['temperature']
                
                # High learning rate + low temperature can cause instability
                if lr > 0.1 and temp < 0.1:
                    result.warnings.append("High learning rate with low temperature may cause instability")
                
                # Very low learning rate may prevent learning
                if lr < 1e-6:
                    result.warnings.append(f"Very low learning rate ({lr}) may prevent effective learning")
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Model parameter validation error: {str(e)}")
        
        return result
    
    def _validate_agent_state(self, agent_state: Dict, context: Dict) -> ValidationResult:
        """Validate agent state for consistency and health."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check agent health indicators
            error_rate = agent_state.get('error_rate', 0)
            if error_rate > 0.1:  # More than 10% errors
                result.warnings.append(f"High agent error rate: {error_rate:.1%}")
            
            # Check belief consistency
            beliefs = agent_state.get('beliefs', {})
            if beliefs:
                for belief_name, belief_data in beliefs.items():
                    if 'variance' in belief_data:
                        variance = belief_data['variance']
                        if np.any(variance < 0):
                            result.errors.append(f"Negative variance in belief '{belief_name}'")
                            result.is_valid = False
                        
                        # Check for collapsed beliefs (very low variance)
                        if np.any(variance < 1e-8):
                            result.warnings.append(f"Very low variance in belief '{belief_name}' (may indicate collapse)")
            
            # Check temporal consistency
            step_count = agent_state.get('step_count', 0)
            episode_count = agent_state.get('episode_count', 0)
            if step_count > 0 and episode_count == 0:
                result.warnings.append("Steps taken but no episodes recorded")
            
            # Check learning progress
            total_reward = agent_state.get('total_reward', 0)
            if step_count > 1000 and total_reward == 0:
                result.warnings.append("No reward accumulated after many steps (learning may be ineffective)")
            
            # Memory usage check
            history_length = agent_state.get('history_length', 0)
            max_history = context.get('max_history_length', 10000)
            if history_length > max_history * 0.9:
                result.warnings.append(f"History length {history_length} approaching maximum {max_history}")
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Agent state validation error: {str(e)}")
        
        return result
    
    def _validate_environment_compatibility(self, env_info: Dict, context: Dict) -> ValidationResult:
        """Validate environment compatibility and configuration."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check dimensions compatibility
            obs_dim = env_info.get('observation_dim')
            action_dim = env_info.get('action_dim')
            agent_obs_dim = context.get('agent_obs_dim')
            agent_action_dim = context.get('agent_action_dim')
            
            if obs_dim and agent_obs_dim and obs_dim != agent_obs_dim:
                result.errors.append(f"Observation dimension mismatch: env={obs_dim}, agent={agent_obs_dim}")
                result.is_valid = False
            
            if action_dim and agent_action_dim and action_dim != agent_action_dim:
                result.errors.append(f"Action dimension mismatch: env={action_dim}, agent={agent_action_dim}")
                result.is_valid = False
            
            # Check environment stability
            episode_length = env_info.get('episode_length', 0)
            max_episode_length = context.get('max_episode_length', 10000)
            if episode_length > max_episode_length:
                result.warnings.append(f"Very long episodes ({episode_length}) may cause memory issues")
            
            # Check reward range
            reward_range = env_info.get('reward_range')
            if reward_range:
                min_reward, max_reward = reward_range
                if max_reward - min_reward > 1000:
                    result.warnings.append(f"Wide reward range [{min_reward}, {max_reward}] may cause learning instability")
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Environment compatibility validation error: {str(e)}")
        
        return result
    
    def _validate_performance_requirements(self, perf_data: Dict, context: Dict) -> ValidationResult:
        """Validate performance requirements and constraints."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Response time validation
            response_time = perf_data.get('response_time', 0)
            max_response_time = context.get('max_response_time', 1.0)  # 1 second default
            if response_time > max_response_time:
                result.warnings.append(f"Slow response time: {response_time:.3f}s > {max_response_time}s")
            
            # Throughput validation
            throughput = perf_data.get('throughput', 0)
            min_throughput = context.get('min_throughput', 1.0)  # 1 ops/sec default
            if throughput < min_throughput:
                result.warnings.append(f"Low throughput: {throughput:.2f} < {min_throughput} ops/sec")
            
            # Memory usage validation
            memory_usage = perf_data.get('memory_usage_mb', 0)
            max_memory = context.get('max_memory_mb', 2000)  # 2GB default
            if memory_usage > max_memory:
                result.errors.append(f"Memory usage {memory_usage:.1f}MB exceeds limit {max_memory}MB")
                result.is_valid = False
            
            # CPU usage validation
            cpu_usage = perf_data.get('cpu_usage_percent', 0)
            max_cpu = context.get('max_cpu_percent', 90)  # 90% default
            if cpu_usage > max_cpu:
                result.warnings.append(f"High CPU usage: {cpu_usage:.1f}% > {max_cpu}%")
            
            # Error rate validation
            error_rate = perf_data.get('error_rate', 0)
            max_error_rate = context.get('max_error_rate', 0.05)  # 5% default
            if error_rate > max_error_rate:
                result.errors.append(f"High error rate: {error_rate:.1%} > {max_error_rate:.1%}")
                result.is_valid = False
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Performance validation error: {str(e)}")
        
        return result
    
    def _validate_semantic_consistency(self, data: Any, validation_type: str, context: Dict) -> ValidationResult:
        """Validate semantic consistency and logical constraints."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Contextual validation based on type
            if validation_type == 'agent_validation':
                # Check temporal consistency
                if isinstance(data, dict):
                    step_count = data.get('step_count', 0)
                    episode_count = data.get('episode_count', 0)
                    
                    if step_count < episode_count:
                        result.warnings.append("Steps less than episodes (unusual pattern)")
            
            elif validation_type == 'array_validation':
                # Check array semantic consistency
                if isinstance(data, np.ndarray):
                    # For probability distributions, check sum to 1
                    if context.get('is_probability_distribution'):
                        array_sum = np.sum(data)
                        if not np.isclose(array_sum, 1.0, atol=1e-6):
                            result.errors.append(f"Probability distribution sum {array_sum:.6f} != 1.0")
                            result.is_valid = False
                    
                    # For correlation matrices, check symmetry
                    if context.get('is_correlation_matrix'):
                        if data.shape[0] != data.shape[1]:
                            result.errors.append("Correlation matrix must be square")
                            result.is_valid = False
                        elif not np.allclose(data, data.T, atol=1e-6):
                            result.warnings.append("Correlation matrix not symmetric")
        
        except Exception as e:
            result.warnings.append(f"Semantic validation error: {str(e)}")
        
        return result
    
    def _validate_security_constraints(self, data: Any, context: Dict) -> ValidationResult:
        """Validate security constraints and detect potential threats."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check for potential injection attacks in string data
            if isinstance(data, str):
                suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(', '__import__']
                for pattern in suspicious_patterns:
                    if pattern.lower() in data.lower():
                        result.errors.append(f"Potential code injection detected: '{pattern}'")
                        result.is_valid = False
            
            # Check for excessively large data (DoS attack)
            data_size = self._estimate_data_size(data)
            max_size_mb = context.get('max_data_size_mb', 100)  # 100MB default
            if data_size > max_size_mb * 1024 * 1024:
                result.errors.append(f"Data size {data_size/(1024*1024):.1f}MB exceeds security limit {max_size_mb}MB")
                result.is_valid = False
            
            # Check for suspicious file paths
            if isinstance(data, (str, Path)):
                path_str = str(data)
                suspicious_paths = ['../', '..\\', '/etc/', 'C:\\Windows', '/proc/']
                for sus_path in suspicious_paths:
                    if sus_path in path_str:
                        result.warnings.append(f"Suspicious path pattern detected: '{sus_path}'")
            
            # Check for known malicious patterns in arrays
            if isinstance(data, np.ndarray) and data.dtype == object:
                result.warnings.append("Object arrays may contain arbitrary code")
        
        except Exception as e:
            result.warnings.append(f"Security validation error: {str(e)}")
        
        return result
    
    def _validate_performance_impact(self, data: Any, context: Dict) -> ValidationResult:
        """Validate potential performance impact of data."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Estimate computational complexity
            if isinstance(data, np.ndarray):
                # Large arrays may cause performance issues
                if data.size > 1000000:  # 1M elements
                    result.warnings.append(f"Large array ({data.size} elements) may impact performance")
                
                # High-dimensional arrays
                if data.ndim > 5:
                    result.warnings.append(f"High-dimensional array ({data.ndim}D) may be computationally expensive")
            
            elif isinstance(data, dict):
                # Deep nested structures
                max_depth = self._compute_dict_depth(data)
                if max_depth > 10:
                    result.warnings.append(f"Deeply nested structure (depth {max_depth}) may impact performance")
                
                # Large number of keys
                if len(data) > 1000:
                    result.warnings.append(f"Large dictionary ({len(data)} keys) may impact performance")
        
        except Exception as e:
            result.warnings.append(f"Performance impact validation error: {str(e)}")
        
        return result
    
    def _merge_validation_results(self, target: ValidationResult, source: ValidationResult):
        """Merge validation results."""
        target.errors.extend(source.errors)
        target.warnings.extend(source.warnings)
        target.metadata.update(source.metadata)
        target.is_valid = target.is_valid and source.is_valid
    
    def _compute_validation_confidence(self, result: ValidationResult) -> float:
        """Compute confidence score for validation result."""
        if not result.is_valid:
            return 0.0
        
        # Base confidence
        confidence = 1.0
        
        # Reduce confidence based on warnings
        warning_penalty = min(0.5, len(result.warnings) * 0.1)
        confidence -= warning_penalty
        
        # Validation time penalty (slower validation = less confidence)
        if result.validation_time > 1.0:
            time_penalty = min(0.2, (result.validation_time - 1.0) * 0.05)
            confidence -= time_penalty
        
        return max(0.0, confidence)
    
    def _check_parameter_interaction(self, params: Dict, interaction: Dict) -> bool:
        """Check parameter interaction constraint."""
        try:
            # Example: check if learning_rate * temperature < threshold
            condition = interaction.get('condition')
            if condition == 'lr_temp_product':
                lr = params.get('learning_rate', 0)
                temp = params.get('temperature', 1)
                threshold = interaction.get('threshold', 0.1)
                return lr * temp <= threshold
            
            # Add more interaction checks as needed
            return True
        
        except Exception:
            return True  # Assume valid if check fails
    
    def _compute_skewness(self, array: np.ndarray) -> float:
        """Compute skewness of array."""
        if array.size < 3:
            return 0.0
        
        mean = np.mean(array)
        std = np.std(array)
        
        if std == 0:
            return 0.0
        
        return np.mean(((array - mean) / std) ** 3)
    
    def _compute_entropy(self, array: np.ndarray) -> float:
        """Compute entropy of array values."""
        try:
            # Discretize continuous values
            hist, _ = np.histogram(array, bins=min(50, array.size // 10))
            hist = hist[hist > 0]  # Remove zero bins
            
            # Normalize to probabilities
            probs = hist / np.sum(hist)
            
            # Compute entropy
            entropy = -np.sum(probs * np.log2(probs))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(probs))
            return entropy / max_entropy if max_entropy > 0 else 0.0
        
        except Exception:
            return 0.5  # Default entropy
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate memory size of data object."""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_data_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._estimate_data_size(k) + self._estimate_data_size(v) 
                          for k, v in data.items())
            else:
                return 64  # Rough estimate for other objects
        except Exception:
            return 0
    
    def _compute_dict_depth(self, data: Dict, current_depth: int = 0) -> int:
        """Compute maximum depth of nested dictionary."""
        if not isinstance(data, dict):
            return current_depth
        
        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self._compute_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics."""
        if not self.validation_history:
            return {'status': 'no_validations_performed'}
        
        recent_validations = list(self.validation_history)[-100:]  # Last 100
        
        stats = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'success_rate': np.mean([v['is_valid'] for v in recent_validations]),
            'avg_validation_time': np.mean([v['validation_time'] for v in recent_validations]),
            'avg_errors_per_validation': np.mean([v['error_count'] for v in recent_validations]),
            'avg_warnings_per_validation': np.mean([v['warning_count'] for v in recent_validations]),
            'most_common_errors': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
            'validation_types': list(set(v['validation_type'] for v in recent_validations))
        }
        
        return stats


class HealthMonitor:
    """Real-time health monitoring system."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger("HealthMonitor")
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 0.1,
            'response_time': 2.0
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        self.last_alert_time = 0
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                self._check_alert_thresholds(metrics)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> HealthMetrics:
        """Collect current system health metrics."""
        try:
            # System resource usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            # Application metrics
            error_rate = self.error_count / max(1, self.total_requests)
            avg_response_time = np.mean(self.request_times) if self.request_times else 0
            throughput = len(self.request_times) / max(1, self.monitoring_interval)
            
            return HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_info.percent,
                disk_usage=disk_info.percent,
                error_rate=error_rate,
                response_time=avg_response_time,
                throughput=throughput,
                timestamp=time.time()
            )
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return HealthMetrics(
                cpu_usage=0, memory_usage=0, disk_usage=0,
                error_rate=0, response_time=0, throughput=0,
                timestamp=time.time()
            )
    
    def _check_alert_thresholds(self, metrics: HealthMetrics):
        """Check if any metrics exceed alert thresholds."""
        current_time = time.time()
        
        # Prevent alert spam (minimum 60 seconds between similar alerts)
        if current_time - self.last_alert_time < 60:
            return
        
        alerts = []
        
        # Check each threshold
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        
        if metrics.response_time > self.alert_thresholds['response_time']:
            alerts.append(f"Slow response time: {metrics.response_time:.3f}s")
        
        # Trigger alerts
        if alerts:
            self.last_alert_time = current_time
            for alert_msg in alerts:
                self.logger.warning(f"Health alert: {alert_msg}")
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_msg, metrics)
                    except Exception as e:
                        self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, HealthMetrics], None]):
        """Add callback function for health alerts."""
        self.alert_callbacks.append(callback)
    
    def record_request_time(self, request_time: float):
        """Record response time for a request."""
        self.request_times.append(request_time)
        self.total_requests += 1
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        else:
            return self._collect_system_metrics()
    
    def get_health_history(self, last_n: int = 100) -> List[HealthMetrics]:
        """Get recent health metrics history."""
        return list(self.metrics_history)[-last_n:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 readings
        
        summary = {
            'monitoring_active': self.is_monitoring,
            'total_metrics_collected': len(self.metrics_history),
            'recent_metrics_count': len(recent_metrics),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_disk_usage': np.mean([m.disk_usage for m in recent_metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in recent_metrics]),
            'avg_response_time': np.mean([m.response_time for m in recent_metrics]),
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'alert_thresholds': self.alert_thresholds,
            'total_requests': self.total_requests,
            'total_errors': self.error_count,
            'overall_error_rate': self.error_count / max(1, self.total_requests)
        }
        
        return summary


class SecurityMonitor:
    """Security monitoring and threat detection system."""
    
    def __init__(self):
        self.logger = logging.getLogger("SecurityMonitor")
        self.security_events = deque(maxlen=10000)
        self.threat_patterns = self._initialize_threat_patterns()
        self.blocked_ips = set()
        self.rate_limits = defaultdict(deque)  # IP -> request times
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_minute = 100
    
    def _initialize_threat_patterns(self) -> Dict[str, Dict]:
        """Initialize threat detection patterns."""
        return {
            'sql_injection': {
                'patterns': ['union select', 'drop table', "'; --", "' or 1=1"],
                'severity': 'high'
            },
            'code_injection': {
                'patterns': ['eval(', 'exec(', '__import__', 'subprocess.'],
                'severity': 'critical'
            },
            'path_traversal': {
                'patterns': ['../', '..\\', '/etc/passwd', 'C:\\Windows'],
                'severity': 'high'
            },
            'xss_attempt': {
                'patterns': ['<script', 'javascript:', 'onerror=', 'onload='],
                'severity': 'medium'
            }
        }
    
    def validate_input_security(self, input_data: str, source_ip: str = None) -> ValidationResult:
        """Validate input data for security threats."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check rate limiting
            if source_ip and not self._check_rate_limit(source_ip):
                result.is_valid = False
                result.errors.append(f"Rate limit exceeded for IP {source_ip}")
                self._record_security_event("rate_limit_exceeded", "medium", 
                                           f"IP {source_ip} exceeded rate limit", source_ip or "unknown")
                return result
            
            # Check against threat patterns
            input_lower = input_data.lower()
            
            for threat_type, threat_info in self.threat_patterns.items():
                for pattern in threat_info['patterns']:
                    if pattern in input_lower:
                        severity = threat_info['severity']
                        result.is_valid = False
                        result.errors.append(f"{threat_type.replace('_', ' ').title()} detected: '{pattern}'")
                        
                        # Record security event
                        self._record_security_event(
                            threat_type, severity,
                            f"Threat pattern '{pattern}' detected in input",
                            source_ip or "unknown",
                            {'input_sample': input_data[:100]}  # First 100 chars
                        )
                        
                        # Block IP for critical threats
                        if severity == 'critical' and source_ip:
                            self._block_ip(source_ip)
            
            # Check input length (potential DoS)
            if len(input_data) > 1000000:  # 1MB
                result.warnings.append(f"Very large input: {len(input_data)} bytes")
                self._record_security_event(
                    "large_input", "low",
                    f"Large input received: {len(input_data)} bytes",
                    source_ip or "unknown"
                )
        
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            result.warnings.append(f"Security validation error: {str(e)}")
        
        return result
    
    def _check_rate_limit(self, source_ip: str) -> bool:
        """Check if IP is within rate limits."""
        current_time = time.time()
        
        # Clean old entries
        self.rate_limits[source_ip] = deque(
            [t for t in self.rate_limits[source_ip] 
             if current_time - t <= self.rate_limit_window],
            maxlen=self.max_requests_per_minute
        )
        
        # Check current count
        if len(self.rate_limits[source_ip]) >= self.max_requests_per_minute:
            return False
        
        # Record this request
        self.rate_limits[source_ip].append(current_time)
        return True
    
    def _block_ip(self, source_ip: str):
        """Block an IP address."""
        self.blocked_ips.add(source_ip)
        self.logger.warning(f"IP {source_ip} has been blocked due to security threat")
        
        # Record blocking event
        self._record_security_event(
            "ip_blocked", "high",
            f"IP {source_ip} blocked due to security threat",
            source_ip
        )
    
    def _record_security_event(self, event_type: str, severity: str, 
                              description: str, source: str, metadata: Dict = None):
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Log based on severity
        log_msg = f"Security event [{severity}]: {description} from {source}"
        if severity == 'critical':
            self.logger.critical(log_msg)
        elif severity == 'high':
            self.logger.error(log_msg)
        elif severity == 'medium':
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
    
    def is_ip_blocked(self, source_ip: str) -> bool:
        """Check if an IP is blocked."""
        return source_ip in self.blocked_ips
    
    def unblock_ip(self, source_ip: str):
        """Unblock an IP address."""
        if source_ip in self.blocked_ips:
            self.blocked_ips.remove(source_ip)
            self.logger.info(f"IP {source_ip} has been unblocked")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        recent_events = [e for e in self.security_events 
                        if time.time() - e.timestamp <= 3600]  # Last hour
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
        
        return {
            'total_security_events': len(self.security_events),
            'recent_events_1h': len(recent_events),
            'blocked_ips_count': len(self.blocked_ips),
            'blocked_ips': list(self.blocked_ips),
            'recent_event_types': dict(event_counts),
            'recent_severity_distribution': dict(severity_counts),
            'threat_patterns_monitored': len(self.threat_patterns),
            'rate_limit_active_ips': len([ip for ip, times in self.rate_limits.items() if times])
        }


def robust_execution(max_retries: int = 3, 
                    fallback_value: Any = None,
                    exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator for robust execution with retries and fallback."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Exponential backoff
                        sleep_time = 0.1 * (2 ** attempt)
                        time.sleep(sleep_time)
                        
                        logging.getLogger(func.__module__).warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                        )
                    else:
                        logging.getLogger(func.__module__).error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            # Return fallback value if all attempts failed
            if fallback_value is not None:
                logging.getLogger(func.__module__).info(
                    f"Using fallback value for {func.__name__}"
                )
                return fallback_value
            
            # Re-raise the last exception if no fallback
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def error_recovery_context(recovery_actions: List[Callable] = None):
    """Context manager for automatic error recovery."""
    recovery_actions = recovery_actions or []
    
    try:
        yield
    except Exception as e:
        logging.getLogger("ErrorRecovery").error(f"Error occurred: {e}")
        
        # Execute recovery actions
        for action in recovery_actions:
            try:
                action()
                logging.getLogger("ErrorRecovery").info(f"Recovery action executed: {action.__name__}")
            except Exception as recovery_error:
                logging.getLogger("ErrorRecovery").error(
                    f"Recovery action failed: {recovery_error}"
                )
        
        # Re-raise the original exception
        raise


class RobustActiveInferenceFramework:
    """Robust Active Inference framework with comprehensive error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger("RobustActiveInference")
        
        # Initialize monitoring and validation systems
        self.validator = AdvancedValidator(strict_mode=False)
        self.health_monitor = HealthMonitor(monitoring_interval=10.0)
        self.security_monitor = SecurityMonitor()
        
        # Framework state
        self.is_initialized = False
        self.framework_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'start_time': time.time()
        }
        
        # Start monitoring
        self.health_monitor.start_monitoring()
        
        self.logger.info("Robust Active Inference Framework initialized")
    
    @robust_execution(max_retries=2, fallback_value={'status': 'error', 'value': None})
    def safe_agent_operation(self, agent: Any, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute agent operation with comprehensive error handling."""
        operation_start_time = time.time()
        
        try:
            self.framework_stats['total_operations'] += 1
            
            # Validate agent state before operation
            if hasattr(agent, 'get_statistics'):
                agent_state = agent.get_statistics()
                validation_result = self.validator.validate_comprehensive(
                    agent_state, 'agent_validation'
                )
                
                if not validation_result.is_valid:
                    raise ValidationException(f"Agent validation failed: {validation_result.errors}")
            
            # Execute the operation
            if hasattr(agent, operation):
                method = getattr(agent, operation)
                result = method(*args, **kwargs)
            else:
                raise AttributeError(f"Agent does not have method '{operation}'")
            
            # Record successful operation
            operation_time = time.time() - operation_start_time
            self.health_monitor.record_request_time(operation_time)
            self.framework_stats['successful_operations'] += 1
            
            return {'status': 'success', 'value': result, 'operation_time': operation_time}
        
        except Exception as e:
            # Record failed operation
            self.health_monitor.record_error()
            self.framework_stats['failed_operations'] += 1
            
            # Log error with context
            self.logger.error(f"Agent operation '{operation}' failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Return error information
            return {
                'status': 'error',
                'error': str(e),
                'operation': operation,
                'operation_time': time.time() - operation_start_time
            }
    
    def get_framework_health(self) -> Dict[str, Any]:
        """Get comprehensive framework health status."""
        health_summary = self.health_monitor.get_health_summary()
        validation_stats = self.validator.get_validation_statistics()
        security_summary = self.security_monitor.get_security_summary()
        
        return {
            'framework_stats': self.framework_stats,
            'health_monitoring': health_summary,
            'validation_stats': validation_stats,
            'security_summary': security_summary,
            'uptime': time.time() - self.framework_stats['start_time'],
            'overall_health_score': self._compute_overall_health_score(health_summary, validation_stats, security_summary)
        }
    
    def _compute_overall_health_score(self, health: Dict, validation: Dict, security: Dict) -> float:
        """Compute overall health score (0-1)."""
        try:
            # Health component (40%)
            health_score = 1.0
            if 'avg_cpu_usage' in health:
                health_score -= min(0.5, health['avg_cpu_usage'] / 200)  # Penalty for high CPU
            if 'overall_error_rate' in health:
                health_score -= min(0.3, health['overall_error_rate'] * 3)  # Penalty for errors
            health_component = max(0, health_score) * 0.4
            
            # Validation component (30%)
            validation_score = 1.0
            if 'success_rate' in validation:
                validation_score = validation['success_rate']
            validation_component = validation_score * 0.3
            
            # Security component (30%)
            security_score = 1.0
            if 'recent_events_1h' in security:
                # Penalty for recent security events
                security_score -= min(0.5, security['recent_events_1h'] / 20)
            if 'blocked_ips_count' in security:
                # Small penalty for blocked IPs (indicates threats but also protection)
                security_score -= min(0.1, security['blocked_ips_count'] / 100)
            security_component = max(0, security_score) * 0.3
            
            overall_score = health_component + validation_component + security_component
            return min(1.0, max(0.0, overall_score))
        
        except Exception as e:
            self.logger.error(f"Error computing health score: {e}")
            return 0.5  # Default score
    
    def shutdown(self):
        """Gracefully shutdown the framework."""
        self.logger.info("Shutting down Robust Active Inference Framework")
        
        try:
            self.health_monitor.stop_monitoring()
            
            # Log final statistics
            final_stats = self.get_framework_health()
            self.logger.info(f"Final framework statistics: {final_stats['framework_stats']}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("Framework shutdown complete")
