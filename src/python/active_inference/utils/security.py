"""
Security monitoring and access control for Active Inference components.

This module provides comprehensive security measures including input sanitization,
access control, security event logging, and threat detection.
"""

import hashlib
import hmac
import secrets
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import numpy as np
from pathlib import Path

from .validation import ValidationError, ActiveInferenceError
from .logging_config import get_logger, LogCategory


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INJECTION_ATTEMPT = "injection_attempt"
    CONFIGURATION_TAMPERING = "config_tampering"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    description: str
    source: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'description': self.description,
            'source': self.source,
            'timestamp': self.timestamp,
            'details': self.details or {},
            'remediation': self.remediation
        }


class InputSanitizer:
    """
    Input sanitization and validation for security.
    
    Provides protection against various injection attacks and malicious inputs.
    """
    
    def __init__(self):
        self.logger = get_logger("security")
        self._suspicious_patterns = [
            # SQL injection patterns
            r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            # Command injection patterns
            r"(?i)([;&|`]|\$\(|\${|<\()",
            # Path traversal patterns
            r"\.\.[\\/]",
            # Script injection patterns
            r"(?i)(<script|javascript:|vbscript:|on\w+\s*=)",
            # LDAP injection patterns
            r"[()=*!&|]",
            # NoSQL injection patterns
            r"(?i)(\$where|\$ne|\$gt|\$lt)"
        ]
        
        self._max_string_length = 10000
        self._max_array_size = 10000
        self._blocked_keywords = {
            'eval', 'exec', 'system', 'shell', 'subprocess',
            'import', '__import__', 'open', 'file', 'input'
        }
    
    def sanitize_string(self, value: str, context: str = "input") -> str:
        """
        Sanitize string input for security.
        
        Args:
            value: String to sanitize
            context: Context of the input for logging
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If input is malicious or invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string input, got {type(value)}")
        
        # Check length
        if len(value) > self._max_string_length:
            self.logger.warning(
                "String input exceeds maximum length",
                LogCategory.SECURITY,
                {
                    'context': context,
                    'length': len(value),
                    'max_length': self._max_string_length
                }
            )
            raise ValidationError(f"String input too long: {len(value)} > {self._max_string_length}")
        
        # Check for suspicious patterns
        import re
        for pattern in self._suspicious_patterns:
            if re.search(pattern, value):
                self._log_security_event(
                    SecurityEventType.INJECTION_ATTEMPT,
                    ThreatLevel.HIGH,
                    f"Suspicious pattern detected in {context}",
                    {'pattern': pattern, 'input': value[:100]}  # Truncate for logging
                )
                raise ValidationError(f"Suspicious pattern detected in {context}")
        
        # Check for blocked keywords
        value_lower = value.lower()
        for keyword in self._blocked_keywords:
            if keyword in value_lower:
                self._log_security_event(
                    SecurityEventType.INJECTION_ATTEMPT,
                    ThreatLevel.MEDIUM,
                    f"Blocked keyword '{keyword}' found in {context}",
                    {'keyword': keyword, 'context': context}
                )
                raise ValidationError(f"Blocked keyword '{keyword}' in {context}")
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def sanitize_array(self, value: Union[List, np.ndarray], context: str = "input") -> Union[List, np.ndarray]:
        """
        Sanitize array input for security.
        
        Args:
            value: Array to sanitize
            context: Context of the input
            
        Returns:
            Sanitized array
            
        Raises:
            ValidationError: If input is malicious or invalid
        """
        if isinstance(value, np.ndarray):
            # Check size
            if value.size > self._max_array_size:
                self._log_security_event(
                    SecurityEventType.RESOURCE_EXHAUSTION,
                    ThreatLevel.MEDIUM,
                    f"Array size exceeds limit in {context}",
                    {'size': value.size, 'max_size': self._max_array_size}
                )
                raise ValidationError(f"Array too large: {value.size} > {self._max_array_size}")
            
            # Check for infinite or NaN values
            if np.any(np.isinf(value)) or np.any(np.isnan(value)):
                self._log_security_event(
                    SecurityEventType.INPUT_VALIDATION_FAILURE,
                    ThreatLevel.LOW,
                    f"Invalid numeric values in {context}",
                    {'has_inf': bool(np.any(np.isinf(value))), 'has_nan': bool(np.any(np.isnan(value)))}
                )
                raise ValidationError(f"Array contains invalid numeric values in {context}")
            
            # Check for extreme values that could cause overflow
            if np.any(np.abs(value) > 1e10):
                self._log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.LOW,
                    f"Extremely large values in {context}",
                    {'max_abs_value': float(np.max(np.abs(value)))}
                )
                # Clip rather than reject
                value = np.clip(value, -1e10, 1e10)
        
        elif isinstance(value, list):
            # Check size
            if len(value) > self._max_array_size:
                self._log_security_event(
                    SecurityEventType.RESOURCE_EXHAUSTION,
                    ThreatLevel.MEDIUM,
                    f"List size exceeds limit in {context}",
                    {'size': len(value), 'max_size': self._max_array_size}
                )
                raise ValidationError(f"List too large: {len(value)} > {self._max_array_size}")
            
            # Sanitize string elements
            sanitized_list = []
            for i, item in enumerate(value):
                if isinstance(item, str):
                    try:
                        sanitized_item = self.sanitize_string(item, f"{context}[{i}]")
                        sanitized_list.append(sanitized_item)
                    except ValidationError:
                        # Skip malicious items
                        self.logger.warning(
                            f"Skipping malicious item at {context}[{i}]",
                            LogCategory.SECURITY
                        )
                        continue
                else:
                    sanitized_list.append(item)
            
            value = sanitized_list
        
        return value
    
    def sanitize_dict(self, value: Dict[str, Any], context: str = "input") -> Dict[str, Any]:
        """
        Sanitize dictionary input for security.
        
        Args:
            value: Dictionary to sanitize
            context: Context of the input
            
        Returns:
            Sanitized dictionary
            
        Raises:
            ValidationError: If input is malicious or invalid
        """
        if not isinstance(value, dict):
            raise ValidationError(f"Expected dict input, got {type(value)}")
        
        # Check size
        if len(value) > 1000:
            self._log_security_event(
                SecurityEventType.RESOURCE_EXHAUSTION,
                ThreatLevel.MEDIUM,
                f"Dictionary size exceeds limit in {context}",
                {'size': len(value), 'max_size': 1000}
            )
            raise ValidationError(f"Dictionary too large: {len(value)} > 1000")
        
        sanitized_dict = {}
        
        for key, val in value.items():
            # Sanitize keys
            if isinstance(key, str):
                try:
                    sanitized_key = self.sanitize_string(key, f"{context}.key")
                except ValidationError:
                    # Skip malicious keys
                    self.logger.warning(
                        f"Skipping malicious key in {context}: {key[:50]}",
                        LogCategory.SECURITY
                    )
                    continue
            else:
                sanitized_key = key
            
            # Sanitize values
            if isinstance(val, str):
                try:
                    sanitized_val = self.sanitize_string(val, f"{context}.{sanitized_key}")
                except ValidationError:
                    # Skip malicious values
                    continue
            elif isinstance(val, (list, np.ndarray)):
                try:
                    sanitized_val = self.sanitize_array(val, f"{context}.{sanitized_key}")
                except ValidationError:
                    # Skip malicious arrays
                    continue
            elif isinstance(val, dict):
                try:
                    sanitized_val = self.sanitize_dict(val, f"{context}.{sanitized_key}")
                except ValidationError:
                    # Skip malicious nested dicts
                    continue
            else:
                sanitized_val = val
            
            sanitized_dict[sanitized_key] = sanitized_val
        
        return sanitized_dict
    
    def _log_security_event(self, event_type: SecurityEventType, 
                           threat_level: ThreatLevel, description: str,
                           details: Optional[Dict[str, Any]] = None):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            source="InputSanitizer",
            timestamp=time.time(),
            details=details
        )
        
        self.logger.log_security_event(
            event_type.value,
            threat_level.value,
            description,
            event.to_dict()
        )


class RateLimiter:
    """
    Rate limiting for API endpoints and operations.
    
    Provides protection against abuse and DoS attacks.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # client_id -> list of timestamps
        self.lock = threading.Lock()
        self.logger = get_logger("security")
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        
        with self.lock:
            # Initialize client if not exists
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # Remove old requests outside window
            cutoff = now - self.window_seconds
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] 
                if req_time > cutoff
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            else:
                # Log rate limit exceeded
                self.logger.log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED.value,
                    ThreatLevel.MEDIUM.value,
                    f"Rate limit exceeded for client {client_id}",
                    {
                        'client_id': client_id,
                        'requests_in_window': len(self.requests[client_id]),
                        'max_requests': self.max_requests,
                        'window_seconds': self.window_seconds
                    }
                )
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            active_clients = 0
            total_requests = 0
            
            for client_id, requests in self.requests.items():
                recent_requests = [req for req in requests if req > cutoff]
                if recent_requests:
                    active_clients += 1
                    total_requests += len(recent_requests)
            
            return {
                'active_clients': active_clients,
                'total_recent_requests': total_requests,
                'max_requests_per_window': self.max_requests,
                'window_seconds': self.window_seconds
            }


class SecurityMonitor:
    """
    Central security monitoring and incident response system.
    
    Aggregates security events, detects patterns, and triggers responses.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize security monitor.
        
        Args:
            config_file: Optional configuration file path
        """
        self.logger = get_logger("security")
        self.events = []  # Security event history
        self.patterns = {}  # Pattern detection state
        self.lock = threading.Lock()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get('max_requests_per_minute', 100),
            window_seconds=60
        )
        
        # Security state
        self.blocked_clients = set()
        self.quarantined_agents = set()
        
        self.logger.info(
            "Security monitor initialized",
            LogCategory.SECURITY,
            {'config': self.config}
        )
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load security configuration."""
        default_config = {
            'max_requests_per_minute': 100,
            'max_failed_auth_attempts': 5,
            'quarantine_threshold': 10,
            'auto_response_enabled': True,
            'log_all_events': True,
            'threat_level_thresholds': {
                'low': 0,
                'medium': 5,
                'high': 2,
                'critical': 1
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                self.logger.warning(
                    f"Failed to load security config from {config_file}: {e}",
                    LogCategory.SECURITY
                )
        
        return default_config
    
    def log_event(self, event: SecurityEvent) -> None:
        """
        Log security event and check for patterns.
        
        Args:
            event: Security event to log
        """
        with self.lock:
            self.events.append(event)
            
            # Keep only recent events (last 24 hours)
            cutoff = time.time() - 86400
            self.events = [e for e in self.events if e.timestamp > cutoff]
            
            # Log the event
            if self.config.get('log_all_events', True):
                self.logger.log_security_event(
                    event.event_type.value,
                    event.threat_level.value,
                    event.description,
                    event.to_dict()
                )
            
            # Check for patterns and trigger responses
            self._check_patterns()
            self._trigger_auto_response(event)
    
    def _check_patterns(self) -> None:
        """Check for suspicious patterns in recent events."""
        now = time.time()
        recent_window = 300  # 5 minutes
        recent_events = [
            e for e in self.events 
            if now - e.timestamp < recent_window
        ]
        
        # Count events by type and source
        event_counts = {}
        source_counts = {}
        
        for event in recent_events:
            event_type = event.event_type.value
            source = event.source
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Check thresholds
        thresholds = self.config.get('threat_level_thresholds', {})
        
        for event_type, count in event_counts.items():
            for threat_level, threshold in thresholds.items():
                if count >= threshold:
                    self._handle_pattern_detection(
                        f"High frequency of {event_type} events",
                        {'event_type': event_type, 'count': count, 'threshold': threshold}
                    )
        
        # Check for suspicious sources
        for source, count in source_counts.items():
            if count >= 20:  # 20 events from same source in 5 minutes
                self._handle_pattern_detection(
                    f"High activity from source {source}",
                    {'source': source, 'count': count}
                )
    
    def _handle_pattern_detection(self, description: str, details: Dict[str, Any]) -> None:
        """Handle detected suspicious patterns."""
        pattern_event = SecurityEvent(
            event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
            threat_level=ThreatLevel.HIGH,
            description=description,
            source="SecurityMonitor",
            timestamp=time.time(),
            details=details,
            remediation="Pattern detected - consider investigation"
        )
        
        self.logger.log_security_event(
            pattern_event.event_type.value,
            pattern_event.threat_level.value,
            pattern_event.description,
            pattern_event.to_dict()
        )
    
    def _trigger_auto_response(self, event: SecurityEvent) -> None:
        """Trigger automatic security responses."""
        if not self.config.get('auto_response_enabled', True):
            return
        
        # Block clients with repeated high-severity events
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            source = event.source
            recent_high_events = [
                e for e in self.events[-10:]  # Last 10 events
                if e.source == source and e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
            
            if len(recent_high_events) >= 3:
                self.blocked_clients.add(source)
                self.logger.log_security_event(
                    "auto_response",
                    ThreatLevel.HIGH.value,
                    f"Auto-blocked client {source} due to repeated high-severity events",
                    {'blocked_client': source, 'trigger_events': len(recent_high_events)}
                )
        
        # Quarantine agents with suspicious behavior
        if event.event_type == SecurityEventType.ANOMALOUS_BEHAVIOR:
            if event.details and 'agent_id' in event.details:
                agent_id = event.details['agent_id']
                self.quarantined_agents.add(agent_id)
                self.logger.log_security_event(
                    "auto_response",
                    ThreatLevel.MEDIUM.value,
                    f"Auto-quarantined agent {agent_id} due to anomalous behavior",
                    {'quarantined_agent': agent_id}
                )
    
    def is_client_blocked(self, client_id: str) -> bool:
        """Check if client is blocked."""
        return client_id in self.blocked_clients
    
    def is_agent_quarantined(self, agent_id: str) -> bool:
        """Check if agent is quarantined."""
        return agent_id in self.quarantined_agents
    
    def unblock_client(self, client_id: str) -> None:
        """Manually unblock a client."""
        with self.lock:
            self.blocked_clients.discard(client_id)
            self.logger.info(
                f"Manually unblocked client {client_id}",
                LogCategory.SECURITY
            )
    
    def unquarantine_agent(self, agent_id: str) -> None:
        """Manually unquarantine an agent."""
        with self.lock:
            self.quarantined_agents.discard(agent_id)
            self.logger.info(
                f"Manually unquarantined agent {agent_id}",
                LogCategory.SECURITY
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        with self.lock:
            now = time.time()
            recent_events = [e for e in self.events if now - e.timestamp < 3600]  # Last hour
            
            threat_counts = {}
            for event in recent_events:
                threat_level = event.threat_level.value
                threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
            
            return {
                'total_events_last_hour': len(recent_events),
                'threat_level_counts': threat_counts,
                'blocked_clients': len(self.blocked_clients),
                'quarantined_agents': len(self.quarantined_agents),
                'rate_limiter_stats': self.rate_limiter.get_stats(),
                'auto_response_enabled': self.config.get('auto_response_enabled', True),
                'last_event_time': max([e.timestamp for e in self.events]) if self.events else None
            }


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


def secure_function(require_auth: bool = False, 
                   rate_limit: bool = True,
                   sanitize_inputs: bool = True):
    """
    Decorator for securing functions with authentication, rate limiting, and input sanitization.
    
    Args:
        require_auth: Whether to require authentication
        rate_limit: Whether to apply rate limiting
        sanitize_inputs: Whether to sanitize inputs
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            monitor = get_security_monitor()
            
            # Extract client ID (simplified - in practice would come from request context)
            client_id = kwargs.get('client_id', 'anonymous')
            
            # Check if client is blocked
            if monitor.is_client_blocked(client_id):
                raise ValidationError(f"Client {client_id} is blocked")
            
            # Rate limiting
            if rate_limit and not monitor.rate_limiter.is_allowed(client_id):
                raise ValidationError(f"Rate limit exceeded for client {client_id}")
            
            # Input sanitization
            if sanitize_inputs:
                sanitizer = monitor.sanitizer
                
                # Sanitize string arguments
                sanitized_args = []
                for arg in args:
                    if isinstance(arg, str):
                        sanitized_args.append(sanitizer.sanitize_string(arg, func.__name__))
                    elif isinstance(arg, (list, np.ndarray)):
                        sanitized_args.append(sanitizer.sanitize_array(arg, func.__name__))
                    elif isinstance(arg, dict):
                        sanitized_args.append(sanitizer.sanitize_dict(arg, func.__name__))
                    else:
                        sanitized_args.append(arg)
                
                # Sanitize keyword arguments
                sanitized_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        sanitized_kwargs[key] = sanitizer.sanitize_string(value, f"{func.__name__}.{key}")
                    elif isinstance(value, (list, np.ndarray)):
                        sanitized_kwargs[key] = sanitizer.sanitize_array(value, f"{func.__name__}.{key}")
                    elif isinstance(value, dict):
                        sanitized_kwargs[key] = sanitizer.sanitize_dict(value, f"{func.__name__}.{key}")
                    else:
                        sanitized_kwargs[key] = value
                
                return func(*sanitized_args, **sanitized_kwargs)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Export key classes and functions
__all__ = [
    'SecurityMonitor',
    'InputSanitizer', 
    'RateLimiter',
    'SecurityEvent',
    'SecurityEventType',
    'ThreatLevel',
    'get_security_monitor',
    'secure_function'
]