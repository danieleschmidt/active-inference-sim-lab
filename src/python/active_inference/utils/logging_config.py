"""
Comprehensive logging configuration for Active Inference components.

This module provides structured logging, monitoring, and telemetry capabilities
for tracking agent performance, errors, and system health.
"""

import logging
import logging.handlers
import json
import traceback
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Log category enumeration for structured logging."""
    AGENT = "agent"
    INFERENCE = "inference"
    PLANNING = "planning"
    ENVIRONMENT = "environment"
    MODEL = "model"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Structured log entry for JSON logging."""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values for cleaner logs
        return {k: v for k, v in result.items() if v is not None}


class StructuredLogger:
    """
    Enhanced logger with structured logging, metrics, and monitoring capabilities.
    
    Features:
    - JSON structured logging
    - Performance metrics tracking
    - Error aggregation and reporting
    - Session-based logging
    - Configurable output formats and destinations
    """
    
    def __init__(self,
                 name: str,
                 log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_json: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 session_id: Optional[str] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_level: Minimum log level
            log_file: Optional file path for logging
            enable_console: Whether to log to console
            enable_json: Whether to use JSON formatting
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            session_id: Optional session identifier
        """
        self.name = name
        self.log_level = log_level
        self.enable_json = enable_json
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Setup formatters
        if enable_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level.value)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level.value)
            self.logger.addHandler(file_handler)
        
        # Metrics tracking
        self._metrics = {
            'log_counts': {level.name: 0 for level in LogLevel},
            'error_counts': {},
            'performance_metrics': {},
            'start_time': time.time()
        }
        self._lock = threading.Lock()
        
        # Don't propagate to parent loggers to avoid duplicates
        self.logger.propagate = False
    
    def _log_structured(self,
                       level: LogLevel,
                       category: LogCategory,
                       message: str,
                       data: Optional[Dict[str, Any]] = None,
                       error: Optional[Exception] = None,
                       agent_id: Optional[str] = None):
        """Log structured entry with metadata."""
        try:
            # Update metrics
            with self._lock:
                self._metrics['log_counts'][level.name] += 1
                
                if error:
                    error_type = type(error).__name__
                    self._metrics['error_counts'][error_type] = \
                        self._metrics['error_counts'].get(error_type, 0) + 1
            
            # Create log entry
            entry = LogEntry(
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                level=level.name,
                category=category.value,
                component=self.name,
                message=message,
                data=data,
                error=str(error) if error else None,
                stack_trace=traceback.format_exc() if error else None,
                agent_id=agent_id,
                session_id=self.session_id
            )
            
            # Log the entry
            if self.enable_json:
                # Pass the structured data to the JSON formatter
                extra = {'structured_data': entry.to_dict()}
                self.logger.log(level.value, message, extra=extra)
            else:
                # Standard text logging
                log_msg = f"[{category.value}] {message}"
                if data:
                    log_msg += f" | Data: {data}"
                if error:
                    log_msg += f" | Error: {error}"
                
                self.logger.log(level.value, log_msg)
                
        except Exception as e:
            # Fallback logging if structured logging fails
            self.logger.error(f"Structured logging failed: {e}")
            self.logger.log(level.value, message)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log debug message."""
        self._log_structured(LogLevel.DEBUG, category, message, data, agent_id=agent_id)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log info message."""
        self._log_structured(LogLevel.INFO, category, message, data, agent_id=agent_id)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                data: Optional[Dict[str, Any]] = None, agent_id: Optional[str] = None):
        """Log warning message."""
        self._log_structured(LogLevel.WARNING, category, message, data, agent_id=agent_id)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM,
              data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None,
              agent_id: Optional[str] = None):
        """Log error message."""
        self._log_structured(LogLevel.ERROR, category, message, data, error, agent_id)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                 data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None,
                 agent_id: Optional[str] = None):
        """Log critical message."""
        self._log_structured(LogLevel.CRITICAL, category, message, data, error, agent_id)
    
    def log_performance(self, operation: str, duration: float, 
                       data: Optional[Dict[str, Any]] = None,
                       agent_id: Optional[str] = None):
        """Log performance metrics."""
        with self._lock:
            if operation not in self._metrics['performance_metrics']:
                self._metrics['performance_metrics'][operation] = []
            self._metrics['performance_metrics'][operation].append(duration)
        
        perf_data = {'operation': operation, 'duration': duration}
        if data:
            perf_data.update(data)
        
        self.info(f"Performance: {operation} took {duration:.4f}s", 
                 LogCategory.PERFORMANCE, perf_data, agent_id)
    
    def log_agent_step(self, agent_id: str, step_count: int, 
                      free_energy: float, reward: float,
                      additional_data: Optional[Dict[str, Any]] = None):
        """Log agent step information."""
        data = {
            'step_count': step_count,
            'free_energy': free_energy,
            'reward': reward
        }
        if additional_data:
            data.update(additional_data)
        
        self.debug(f"Agent step {step_count}", LogCategory.AGENT, data, agent_id)
    
    def log_inference_update(self, agent_id: str, belief_entropy: float,
                           observation_likelihood: float,
                           additional_data: Optional[Dict[str, Any]] = None):
        """Log inference update information."""
        data = {
            'belief_entropy': belief_entropy,
            'observation_likelihood': observation_likelihood
        }
        if additional_data:
            data.update(additional_data)
        
        self.debug("Belief update", LogCategory.INFERENCE, data, agent_id)
    
    def log_planning_decision(self, agent_id: str, selected_action: np.ndarray,
                            expected_free_energy: float, n_candidates: int,
                            additional_data: Optional[Dict[str, Any]] = None):
        """Log planning decision information."""
        data = {
            'selected_action': selected_action.tolist() if isinstance(selected_action, np.ndarray) else selected_action,
            'expected_free_energy': expected_free_energy,
            'n_candidates': n_candidates
        }
        if additional_data:
            data.update(additional_data)
        
        self.debug("Action planned", LogCategory.PLANNING, data, agent_id)
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, data: Optional[Dict[str, Any]] = None):
        """Log security-related events."""
        security_data = {
            'event_type': event_type,
            'severity': severity,
            'description': description
        }
        if data:
            security_data.update(data)
        
        level = LogLevel.WARNING if severity == 'medium' else LogLevel.ERROR
        self._log_structured(level, LogCategory.SECURITY, 
                           f"Security event: {event_type}", security_data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging and performance metrics."""
        with self._lock:
            # Calculate performance statistics
            perf_stats = {}
            for operation, durations in self._metrics['performance_metrics'].items():
                if durations:
                    perf_stats[operation] = {
                        'count': len(durations),
                        'mean': np.mean(durations),
                        'std': np.std(durations),
                        'min': np.min(durations),
                        'max': np.max(durations),
                        'total': np.sum(durations)
                    }
            
            return {
                'session_id': self.session_id,
                'uptime': time.time() - self._metrics['start_time'],
                'log_counts': self._metrics['log_counts'].copy(),
                'error_counts': self._metrics['error_counts'].copy(),
                'performance_stats': perf_stats
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics = {
                'log_counts': {level.name: 0 for level in LogLevel},
                'error_counts': {},
                'performance_metrics': {},
                'start_time': time.time()
            }


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        try:
            # Check if we have structured data
            if hasattr(record, 'structured_data'):
                log_entry = record.structured_data
            else:
                # Create basic structured entry
                log_entry = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'component': record.name,
                    'message': record.getMessage(),
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['error'] = str(record.exc_info[1])
                    log_entry['stack_trace'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry, default=str)
            
        except Exception as e:
            # Fallback to standard formatting if JSON fails
            return f"JSON_FORMAT_ERROR: {str(e)} | {record.getMessage()}"


class PerformanceTimer:
    """Context manager for timing operations and logging performance."""
    
    def __init__(self, logger: StructuredLogger, operation: str, 
                 agent_id: Optional[str] = None, 
                 additional_data: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.agent_id = agent_id
        self.additional_data = additional_data or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.log_performance(
                self.operation, duration, self.additional_data, self.agent_id
            )


def setup_global_logging(log_level: LogLevel = LogLevel.INFO,
                        log_dir: Optional[str] = None,
                        enable_console: bool = True,
                        enable_json: bool = True) -> StructuredLogger:
    """
    Setup global logging configuration for the Active Inference framework.
    
    Args:
        log_level: Global log level
        log_dir: Directory for log files (None to disable file logging)
        enable_console: Whether to enable console logging
        enable_json: Whether to use JSON formatting
        
    Returns:
        Main application logger
    """
    # Setup main application logger
    log_file = None
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = str(log_path / "active_inference.log")
    
    main_logger = StructuredLogger(
        "active_inference",
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console,
        enable_json=enable_json
    )
    
    # Setup component-specific loggers
    component_loggers = {}
    
    for category in LogCategory:
        component_log_file = None
        if log_dir:
            component_log_file = str(Path(log_dir) / f"{category.value}.log")
        
        component_loggers[category.value] = StructuredLogger(
            f"active_inference.{category.value}",
            log_level=log_level,
            log_file=component_log_file,
            enable_console=False,  # Only main logger logs to console
            enable_json=enable_json
        )
    
    # Store loggers globally for easy access
    global _global_loggers
    _global_loggers = {
        'main': main_logger,
        **component_loggers
    }
    
    main_logger.info("Global logging initialized", LogCategory.SYSTEM, {
        'log_level': log_level.name,
        'log_dir': log_dir,
        'enable_console': enable_console,
        'enable_json': enable_json
    })
    
    return main_logger


def get_logger(component: str = "main") -> StructuredLogger:
    """Get logger for specific component."""
    global _global_loggers
    if '_global_loggers' not in globals() or not _global_loggers:
        # Initialize with defaults if not setup
        setup_global_logging()
    
    # If component doesn't exist, create a basic logger
    if component not in _global_loggers:
        if 'main' not in _global_loggers:
            setup_global_logging()
        _global_loggers[component] = _global_loggers['main']
    
    return _global_loggers[component]


# Global logger storage
_global_loggers: Dict[str, StructuredLogger] = {}


# Export key classes and functions
__all__ = [
    'StructuredLogger',
    'LogLevel',
    'LogCategory',
    'PerformanceTimer',
    'setup_global_logging',
    'get_logger'
]