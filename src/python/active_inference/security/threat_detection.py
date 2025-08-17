"""
Advanced Threat Detection System
Generation 2: MAKE IT ROBUST (Reliable)
"""

import numpy as np
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatEvent:
    """Represents a detected threat event."""
    timestamp: float
    threat_type: str
    threat_level: ThreatLevel
    source_id: str
    details: Dict[str, Any]
    mitigation_applied: bool = False
    false_positive: bool = False


class AdaptiveThreatDetector:
    """
    Advanced threat detection system with machine learning capabilities
    and adaptive threshold management.
    """
    
    def __init__(self,
                 detection_window: float = 60.0,
                 max_events_memory: int = 1000,
                 learning_rate: float = 0.01,
                 enable_ml_detection: bool = True):
        """
        Initialize adaptive threat detector.
        
        Args:
            detection_window: Time window for threat aggregation (seconds)
            max_events_memory: Maximum threat events to keep in memory
            learning_rate: Learning rate for adaptive thresholds
            enable_ml_detection: Enable machine learning based detection
        """
        self.detection_window = detection_window
        self.max_events_memory = max_events_memory
        self.learning_rate = learning_rate
        self.enable_ml_detection = enable_ml_detection
        
        # Threat event storage
        self.threat_events = deque(maxlen=max_events_memory)
        self.threat_history = defaultdict(list)
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'rate_limit': 10.0,  # requests per second
            'input_size': 1000000,  # bytes
            'computation_time': 5.0,  # seconds
            'memory_usage': 500000000,  # bytes (500MB)
            'anomaly_score': 0.8,  # ML anomaly threshold
        }
        
        # Pattern detection
        self.known_attack_patterns = {
            'sql_injection': [
                r"union\s+select", r"drop\s+table", r"insert\s+into",
                r"delete\s+from", r"exec\s*\(", r"script\s*:"
            ],
            'xss': [
                r"<script", r"javascript:", r"onload\s*=", r"onerror\s*=",
                r"alert\s*\(", r"document\.cookie"
            ],
            'command_injection': [
                r";\s*rm\s+", r";\s*cat\s+", r";\s*ls\s+", r"&&\s*rm",
                r"\|\s*nc\s+", r"eval\s*\("
            ],
            'path_traversal': [
                r"\.\.\/", r"\.\.\\", r"%2e%2e%2f", r"%252e%252e%252f"
            ]
        }
        
        # Behavioral baseline
        self.behavioral_baseline = {
            'avg_request_rate': 1.0,
            'avg_input_size': 1000,
            'avg_processing_time': 0.1,
            'common_patterns': set()
        }
        
        # Client tracking
        self.client_profiles = defaultdict(lambda: {
            'request_count': 0,
            'last_request_time': 0,
            'request_rate': 0.0,
            'threat_score': 0.0,
            'blocked': False,
            'requests_per_window': deque(maxlen=100)
        })
        
        logger.info("Advanced threat detector initialized")
    
    def detect_threats(self, 
                      client_id: str,
                      input_data: Any,
                      processing_time: float = 0.0,
                      context: Optional[Dict[str, Any]] = None) -> List[ThreatEvent]:
        """
        Comprehensive threat detection across multiple vectors.
        
        Args:
            client_id: Unique client identifier
            input_data: Input data to analyze
            processing_time: Time taken to process request
            context: Additional context information
            
        Returns:
            List of detected threat events
        """
        threats = []
        current_time = time.time()
        
        try:
            # Update client profile
            self._update_client_profile(client_id, current_time)
            
            # Rate limiting detection
            rate_threats = self._detect_rate_limiting(client_id, current_time)
            threats.extend(rate_threats)
            
            # Input validation threats
            input_threats = self._detect_input_threats(client_id, input_data)
            threats.extend(input_threats)
            
            # Performance anomaly detection
            perf_threats = self._detect_performance_anomalies(
                client_id, processing_time
            )
            threats.extend(perf_threats)
            
            # Behavioral anomaly detection
            if self.enable_ml_detection:
                behavioral_threats = self._detect_behavioral_anomalies(
                    client_id, input_data, context
                )
                threats.extend(behavioral_threats)
            
            # Pattern-based detection
            pattern_threats = self._detect_attack_patterns(client_id, input_data)
            threats.extend(pattern_threats)
            
            # Store threat events
            for threat in threats:
                self.threat_events.append(threat)
                self.threat_history[threat.threat_type].append(threat)
            
            # Adaptive threshold updates
            self._update_adaptive_thresholds(threats)
            
            return threats
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return []
    
    def _update_client_profile(self, client_id: str, timestamp: float):
        """Update client behavioral profile."""
        profile = self.client_profiles[client_id]
        
        # Update request count and timing
        profile['request_count'] += 1
        
        if profile['last_request_time'] > 0:
            time_delta = timestamp - profile['last_request_time']
            if time_delta > 0:
                # Update request rate with exponential moving average
                new_rate = 1.0 / time_delta
                alpha = 0.1
                profile['request_rate'] = (
                    alpha * new_rate + (1 - alpha) * profile['request_rate']
                )
        
        profile['last_request_time'] = timestamp
        profile['requests_per_window'].append(timestamp)
    
    def _detect_rate_limiting(self, client_id: str, timestamp: float) -> List[ThreatEvent]:
        """Detect rate limiting violations."""
        threats = []
        profile = self.client_profiles[client_id]
        
        # Check requests in current window
        window_start = timestamp - self.detection_window
        recent_requests = [
            t for t in profile['requests_per_window'] 
            if t >= window_start
        ]
        
        request_rate = len(recent_requests) / self.detection_window
        
        if request_rate > self.adaptive_thresholds['rate_limit']:
            threat_level = ThreatLevel.HIGH if request_rate > self.adaptive_thresholds['rate_limit'] * 2 else ThreatLevel.MEDIUM
            
            threats.append(ThreatEvent(
                timestamp=timestamp,
                threat_type="rate_limiting",
                threat_level=threat_level,
                source_id=client_id,
                details={
                    'request_rate': request_rate,
                    'threshold': self.adaptive_thresholds['rate_limit'],
                    'window_size': self.detection_window
                }
            ))
        
        return threats
    
    def _detect_input_threats(self, client_id: str, input_data: Any) -> List[ThreatEvent]:
        """Detect input-based threats."""
        threats = []
        
        try:
            # Convert input to string for analysis
            if isinstance(input_data, np.ndarray):
                input_str = str(input_data.tolist())
                input_size = input_data.nbytes
            elif isinstance(input_data, (list, dict)):
                input_str = str(input_data)
                input_size = len(input_str.encode('utf-8'))
            else:
                input_str = str(input_data)
                input_size = len(input_str.encode('utf-8'))
            
            # Size-based detection
            if input_size > self.adaptive_thresholds['input_size']:
                threats.append(ThreatEvent(
                    timestamp=time.time(),
                    threat_type="oversized_input",
                    threat_level=ThreatLevel.MEDIUM,
                    source_id=client_id,
                    details={
                        'input_size': input_size,
                        'threshold': self.adaptive_thresholds['input_size']
                    }
                ))
            
            # Check for NaN/Inf values in numerical data
            if isinstance(input_data, np.ndarray):
                if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
                    threats.append(ThreatEvent(
                        timestamp=time.time(),
                        threat_type="invalid_numerical_input",
                        threat_level=ThreatLevel.HIGH,
                        source_id=client_id,
                        details={
                            'nan_count': np.sum(np.isnan(input_data)),
                            'inf_count': np.sum(np.isinf(input_data))
                        }
                    ))
            
        except Exception as e:
            logger.warning(f"Input threat detection failed: {e}")
        
        return threats
    
    def _detect_performance_anomalies(self, 
                                    client_id: str, 
                                    processing_time: float) -> List[ThreatEvent]:
        """Detect performance-based anomalies."""
        threats = []
        
        if processing_time > self.adaptive_thresholds['computation_time']:
            threat_level = (ThreatLevel.HIGH if processing_time > 
                          self.adaptive_thresholds['computation_time'] * 2 
                          else ThreatLevel.MEDIUM)
            
            threats.append(ThreatEvent(
                timestamp=time.time(),
                threat_type="performance_anomaly",
                threat_level=threat_level,
                source_id=client_id,
                details={
                    'processing_time': processing_time,
                    'threshold': self.adaptive_thresholds['computation_time']
                }
            ))
        
        return threats
    
    def _detect_behavioral_anomalies(self, 
                                   client_id: str,
                                   input_data: Any,
                                   context: Optional[Dict[str, Any]]) -> List[ThreatEvent]:
        """ML-based behavioral anomaly detection."""
        threats = []
        
        try:
            # Simple anomaly scoring based on deviation from baseline
            profile = self.client_profiles[client_id]
            
            # Request rate anomaly
            baseline_rate = self.behavioral_baseline['avg_request_rate']
            if profile['request_rate'] > baseline_rate * 5:
                anomaly_score = min(1.0, profile['request_rate'] / baseline_rate / 10)
                
                if anomaly_score > self.adaptive_thresholds['anomaly_score']:
                    threats.append(ThreatEvent(
                        timestamp=time.time(),
                        threat_type="behavioral_anomaly",
                        threat_level=ThreatLevel.MEDIUM,
                        source_id=client_id,
                        details={
                            'anomaly_score': anomaly_score,
                            'anomaly_type': 'request_rate',
                            'threshold': self.adaptive_thresholds['anomaly_score']
                        }
                    ))
        
        except Exception as e:
            logger.warning(f"Behavioral anomaly detection failed: {e}")
        
        return threats
    
    def _detect_attack_patterns(self, client_id: str, input_data: Any) -> List[ThreatEvent]:
        """Pattern-based attack detection."""
        threats = []
        
        try:
            import re
            
            # Convert input to searchable string
            if isinstance(input_data, str):
                search_str = input_data.lower()
            else:
                search_str = str(input_data).lower()
            
            # Check against known attack patterns
            for attack_type, patterns in self.known_attack_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, search_str, re.IGNORECASE):
                        threats.append(ThreatEvent(
                            timestamp=time.time(),
                            threat_type=f"pattern_match_{attack_type}",
                            threat_level=ThreatLevel.HIGH,
                            source_id=client_id,
                            details={
                                'attack_type': attack_type,
                                'matched_pattern': pattern,
                                'input_excerpt': search_str[:100]
                            }
                        ))
                        break  # One match per attack type sufficient
        
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        
        return threats
    
    def _update_adaptive_thresholds(self, threats: List[ThreatEvent]):
        """Update adaptive thresholds based on threat patterns."""
        if not threats:
            return
        
        # Increase thresholds if too many false positives
        # Decrease thresholds if missing actual attacks
        # (This is a simplified adaptive mechanism)
        
        threat_counts = defaultdict(int)
        for threat in threats:
            threat_counts[threat.threat_type] += 1
        
        # Simple adaptation: if we're generating too many threats, increase thresholds
        total_threats = len(threats)
        if total_threats > 5:  # Too many threats, likely false positives
            for threshold_name in self.adaptive_thresholds:
                self.adaptive_thresholds[threshold_name] *= (1 + self.learning_rate)
        elif total_threats == 0:  # No threats, possibly too conservative
            for threshold_name in self.adaptive_thresholds:
                self.adaptive_thresholds[threshold_name] *= (1 - self.learning_rate * 0.5)
    
    def block_client(self, client_id: str, duration: float = 300.0):
        """Block a client for specified duration."""
        profile = self.client_profiles[client_id]
        profile['blocked'] = True
        profile['block_expiry'] = time.time() + duration
        
        logger.warning(f"Client {client_id} blocked for {duration} seconds")
    
    def is_client_blocked(self, client_id: str) -> bool:
        """Check if client is currently blocked."""
        profile = self.client_profiles[client_id]
        
        if not profile['blocked']:
            return False
        
        if time.time() > profile.get('block_expiry', 0):
            profile['blocked'] = False
            return False
        
        return True
    
    def get_threat_summary(self, time_window: float = 3600.0) -> Dict[str, Any]:
        """Get threat detection summary for specified time window."""
        current_time = time.time()
        window_start = current_time - time_window
        
        recent_threats = [
            threat for threat in self.threat_events
            if threat.timestamp >= window_start
        ]
        
        threat_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.threat_type] += 1
            threat_levels[threat.threat_level.name] += 1
        
        return {
            'time_window': time_window,
            'total_threats': len(recent_threats),
            'threat_counts': dict(threat_counts),
            'threat_levels': dict(threat_levels),
            'active_clients': len(self.client_profiles),
            'blocked_clients': sum(1 for p in self.client_profiles.values() if p['blocked']),
            'adaptive_thresholds': self.adaptive_thresholds.copy()
        }