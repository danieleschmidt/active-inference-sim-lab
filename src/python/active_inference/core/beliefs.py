"""
Belief representation and manipulation for active inference.

This module implements belief states and their operations, including
uncertainty quantification and belief updating mechanics.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
import warnings
import traceback

from ..utils.validation import (
    ValidationError, ActiveInferenceError,
    validate_array, validate_matrix, validate_inputs, handle_errors,
    safe_log, safe_divide, clip_values
)


@dataclass
class Belief:
    """
    Represents a belief about a hidden state variable.
    
    Attributes:
        mean: Expected value of the belief
        variance: Uncertainty in the belief  
        support: Valid range/domain for the belief
        precision: Inverse of variance (1/variance)
    """
    mean: np.ndarray
    variance: np.ndarray
    support: Optional[tuple] = None
    
    def __post_init__(self):
        """Validate belief parameters after initialization.
        
        Raises:
            ValidationError: If belief parameters are invalid
        """
        try:
            # Validate mean
            validate_array(self.mean, "belief mean")
            
            # Validate variance
            validate_array(self.variance, "belief variance", min_val=1e-12)
            
            # Check shape compatibility
            if self.mean.shape != self.variance.shape:
                raise ValidationError(
                    f"Mean and variance shapes must match: {self.mean.shape} vs {self.variance.shape}"
                )
            
            # Validate support if provided
            if self.support is not None:
                if not isinstance(self.support, (tuple, list)) or len(self.support) != 2:
                    raise ValidationError("Support must be a tuple/list of length 2 (min, max)")
                
                min_val, max_val = self.support
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    raise ValidationError("Support bounds must be numeric")
                
                if min_val >= max_val:
                    raise ValidationError(f"Support min ({min_val}) must be less than max ({max_val})")
            
            # Ensure numerical stability
            self.variance = np.maximum(self.variance, 1e-12)
            
            # Check for NaN/Inf values
            if np.any(np.isnan(self.mean)) or np.any(np.isinf(self.mean)):
                raise ValidationError("Belief mean contains NaN or infinite values")
            
            if np.any(np.isnan(self.variance)) or np.any(np.isinf(self.variance)):
                raise ValidationError("Belief variance contains NaN or infinite values")
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to validate belief: {e}")
    
    @property
    def precision(self) -> np.ndarray:
        """Precision (inverse variance) of the belief.
        
        Returns:
            Precision matrix with numerical stability
        """
        try:
            # Use safe division for numerical stability
            return safe_divide(np.ones_like(self.variance), self.variance, epsilon=1e-12)
        except Exception as e:
            warnings.warn(f"Error computing precision: {e}")
            # Fallback to basic computation
            return 1.0 / np.maximum(self.variance, 1e-12)
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief (measure of uncertainty).
        
        Returns:
            Entropy value (always non-negative)
        """
        try:
            # Use safe logarithm to prevent numerical issues
            log_terms = safe_log(2 * np.pi * np.e * self.variance)
            entropy = 0.5 * np.sum(log_terms)
            
            # Ensure entropy is non-negative
            return max(0.0, float(entropy))
            
        except Exception as e:
            warnings.warn(f"Error computing entropy: {e}")
            # Fallback: simple variance-based uncertainty
            return float(np.sum(self.variance))
    
    @property
    def confidence(self) -> float:
        """Confidence level (inverse of entropy).
        
        Returns:
            Confidence value between 0 and 1
        """
        try:
            entropy = self.entropy
            # Bound confidence between 0 and 1
            confidence = 1.0 / (1.0 + entropy)
            return clip_values(np.array([confidence]), 0.0, 1.0)[0]
            
        except Exception as e:
            warnings.warn(f"Error computing confidence: {e}")
            # Fallback: inverse of average variance
            avg_variance = np.mean(self.variance)
            return 1.0 / (1.0 + avg_variance)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the belief distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples from the belief distribution
            
        Raises:
            ValidationError: If n_samples is invalid
        """
        try:
            # Validate input
            if not isinstance(n_samples, int) or n_samples <= 0:
                raise ValidationError(f"n_samples must be positive integer, got {n_samples}")
            
            if n_samples > 100000:
                raise ValidationError(f"n_samples too large ({n_samples}), maximum is 100000")
            
            # For 1D or independent sampling
            if len(self.mean.shape) == 1 and len(self.mean) <= 1:
                # Simple case: independent sampling
                std_dev = np.sqrt(np.maximum(self.variance, 1e-12))
                samples = np.random.normal(
                    self.mean,
                    std_dev,
                    size=(n_samples,) + self.mean.shape
                )
                return samples if n_samples > 1 else samples[0]
            
            # Multivariate case with regularization for numerical stability
            variance_matrix = np.diag(self.variance.flatten()) + 1e-8 * np.eye(len(self.mean.flatten()))
            
            try:
                samples = np.random.multivariate_normal(
                    self.mean.flatten(), 
                    variance_matrix,
                    size=n_samples
                )
                
                # Reshape to original shape
                if n_samples == 1:
                    return samples.reshape(self.mean.shape)
                else:
                    return samples.reshape((n_samples,) + self.mean.shape)
                    
            except np.linalg.LinAlgError:
                # Fallback: sample each dimension independently
                std_dev = np.sqrt(np.maximum(self.variance, 1e-12))
                samples = np.random.normal(
                    self.mean,
                    std_dev,
                    size=(n_samples,) + self.mean.shape
                )
                return samples if n_samples > 1 else samples[0]
                
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Failed to sample from belief: {e}")
    
    def log_probability(self, x: np.ndarray) -> float:
        """Compute log probability of observation under this belief.
        
        Args:
            x: Observation to evaluate
            
        Returns:
            Log probability value
            
        Raises:
            ValidationError: If observation shape is invalid
        """
        try:
            # Validate input
            validate_array(x, "observation")
            
            if x.shape != self.mean.shape:
                raise ValidationError(
                    f"Observation shape {x.shape} doesn't match belief shape {self.mean.shape}"
                )
            
            # Compute difference
            diff = x - self.mean
            
            # Compute log probability with numerical stability
            precision_diag = self.precision
            
            # Quadratic term
            quad_term = np.sum(diff * precision_diag * diff)
            
            # Normalization term (log determinant)
            log_det_term = np.sum(safe_log(2 * np.pi * self.variance))
            
            log_prob = -0.5 * (quad_term + log_det_term)
            
            # Check for numerical issues
            if np.isnan(log_prob) or np.isinf(log_prob):
                warnings.warn("Log probability computation resulted in NaN/Inf")
                # Fallback: simplified computation
                mse = np.mean((diff)**2)
                return -mse  # Negative MSE as proxy
            
            return float(log_prob)
            
        except ValidationError:
            raise
        except Exception as e:
            warnings.warn(f"Error computing log probability: {e}")
            # Fallback: negative squared error
            try:
                diff = x - self.mean
                return float(-np.sum(diff**2))
            except:
                return -np.inf
    
    def distance_to(self, other: 'Belief') -> float:
        """Compute distance to another belief (KL divergence approximation).
        
        Args:
            other: Another belief to compare with
            
        Returns:
            Distance measure (non-negative)
        """
        try:
            if not isinstance(other, Belief):
                raise ValidationError("Can only compute distance to another Belief")
            
            if self.mean.shape != other.mean.shape:
                raise ValidationError("Beliefs must have same shape for distance computation")
            
            # Approximate KL divergence for Gaussian distributions
            mean_diff = self.mean - other.mean
            
            # Use safe operations
            var_ratio = safe_divide(other.variance, self.variance, epsilon=1e-12)
            log_var_ratio = safe_log(var_ratio)
            
            # KL divergence approximation
            kl_div = 0.5 * np.sum(
                var_ratio + 
                (mean_diff**2) / np.maximum(self.variance, 1e-12) - 
                1.0 - 
                log_var_ratio
            )
            
            return max(0.0, float(kl_div))
            
        except Exception as e:
            warnings.warn(f"Error computing belief distance: {e}")
            # Fallback: Euclidean distance of means
            try:
                return float(np.linalg.norm(self.mean - other.mean))
            except:
                return np.inf
    
    def is_valid(self) -> bool:
        """Check if belief is in a valid state.
        
        Returns:
            True if belief is valid, False otherwise
        """
        try:
            # Check for NaN/Inf values
            if np.any(np.isnan(self.mean)) or np.any(np.isinf(self.mean)):
                return False
            
            if np.any(np.isnan(self.variance)) or np.any(np.isinf(self.variance)):
                return False
            
            # Check for positive variance
            if np.any(self.variance <= 0):
                return False
            
            # Check shape compatibility
            if self.mean.shape != self.variance.shape:
                return False
            
            return True
            
        except Exception:
            return False


class BeliefState:
    """
    Container for multiple beliefs about different aspects of the world state.
    
    This class manages a collection of beliefs about different hidden state
    variables and provides methods for accessing and updating them.
    """
    
    def __init__(self, max_history_length: int = 1000):
        """
        Initialize belief state.
        
        Args:
            max_history_length: Maximum number of history snapshots to keep
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            if not isinstance(max_history_length, int) or max_history_length <= 0:
                raise ValidationError(f"max_history_length must be positive integer, got {max_history_length}")
            
            self._beliefs: Dict[str, Belief] = {}
            self._history: list = []
            self._max_history_length = max_history_length
            self._error_count = 0
            self._last_error = None
            
            # Setup logging
            self.logger = logging.getLogger("BeliefState")
            
        except Exception as e:
            raise ValidationError(f"Failed to initialize BeliefState: {e}")
    
    def add_belief(self, name: str, belief: Belief) -> None:
        """Add a named belief to the state.
        
        Args:
            name: Name for the belief
            belief: Belief object to add
            
        Raises:
            ValidationError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("Belief name must be a non-empty string")
            
            if not isinstance(belief, Belief):
                raise ValidationError(f"Expected Belief object, got {type(belief)}")
            
            # Check if belief is valid
            if not belief.is_valid():
                raise ValidationError(f"Invalid belief for '{name}': belief failed validation")
            
            # Store the belief
            self._beliefs[name.strip()] = belief
            self.logger.debug(f"Added belief '{name}' with shape {belief.mean.shape}")
            
        except ValidationError:
            self._record_error("add_belief", name)
            raise
        except Exception as e:
            self._record_error("add_belief", name)
            raise ValidationError(f"Failed to add belief '{name}': {e}")
    
    def get_belief(self, name: str) -> Optional[Belief]:
        """Get a specific belief by name.
        
        Args:
            name: Name of the belief
            
        Returns:
            Belief object if found, None otherwise
            
        Raises:
            ValidationError: If name is invalid
        """
        try:
            if not isinstance(name, str):
                raise ValidationError(f"Belief name must be string, got {type(name)}")
            
            belief = self._beliefs.get(name.strip())
            
            # Validate belief if found
            if belief is not None and not belief.is_valid():
                self.logger.warning(f"Retrieved belief '{name}' is invalid")
                return None
            
            return belief
            
        except ValidationError:
            self._record_error("get_belief", name)
            raise
        except Exception as e:
            self._record_error("get_belief", name)
            self.logger.error(f"Error getting belief '{name}': {e}")
            return None
    
    def update_belief(self, name: str, new_mean: np.ndarray, 
                     new_variance: np.ndarray, support: Optional[tuple] = None) -> None:
        """Update an existing belief with new parameters.
        
        Args:
            name: Name of the belief to update
            new_mean: New mean value
            new_variance: New variance value
            support: Optional support bounds
            
        Raises:
            ValidationError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("Belief name must be a non-empty string")
            
            validate_array(new_mean, "new_mean")
            validate_array(new_variance, "new_variance", min_val=1e-12)
            
            # Check shape compatibility
            if new_mean.shape != new_variance.shape:
                raise ValidationError(
                    f"Mean and variance shapes must match: {new_mean.shape} vs {new_variance.shape}"
                )
            
            name = name.strip()
            
            # Create new belief
            new_belief = Belief(
                mean=new_mean.copy(),
                variance=new_variance.copy(),
                support=support
            )
            
            # Validate the new belief
            if not new_belief.is_valid():
                raise ValidationError(f"New belief parameters for '{name}' are invalid")
            
            # Update existing or add new
            if name in self._beliefs:
                self._beliefs[name] = new_belief
                self.logger.debug(f"Updated belief '{name}'")
            else:
                self._beliefs[name] = new_belief
                self.logger.debug(f"Added new belief '{name}'")
                
        except ValidationError:
            self._record_error("update_belief", name)
            raise
        except Exception as e:
            self._record_error("update_belief", name)
            raise ValidationError(f"Failed to update belief '{name}': {e}")
    
    def get_all_beliefs(self) -> Dict[str, Belief]:
        """Get all beliefs in the state.
        
        Returns:
            Copy of all beliefs dictionary
        """
        try:
            # Validate all beliefs before returning
            valid_beliefs = {}
            invalid_beliefs = []
            
            for name, belief in self._beliefs.items():
                if belief.is_valid():
                    valid_beliefs[name] = belief
                else:
                    invalid_beliefs.append(name)
                    self.logger.warning(f"Belief '{name}' is invalid, excluding from results")
            
            if invalid_beliefs:
                self.logger.warning(f"Found {len(invalid_beliefs)} invalid beliefs: {invalid_beliefs}")
            
            return valid_beliefs.copy()
            
        except Exception as e:
            self.logger.error(f"Error getting all beliefs: {e}")
            # Return empty dict as fallback
            return {}
    
    def total_entropy(self) -> float:
        """Compute total entropy across all beliefs.
        
        Returns:
            Total entropy (non-negative)
        """
        try:
            if not self._beliefs:
                return 0.0
            
            total = 0.0
            valid_count = 0
            
            for name, belief in self._beliefs.items():
                if belief.is_valid():
                    entropy = belief.entropy
                    if not (np.isnan(entropy) or np.isinf(entropy)):
                        total += entropy
                        valid_count += 1
                    else:
                        self.logger.warning(f"Belief '{name}' has invalid entropy: {entropy}")
                else:
                    self.logger.warning(f"Belief '{name}' is invalid, skipping entropy calculation")
            
            return max(0.0, total)
            
        except Exception as e:
            self.logger.error(f"Error computing total entropy: {e}")
            return 0.0
    
    def average_confidence(self) -> float:
        """Compute average confidence across all beliefs.
        
        Returns:
            Average confidence between 0 and 1
        """
        try:
            if not self._beliefs:
                return 0.0
            
            confidences = []
            
            for name, belief in self._beliefs.items():
                if belief.is_valid():
                    confidence = belief.confidence
                    if not (np.isnan(confidence) or np.isinf(confidence)):
                        confidences.append(confidence)
                    else:
                        self.logger.warning(f"Belief '{name}' has invalid confidence: {confidence}")
                else:
                    self.logger.warning(f"Belief '{name}' is invalid, skipping confidence calculation")
            
            if not confidences:
                return 0.0
            
            avg_confidence = np.mean(confidences)
            return clip_values(np.array([avg_confidence]), 0.0, 1.0)[0]
            
        except Exception as e:
            self.logger.error(f"Error computing average confidence: {e}")
            return 0.0
    
    def save_snapshot(self) -> None:
        """Save current state to history.
        
        Raises:
            ValidationError: If snapshot creation fails
        """
        try:
            snapshot = {
                'timestamp': np.datetime64('now'),
                'beliefs': {},
                'total_entropy': 0.0,
                'average_confidence': 0.0
            }
            
            # Save beliefs that are valid
            for name, belief in self._beliefs.items():
                if belief.is_valid():
                    try:
                        snapshot['beliefs'][name] = {
                            'mean': belief.mean.copy(),
                            'variance': belief.variance.copy(),
                            'entropy': belief.entropy,
                            'confidence': belief.confidence
                        }
                    except Exception as e:
                        self.logger.warning(f"Failed to snapshot belief '{name}': {e}")
                else:
                    self.logger.warning(f"Skipping invalid belief '{name}' in snapshot")
            
            # Save aggregate statistics
            snapshot['total_entropy'] = self.total_entropy()
            snapshot['average_confidence'] = self.average_confidence()
            
            self._history.append(snapshot)
            
            # Trim history if too long
            if len(self._history) > self._max_history_length:
                trim_size = self._max_history_length // 2
                self._history = self._history[-trim_size:]
                self.logger.debug(f"Trimmed belief history to {trim_size} snapshots")
            
        except Exception as e:
            self._record_error("save_snapshot", "")
            raise ValidationError(f"Failed to save belief snapshot: {e}")
    
    def get_history(self) -> list:
        """Get belief evolution history.
        
        Returns:
            Copy of history list
        """
        try:
            return self._history.copy()
        except Exception as e:
            self.logger.error(f"Error getting history: {e}")
            return []
    
    def _record_error(self, operation: str, belief_name: str) -> None:
        """Record error for monitoring.
        
        Args:
            operation: Type of operation that failed
            belief_name: Name of belief involved
        """
        self._error_count += 1
        self._last_error = {
            'operation': operation,
            'belief_name': belief_name,
            'timestamp': np.datetime64('now')
        }
        self.logger.error(f"BeliefState error in {operation} for belief '{belief_name}'")
    
    def validate_all_beliefs(self) -> Dict[str, bool]:
        """Validate all beliefs and return status.
        
        Returns:
            Dictionary mapping belief names to validation status
        """
        try:
            validation_results = {}
            
            for name, belief in self._beliefs.items():
                try:
                    validation_results[name] = belief.is_valid()
                except Exception as e:
                    self.logger.warning(f"Error validating belief '{name}': {e}")
                    validation_results[name] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating beliefs: {e}")
            return {}
    
    def remove_invalid_beliefs(self) -> int:
        """Remove all invalid beliefs from the state.
        
        Returns:
            Number of beliefs removed
        """
        try:
            invalid_beliefs = []
            
            for name, belief in self._beliefs.items():
                if not belief.is_valid():
                    invalid_beliefs.append(name)
            
            for name in invalid_beliefs:
                del self._beliefs[name]
                self.logger.info(f"Removed invalid belief '{name}'")
            
            return len(invalid_beliefs)
            
        except Exception as e:
            self.logger.error(f"Error removing invalid beliefs: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get belief state statistics.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            validation_results = self.validate_all_beliefs()
            valid_count = sum(validation_results.values())
            total_count = len(self._beliefs)
            
            return {
                'total_beliefs': total_count,
                'valid_beliefs': valid_count,
                'invalid_beliefs': total_count - valid_count,
                'total_entropy': self.total_entropy(),
                'average_confidence': self.average_confidence(),
                'history_length': len(self._history),
                'error_count': self._error_count,
                'last_error': self._last_error
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def __len__(self) -> int:
        """Number of beliefs in the state."""
        return len(self._beliefs)
    
    def __contains__(self, name: str) -> bool:
        """Check if a belief exists.
        
        Args:
            name: Name to check
            
        Returns:
            True if belief exists and is valid
        """
        try:
            if not isinstance(name, str):
                return False
            
            belief = self._beliefs.get(name.strip())
            return belief is not None and belief.is_valid()
            
        except Exception:
            return False
    
    def __getitem__(self, name: str) -> Belief:
        """Get belief by name using bracket notation.
        
        Args:
            name: Name of belief to retrieve
            
        Returns:
            Belief object
            
        Raises:
            KeyError: If belief doesn't exist
            ValidationError: If belief is invalid
        """
        try:
            if not isinstance(name, str):
                raise ValidationError(f"Belief name must be string, got {type(name)}")
            
            name = name.strip()
            
            if name not in self._beliefs:
                raise KeyError(f"Belief '{name}' not found")
            
            belief = self._beliefs[name]
            
            if not belief.is_valid():
                raise ValidationError(f"Belief '{name}' is in invalid state")
            
            return belief
            
        except (KeyError, ValidationError):
            raise
        except Exception as e:
            raise ValidationError(f"Error accessing belief '{name}': {e}")
    
    def __setitem__(self, name: str, belief: Belief) -> None:
        """Set belief by name using bracket notation.
        
        Args:
            name: Name for the belief
            belief: Belief object to store
            
        Raises:
            ValidationError: If inputs are invalid
        """
        self.add_belief(name, belief)  # Use the validated add_belief method