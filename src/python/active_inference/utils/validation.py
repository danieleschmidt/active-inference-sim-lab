"""
Input validation and error handling for active inference components.

This module provides comprehensive validation functions to ensure
data integrity and catch common errors early.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import wraps
import logging


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ActiveInferenceError(Exception):
    """Base exception for active inference specific errors.""" 
    pass


class ModelError(ActiveInferenceError):
    """Exception for generative model errors."""
    pass


class InferenceError(ActiveInferenceError):
    """Exception for inference engine errors."""
    pass


class PlanningError(ActiveInferenceError):
    """Exception for planning errors."""
    pass


def validate_array(arr: np.ndarray, 
                  name: str,
                  expected_shape: Optional[Tuple] = None,
                  expected_dtype: Optional[np.dtype] = None,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None,
                  allow_nan: bool = False,
                  allow_inf: bool = False) -> None:
    """
    Validate numpy array properties.
    
    Args:
        arr: Array to validate
        name: Name of the array for error messages
        expected_shape: Expected shape (None to skip check)
        expected_dtype: Expected data type (None to skip check)
        min_val: Minimum allowed value (None to skip check)
        max_val: Maximum allowed value (None to skip check)
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether infinite values are allowed
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array, got {type(arr)}")
    
    if arr.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValidationError(f"{name} has shape {arr.shape}, expected {expected_shape}")
    
    if expected_dtype is not None and arr.dtype != expected_dtype:
        raise ValidationError(f"{name} has dtype {arr.dtype}, expected {expected_dtype}")
    
    if not allow_nan and np.any(np.isnan(arr)):
        raise ValidationError(f"{name} contains NaN values")
    
    if not allow_inf and np.any(np.isinf(arr)):
        raise ValidationError(f"{name} contains infinite values")
    
    if min_val is not None and np.any(arr < min_val):
        raise ValidationError(f"{name} contains values below {min_val}")
    
    if max_val is not None and np.any(arr > max_val):
        raise ValidationError(f"{name} contains values above {max_val}")


def validate_matrix(matrix: np.ndarray,
                   name: str,
                   square: bool = False,
                   positive_definite: bool = False,
                   symmetric: bool = False) -> None:
    """
    Validate matrix properties.
    
    Args:
        matrix: Matrix to validate
        name: Name for error messages
        square: Whether matrix must be square
        positive_definite: Whether matrix must be positive definite
        symmetric: Whether matrix must be symmetric
        
    Raises:
        ValidationError: If validation fails
    """
    validate_array(matrix, name)
    
    if matrix.ndim != 2:
        raise ValidationError(f"{name} must be 2D, got {matrix.ndim}D")
    
    if square and matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(f"{name} must be square, got shape {matrix.shape}")
    
    if symmetric:
        if not square:
            raise ValidationError(f"{name} must be square to be symmetric")
        
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10):
            raise ValidationError(f"{name} is not symmetric")
    
    if positive_definite:
        if not square:
            raise ValidationError(f"{name} must be square to be positive definite")
        
        try:
            eigenvals = np.linalg.eigvals(matrix)
            if np.any(eigenvals <= 1e-10):
                raise ValidationError(f"{name} is not positive definite (min eigenvalue: {eigenvals.min()})")
        except np.linalg.LinAlgError:
            raise ValidationError(f"{name} eigenvalue computation failed - likely not positive definite")


def validate_probability_distribution(probs: np.ndarray, name: str) -> None:
    """
    Validate probability distribution.
    
    Args:
        probs: Probability values
        name: Name for error messages
        
    Raises:
        ValidationError: If not a valid probability distribution
    """
    validate_array(probs, name, min_val=0.0, max_val=1.0)
    
    prob_sum = np.sum(probs)
    if not np.isclose(prob_sum, 1.0, rtol=1e-6, atol=1e-6):
        raise ValidationError(f"{name} probabilities sum to {prob_sum}, expected 1.0")


def validate_belief_state(beliefs_dict: Dict[str, Any]) -> None:
    """
    Validate belief state dictionary.
    
    Args:
        beliefs_dict: Dictionary of belief components
        
    Raises:
        ValidationError: If belief state is invalid
    """
    if not isinstance(beliefs_dict, dict):
        raise ValidationError(f"Beliefs must be a dictionary, got {type(beliefs_dict)}")
    
    if len(beliefs_dict) == 0:
        raise ValidationError("Beliefs dictionary cannot be empty")
    
    for name, belief in beliefs_dict.items():
        if not hasattr(belief, 'mean') or not hasattr(belief, 'variance'):
            raise ValidationError(f"Belief '{name}' must have 'mean' and 'variance' attributes")
        
        try:
            validate_array(belief.mean, f"belief '{name}' mean")
            validate_array(belief.variance, f"belief '{name}' variance", min_val=0.0)
            
            # Check matching dimensions
            if belief.mean.shape != belief.variance.shape:
                raise ValidationError(
                    f"Belief '{name}' mean and variance shapes don't match: "
                    f"{belief.mean.shape} vs {belief.variance.shape}"
                )
        except AttributeError as e:
            raise ValidationError(f"Belief '{name}' has invalid structure: {e}")


def validate_dimensions(state_dim: int, obs_dim: int, action_dim: int) -> None:
    """
    Validate dimension parameters.
    
    Args:
        state_dim: Hidden state dimensionality
        obs_dim: Observation dimensionality  
        action_dim: Action dimensionality
        
    Raises:
        ValidationError: If dimensions are invalid
    """
    if not isinstance(state_dim, int) or state_dim <= 0:
        raise ValidationError(f"state_dim must be positive integer, got {state_dim}")
    
    if not isinstance(obs_dim, int) or obs_dim <= 0:
        raise ValidationError(f"obs_dim must be positive integer, got {obs_dim}")
    
    if not isinstance(action_dim, int) or action_dim <= 0:
        raise ValidationError(f"action_dim must be positive integer, got {action_dim}")
    
    # Reasonable limits to prevent memory issues
    MAX_DIM = 10000
    if state_dim > MAX_DIM:
        raise ValidationError(f"state_dim {state_dim} exceeds maximum {MAX_DIM}")
    
    if obs_dim > MAX_DIM:
        raise ValidationError(f"obs_dim {obs_dim} exceeds maximum {MAX_DIM}")
    
    if action_dim > MAX_DIM:
        raise ValidationError(f"action_dim {action_dim} exceeds maximum {MAX_DIM}")


def validate_hyperparameters(**kwargs) -> None:
    """
    Validate common hyperparameters.
    
    Args:
        **kwargs: Hyperparameters to validate
        
    Raises:
        ValidationError: If any hyperparameter is invalid
    """
    for name, value in kwargs.items():
        if name in ['learning_rate', 'temperature']:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{name} must be positive number, got {value}")
        
        elif name in ['horizon', 'max_iterations', 'n_samples']:
            if not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be positive integer, got {value}")
        
        elif name in ['convergence_threshold']:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{name} must be positive number, got {value}")


def validate_inputs(**kwargs) -> None:
    """
    General input validation function.
    
    Args:
        **kwargs: Named inputs to validate
        
    Raises:
        ValidationError: If any input is invalid
    """
    for name, value in kwargs.items():
        if value is None:
            continue  # Allow None values
        
        if name.endswith('_dim'):
            if not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be positive integer, got {value}")
        
        elif name.endswith('_array') or name in ['observation', 'action', 'state']:
            if isinstance(value, np.ndarray):
                validate_array(value, name)
        
        elif name.endswith('_matrix') or name in ['covariance', 'precision']:
            if isinstance(value, np.ndarray):
                validate_matrix(value, name)


def handle_errors(error_types: Tuple = (Exception,), 
                 default_return=None,
                 log_errors: bool = True):
    """
    Decorator for robust error handling.
    
    Args:
        error_types: Tuple of exception types to handle
        default_return: Value to return on error (if not re-raising)
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                if default_return is not None:
                    return default_return
                else:
                    raise
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
               epsilon: float = 1e-8) -> np.ndarray:
    """
    Safely divide arrays with numerical stability.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        epsilon: Small value to add for stability
        
    Returns:
        Division result with numerical stability
    """
    return numerator / (denominator + epsilon)


def safe_log(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Safely compute logarithm with numerical stability.
    
    Args:
        x: Input array
        epsilon: Small value to clamp minimum
        
    Returns:
        Logarithm with numerical stability
    """
    return np.log(np.maximum(x, epsilon))


def clip_values(x: np.ndarray, min_val: Optional[float] = None, 
               max_val: Optional[float] = None) -> np.ndarray:
    """
    Clip values to valid range.
    
    Args:
        x: Input array
        min_val: Minimum value (None to skip)
        max_val: Maximum value (None to skip)
        
    Returns:
        Clipped array
    """
    result = x.copy()
    
    if min_val is not None:
        result = np.maximum(result, min_val)
    
    if max_val is not None:
        result = np.minimum(result, max_val)
    
    return result