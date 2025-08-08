"""
Intelligent caching system for Active Inference components.

This module provides adaptive caching strategies to optimize
performance by reusing expensive computations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging
import hashlib
import pickle
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass

from ..core.beliefs import Belief, BeliefState
from ..core.generative_model import GenerativeModel


class CacheStrategy(Enum):
    """Caching strategy options."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on hit rate


@dataclass
class CacheEntry:
    """Entry in cache with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int


class BaseCache:
    """Base class for caching implementations."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            strategy: Caching strategy
        """
        self.max_size = max_size
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        current_time = time.time()
        
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=current_time,
            access_count=1,
            last_access=current_time,
            size_bytes=size_bytes
        )
        
        # Add to cache
        if key in self.cache:
            # Update existing entry
            self.cache[key] = entry
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = entry
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self._evict()
    
    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_access = min(entry.access_count for entry in self.cache.values())
            for key, entry in list(self.cache.items()):
                if entry.access_count == min_access:
                    del self.cache[key]
                    break
        elif self.strategy == CacheStrategy.TTL:
            # Remove oldest entries
            current_time = time.time()
            ttl = 3600  # 1 hour default TTL
            
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry.timestamp > ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            # If no expired entries, fall back to LRU
            if not expired_keys and len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._adaptive_evict()
        
        self.evictions += 1
    
    def _adaptive_evict(self) -> None:
        """Adaptive eviction based on access patterns."""
        current_time = time.time()
        
        # Calculate scores for each entry
        scores = {}
        for key, entry in self.cache.items():
            # Score based on recency, frequency, and size
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            frequency_score = entry.access_count
            size_penalty = entry.size_bytes / 1024  # Size in KB
            
            scores[key] = recency_score * frequency_score / size_penalty
        
        # Remove entry with lowest score
        worst_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[worst_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate(),
            'total_size_bytes': total_size,
            'average_entry_size': total_size / max(1, len(self.cache)),
            'strategy': self.strategy.value
        }
    
    def clear_old_entries(self, age_threshold: float = 3600) -> int:
        """Clear entries older than threshold."""
        current_time = time.time()
        old_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > age_threshold
        ]
        
        for key in old_keys:
            del self.cache[key]
        
        return len(old_keys)


class BeliefCache(BaseCache):
    """Specialized cache for belief states."""
    
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size, CacheStrategy.ADAPTIVE)
        
    def generate_belief_key(self, 
                           observation: np.ndarray,
                           prior_beliefs: BeliefState) -> str:
        """Generate cache key for belief update."""
        
        # Hash observation
        obs_hash = hashlib.md5(observation.tobytes()).hexdigest()[:8]
        
        # Hash prior beliefs
        belief_data = []
        for name, belief in prior_beliefs.get_all_beliefs().items():
            belief_data.append(f"{name}:{belief.mean.tobytes()}:{belief.variance.tobytes()}")
        
        beliefs_str = "|".join(belief_data)
        beliefs_hash = hashlib.md5(beliefs_str.encode()).hexdigest()[:8]
        
        return f"belief_{obs_hash}_{beliefs_hash}"
    
    def cache_belief_update(self,
                           observation: np.ndarray,
                           prior_beliefs: BeliefState,
                           updated_beliefs: BeliefState) -> None:
        """Cache belief update result."""
        
        key = self.generate_belief_key(observation, prior_beliefs)
        self.put(key, updated_beliefs)
    
    def get_cached_belief_update(self,
                                observation: np.ndarray,
                                prior_beliefs: BeliefState) -> Optional[BeliefState]:
        """Get cached belief update result."""
        
        key = self.generate_belief_key(observation, prior_beliefs)
        return self.get(key)


class ModelCache(BaseCache):
    """Specialized cache for model computations."""
    
    def __init__(self, max_size: int = 500):
        super().__init__(max_size, CacheStrategy.LFU)
    
    def generate_likelihood_key(self,
                               state: np.ndarray,
                               observation: np.ndarray) -> str:
        """Generate cache key for likelihood computation."""
        
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:8]
        obs_hash = hashlib.md5(observation.tobytes()).hexdigest()[:8]
        
        return f"likelihood_{state_hash}_{obs_hash}"
    
    def cache_likelihood(self,
                        state: np.ndarray,
                        observation: np.ndarray,
                        likelihood: float) -> None:
        """Cache likelihood computation."""
        
        key = self.generate_likelihood_key(state, observation)
        self.put(key, likelihood)
    
    def get_cached_likelihood(self,
                             state: np.ndarray,
                             observation: np.ndarray) -> Optional[float]:
        """Get cached likelihood."""
        
        key = self.generate_likelihood_key(state, observation)
        return self.get(key)
    
    def generate_dynamics_key(self,
                             state: np.ndarray,
                             action: np.ndarray) -> str:
        """Generate cache key for dynamics prediction."""
        
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:8]
        action_hash = hashlib.md5(action.tobytes()).hexdigest()[:8]
        
        return f"dynamics_{state_hash}_{action_hash}"
    
    def cache_dynamics(self,
                      state: np.ndarray,
                      action: np.ndarray,
                      next_state: np.ndarray) -> None:
        """Cache dynamics prediction."""
        
        key = self.generate_dynamics_key(state, action)
        self.put(key, next_state)
    
    def get_cached_dynamics(self,
                           state: np.ndarray,
                           action: np.ndarray) -> Optional[np.ndarray]:
        """Get cached dynamics prediction."""
        
        key = self.generate_dynamics_key(state, action)
        return self.get(key)


class AdaptiveCache(BaseCache):
    """
    Adaptive cache that adjusts strategy based on performance.
    
    Monitors hit rates and automatically switches between
    caching strategies to optimize performance.
    """
    
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size, CacheStrategy.ADAPTIVE)
        
        # Strategy performance tracking
        self.strategy_stats = {
            strategy: {'hits': 0, 'misses': 0, 'score': 0.0}
            for strategy in CacheStrategy
        }
        
        self.current_strategy = CacheStrategy.LRU
        self.adaptation_interval = 1000  # Adapt every N operations
        self.operations_count = 0
        
        # Performance history
        self.performance_history = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get with adaptive strategy tracking."""
        self.operations_count += 1
        
        result = super().get(key)
        
        # Update strategy stats
        if result is not None:
            self.strategy_stats[self.current_strategy]['hits'] += 1
        else:
            self.strategy_stats[self.current_strategy]['misses'] += 1
        
        # Adapt strategy if needed
        if self.operations_count % self.adaptation_interval == 0:
            self._adapt_strategy()
        
        return result
    
    def _adapt_strategy(self) -> None:
        """Adapt caching strategy based on performance."""
        
        # Calculate current performance
        current_hit_rate = self.hit_rate()
        self.performance_history.append(current_hit_rate)
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Try different strategy if performance is declining
        if len(self.performance_history) >= 3:
            recent_trend = np.mean(self.performance_history[-3:])
            older_trend = np.mean(self.performance_history[-6:-3]) if len(self.performance_history) >= 6 else recent_trend
            
            if recent_trend < older_trend * 0.95:  # 5% decline
                self._try_new_strategy()
        
        self.logger.debug(f"Cache adaptation: strategy={self.current_strategy.value}, "
                         f"hit_rate={current_hit_rate:.3f}")
    
    def _try_new_strategy(self) -> None:
        """Try a new caching strategy."""
        
        # Calculate scores for each strategy
        for strategy, stats in self.strategy_stats.items():
            total = stats['hits'] + stats['misses']
            if total > 0:
                hit_rate = stats['hits'] / total
                # Score includes hit rate and recency bias
                recency_bonus = 1.0 if strategy == self.current_strategy else 0.9
                stats['score'] = hit_rate * recency_bonus
        
        # Find best performing strategy (excluding current)
        alternative_strategies = [s for s in CacheStrategy if s != self.current_strategy]
        if alternative_strategies:
            best_strategy = max(alternative_strategies, 
                              key=lambda s: self.strategy_stats[s]['score'])
            
            # Switch if significantly better
            current_score = self.strategy_stats[self.current_strategy]['score']
            best_score = self.strategy_stats[best_strategy]['score']
            
            if best_score > current_score * 1.05:  # 5% improvement threshold
                self.logger.info(f"Switching cache strategy from {self.current_strategy.value} "
                               f"to {best_strategy.value}")
                self.current_strategy = best_strategy
                self.strategy = best_strategy
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptive cache statistics."""
        
        stats = self.get_stats()
        
        # Add adaptation-specific metrics
        stats.update({
            'current_strategy': self.current_strategy.value,
            'operations_count': self.operations_count,
            'adaptation_interval': self.adaptation_interval,
            'performance_history': self.performance_history[-5:],  # Recent history
            'strategy_performance': {
                strategy.value: {
                    'hit_rate': stats_data['hits'] / max(1, stats_data['hits'] + stats_data['misses']),
                    'total_operations': stats_data['hits'] + stats_data['misses'],
                    'score': stats_data['score']
                }
                for strategy, stats_data in self.strategy_stats.items()
            }
        })
        
        return stats


class CacheManager:
    """Manager for multiple cache instances."""
    
    def __init__(self):
        self.caches: Dict[str, BaseCache] = {}
        self.logger = logging.getLogger("CacheManager")
    
    def register_cache(self, name: str, cache: BaseCache) -> None:
        """Register a cache instance."""
        self.caches[name] = cache
        self.logger.info(f"Registered cache: {name}")
    
    def get_cache(self, name: str) -> Optional[BaseCache]:
        """Get cache by name."""
        return self.caches.get(name)
    
    def clear_all_caches(self) -> None:
        """Clear all registered caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("Cleared all caches")
    
    def get_global_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            name: cache.get_stats()
            for name, cache in self.caches.items()
        }
    
    def optimize_cache_sizes(self, 
                            total_memory_budget: int = 512 * 1024 * 1024) -> Dict[str, int]:
        """
        Optimize cache sizes based on usage patterns and memory budget.
        
        Args:
            total_memory_budget: Total memory budget in bytes
            
        Returns:
            Dictionary of cache_name -> recommended_size
        """
        
        # Get current usage statistics
        stats = self.get_global_stats()
        
        # Calculate relative importance based on hit rates and usage
        importance_scores = {}
        total_score = 0
        
        for name, cache_stats in stats.items():
            hit_rate = cache_stats.get('hit_rate', 0)
            total_ops = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
            
            # Importance score combines hit rate and usage frequency
            score = hit_rate * np.log(total_ops + 1)
            importance_scores[name] = score
            total_score += score
        
        # Allocate memory budget proportionally
        recommendations = {}
        
        for name, score in importance_scores.items():
            if total_score > 0:
                proportion = score / total_score
                memory_allocation = int(total_memory_budget * proportion)
                
                # Estimate entries per byte (rough approximation)
                avg_entry_size = stats[name].get('average_entry_size', 1024)
                recommended_size = max(100, memory_allocation // avg_entry_size)
                
                recommendations[name] = recommended_size
            else:
                recommendations[name] = 1000  # Default
        
        return recommendations
    
    def apply_recommendations(self, recommendations: Dict[str, int]) -> None:
        """Apply cache size recommendations."""
        
        for name, recommended_size in recommendations.items():
            if name in self.caches:
                cache = self.caches[name]
                old_size = cache.max_size
                cache.max_size = recommended_size
                
                # Evict excess entries if needed
                while len(cache.cache) > recommended_size:
                    cache._evict()
                
                self.logger.info(f"Updated {name} cache size: {old_size} -> {recommended_size}")


# Global cache manager instance
cache_manager = CacheManager()