"""
Grid world environment for active inference research.

This module implements various grid world environments specifically designed
to test active inference capabilities like exploration, goal-seeking, and
uncertainty handling.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import logging


class CellType(Enum):
    """Types of cells in the grid world."""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    HAZARD = 3
    UNCERTAIN = 4  # Cells with high observation noise


class ActiveInferenceGridWorld:
    """
    Grid world environment designed for active inference research.
    
    Features:
    - Partial observability through limited vision range
    - Observation uncertainty in specific regions
    - Multiple goals with different preferences
    - Dynamic elements (moving goals, changing hazards)
    """
    
    def __init__(self,
                 size: int = 10,
                 vision_range: int = 2,
                 observation_noise: float = 0.1,
                 uncertainty_regions: Optional[List[Tuple[int, int]]] = None,
                 goal_locations: Optional[List[Tuple[int, int]]] = None,
                 hazard_locations: Optional[List[Tuple[int, int]]] = None,
                 dynamic_goals: bool = False):
        """
        Initialize active inference grid world.
        
        Args:
            size: Grid size (size x size)
            vision_range: How far agent can see
            observation_noise: Base observation noise level
            uncertainty_regions: Locations with high observation noise
            goal_locations: Goal positions
            hazard_locations: Hazard positions
            dynamic_goals: Whether goals move over time
        """
        self.size = size
        self.vision_range = vision_range
        self.observation_noise = observation_noise
        self.dynamic_goals = dynamic_goals
        
        # Initialize grid
        self.grid = np.zeros((size, size), dtype=int)
        
        # Agent state
        self.agent_pos = np.array([1, 1])
        self.agent_orientation = 0  # 0=North, 1=East, 2=South, 3=West
        
        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Setup environment
        self._setup_environment(uncertainty_regions, goal_locations, hazard_locations)
        
        # History for analysis
        self.position_history = []
        self.reward_history = []
        self.uncertainty_history = []
        
        # Logging
        self.logger = logging.getLogger("GridWorld")
    
    def _setup_environment(self,
                          uncertainty_regions: Optional[List[Tuple[int, int]]],
                          goal_locations: Optional[List[Tuple[int, int]]],
                          hazard_locations: Optional[List[Tuple[int, int]]]):
        """Setup the grid world layout."""
        # Add walls around border
        self.grid[0, :] = CellType.WALL.value
        self.grid[-1, :] = CellType.WALL.value
        self.grid[:, 0] = CellType.WALL.value
        self.grid[:, -1] = CellType.WALL.value
        
        # Add some internal walls for complexity
        if self.size >= 8:
            # Add L-shaped wall
            self.grid[3:6, 3] = CellType.WALL.value
            self.grid[5, 3:6] = CellType.WALL.value
        
        # Set uncertainty regions
        self.uncertainty_regions = uncertainty_regions or []
        for x, y in self.uncertainty_regions:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x, y] = CellType.UNCERTAIN.value
        
        # Set goals
        self.goal_locations = goal_locations or [(self.size-2, self.size-2)]
        for x, y in self.goal_locations:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x, y] = CellType.GOAL.value
        
        # Set hazards
        self.hazard_locations = hazard_locations or []
        for x, y in self.hazard_locations:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x, y] = CellType.HAZARD.value
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        # Reset agent position (avoid walls and hazards)
        while True:
            self.agent_pos = np.array([
                np.random.randint(1, self.size-1),
                np.random.randint(1, self.size-1)
            ])
            if self.grid[tuple(self.agent_pos)] not in [CellType.WALL.value, CellType.HAZARD.value]:
                break
        
        self.agent_orientation = np.random.randint(0, 4)
        self.step_count = 0
        self.episode_count += 1
        self.total_reward = 0.0
        
        # Clear history
        self.position_history = []
        self.reward_history = []
        self.uncertainty_history = []
        
        # Move goals if dynamic
        if self.dynamic_goals:
            self._move_goals()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.step_count += 1
        
        # Convert action to discrete movement
        # action[0] = forward/backward, action[1] = turn left/right
        move_forward = action[0] > 0.0
        turn_direction = 0
        if len(action) > 1:
            if action[1] > 0.3:
                turn_direction = 1  # Turn right
            elif action[1] < -0.3:
                turn_direction = -1  # Turn left
        
        # Execute turn
        if turn_direction != 0:
            self.agent_orientation = (self.agent_orientation + turn_direction) % 4
        
        # Execute movement
        if move_forward:
            new_pos = self._get_next_position()
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
        
        # Compute reward
        reward = self._compute_reward()
        self.total_reward += reward
        
        # Check termination
        terminated = self._is_goal_reached()
        truncated = self.step_count >= 200  # Maximum episode length
        
        # Record history
        self.position_history.append(self.agent_pos.copy())
        self.reward_history.append(reward)
        self.uncertainty_history.append(self._get_observation_uncertainty())
        
        # Move goals if dynamic
        if self.dynamic_goals and self.step_count % 10 == 0:
            self._move_goals()
        
        # Info
        info = {
            'agent_pos': self.agent_pos.copy(),
            'agent_orientation': self.agent_orientation,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'observation_uncertainty': self._get_observation_uncertainty(),
            'distance_to_goal': self._distance_to_nearest_goal(),
            'visited_positions': len(set(map(tuple, self.position_history))),
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_next_position(self) -> np.ndarray:
        """Get next position based on current orientation."""
        direction_vectors = {
            0: np.array([-1, 0]),  # North
            1: np.array([0, 1]),   # East
            2: np.array([1, 0]),   # South
            3: np.array([0, -1])   # West
        }
        return self.agent_pos + direction_vectors[self.agent_orientation]
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (not wall, within bounds)."""
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return self.grid[x, y] != CellType.WALL.value
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with partial observability."""
        obs_size = (2 * self.vision_range + 1)
        observation = np.zeros((obs_size, obs_size, 4))  # 4 channels: walls, goals, hazards, agent
        
        agent_x, agent_y = self.agent_pos
        
        # Fill observation based on vision range
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                world_x, world_y = agent_x + dx, agent_y + dy
                obs_x, obs_y = dx + self.vision_range, dy + self.vision_range
                
                if 0 <= world_x < self.size and 0 <= world_y < self.size:
                    cell_type = self.grid[world_x, world_y]
                    
                    # Add observation noise based on cell type
                    noise_level = self.observation_noise
                    if cell_type == CellType.UNCERTAIN.value:
                        noise_level *= 5  # High uncertainty regions
                    
                    # Channel 0: Walls
                    if cell_type == CellType.WALL.value:
                        observation[obs_x, obs_y, 0] = 1.0
                    
                    # Channel 1: Goals
                    if cell_type == CellType.GOAL.value:
                        observation[obs_x, obs_y, 1] = 1.0
                    
                    # Channel 2: Hazards
                    if cell_type == CellType.HAZARD.value:
                        observation[obs_x, obs_y, 2] = 1.0
                    
                    # Add noise
                    observation[obs_x, obs_y, :3] += np.random.normal(0, noise_level, 3)
                    observation[obs_x, obs_y, :3] = np.clip(observation[obs_x, obs_y, :3], 0, 1)
                else:
                    # Outside world boundary - mark as wall
                    observation[obs_x, obs_y, 0] = 1.0
        
        # Channel 3: Agent position and orientation
        center = self.vision_range
        observation[center, center, 3] = 1.0  # Agent position
        
        # Add orientation information as a slight bias in the forward direction
        if self.agent_orientation == 0:  # North
            if center > 0:
                observation[center-1, center, 3] += 0.3
        elif self.agent_orientation == 1:  # East
            if center < obs_size - 1:
                observation[center, center+1, 3] += 0.3
        elif self.agent_orientation == 2:  # South
            if center < obs_size - 1:
                observation[center+1, center, 3] += 0.3
        elif self.agent_orientation == 3:  # West
            if center > 0:
                observation[center, center-1, 3] += 0.3
        
        # Flatten observation for agent
        return observation.flatten()
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        x, y = self.agent_pos
        cell_type = self.grid[x, y]
        
        # Base reward (small negative to encourage efficiency)
        reward = -0.01
        
        # Goal reward
        if cell_type == CellType.GOAL.value:
            reward += 10.0
        
        # Hazard penalty
        if cell_type == CellType.HAZARD.value:
            reward -= 5.0
        
        # Exploration bonus (reward for visiting new cells)
        if len(self.position_history) > 0:
            unique_positions = len(set(map(tuple, self.position_history)))
            exploration_bonus = 0.01 * unique_positions / len(self.position_history)
            reward += exploration_bonus
        
        # Distance-based reward (closer to goal is better)
        distance_to_goal = self._distance_to_nearest_goal()
        max_distance = np.sqrt(2 * self.size**2)
        proximity_reward = 0.1 * (1.0 - distance_to_goal / max_distance)
        reward += proximity_reward
        
        return reward
    
    def _is_goal_reached(self) -> bool:
        """Check if agent has reached a goal."""
        x, y = self.agent_pos
        return self.grid[x, y] == CellType.GOAL.value
    
    def _distance_to_nearest_goal(self) -> float:
        """Compute distance to nearest goal."""
        min_distance = float('inf')
        for goal_x, goal_y in self.goal_locations:
            distance = np.linalg.norm(self.agent_pos - np.array([goal_x, goal_y]))
            min_distance = min(min_distance, distance)
        return min_distance
    
    def _get_observation_uncertainty(self) -> float:
        """Get observation uncertainty at current position."""
        x, y = self.agent_pos
        if self.grid[x, y] == CellType.UNCERTAIN.value:
            return self.observation_noise * 5
        return self.observation_noise
    
    def _move_goals(self):
        """Move goals to new random locations (for dynamic environments)."""
        # Clear current goals
        for x, y in self.goal_locations:
            if 0 <= x < self.size and 0 <= y < self.size:
                self.grid[x, y] = CellType.EMPTY.value
        
        # Place goals in new locations
        new_goal_locations = []
        for _ in self.goal_locations:
            while True:
                x, y = np.random.randint(1, self.size-1, 2)
                if (self.grid[x, y] == CellType.EMPTY.value and 
                    not np.array_equal([x, y], self.agent_pos)):
                    self.grid[x, y] = CellType.GOAL.value
                    new_goal_locations.append((x, y))
                    break
        
        self.goal_locations = new_goal_locations
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human':
            self._render_ascii()
        elif mode == 'rgb_array':
            return self._render_rgb()
    
    def _render_ascii(self):
        """Render environment as ASCII art."""
        symbols = {
            CellType.EMPTY.value: '.',
            CellType.WALL.value: '#',
            CellType.GOAL.value: 'G',
            CellType.HAZARD.value: 'H',
            CellType.UNCERTAIN.value: '?'
        }
        
        orientation_symbols = ['^', '>', 'v', '<']
        
        print(f"\nStep {self.step_count}, Episode {self.episode_count}")
        print(f"Agent position: {self.agent_pos}, Orientation: {orientation_symbols[self.agent_orientation]}")
        print(f"Total reward: {self.total_reward:.2f}")
        print("Grid:")
        
        for x in range(self.size):
            row = ""
            for y in range(self.size):
                if np.array_equal([x, y], self.agent_pos):
                    row += orientation_symbols[self.agent_orientation]
                else:
                    row += symbols[self.grid[x, y]]
            print(row)
    
    def _render_rgb(self) -> np.ndarray:
        """Render environment as RGB image."""
        # Create RGB image
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Color mapping
        colors = {
            CellType.EMPTY.value: [255, 255, 255],      # White
            CellType.WALL.value: [0, 0, 0],             # Black
            CellType.GOAL.value: [0, 255, 0],           # Green
            CellType.HAZARD.value: [255, 0, 0],         # Red
            CellType.UNCERTAIN.value: [128, 128, 128]    # Gray
        }
        
        # Fill grid colors
        for x in range(self.size):
            for y in range(self.size):
                img[x, y] = colors[self.grid[x, y]]
        
        # Draw agent
        agent_x, agent_y = self.agent_pos
        img[agent_x, agent_y] = [0, 0, 255]  # Blue
        
        return img
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics for analysis."""
        if not self.position_history:
            return {}
        
        unique_positions = len(set(map(tuple, self.position_history)))
        total_positions = len(self.position_history)
        
        stats = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.step_count),
            'exploration_ratio': unique_positions / max(1, total_positions),
            'distance_to_goal': self._distance_to_nearest_goal(),
            'observation_uncertainty': self._get_observation_uncertainty(),
            'current_position': self.agent_pos.tolist(),
            'current_orientation': self.agent_orientation,
            'goal_reached': self._is_goal_reached(),
        }
        
        return stats
    
    def close(self):
        """Close the environment."""
        pass


class ForagingEnvironment(ActiveInferenceGridWorld):
    """Grid world with foraging dynamics - resources appear and disappear."""
    
    def __init__(self, resource_density: float = 0.1, **kwargs):
        self.resource_density = resource_density
        self.resources = set()
        super().__init__(**kwargs)
    
    def reset(self) -> np.ndarray:
        """Reset and spawn initial resources."""
        obs = super().reset()
        self._spawn_resources()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with resource dynamics."""
        # Regular step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Resource collection
        if tuple(self.agent_pos) in self.resources:
            reward += 1.0  # Bonus for collecting resource
            self.resources.remove(tuple(self.agent_pos))
        
        # Spawn/despawn resources
        if self.step_count % 5 == 0:  # Every 5 steps
            self._update_resources()
        
        info['resources_collected'] = len(set(map(tuple, self.position_history)).intersection(self.resources))
        info['active_resources'] = len(self.resources)
        
        return obs, reward, terminated, truncated, info
    
    def _spawn_resources(self):
        """Spawn initial resources."""
        self.resources.clear()
        n_resources = int(self.size * self.size * self.resource_density)
        
        for _ in range(n_resources):
            while True:
                x, y = np.random.randint(1, self.size-1, 2)
                if (self.grid[x, y] == CellType.EMPTY.value and 
                    not np.array_equal([x, y], self.agent_pos)):
                    self.resources.add((x, y))
                    break
    
    def _update_resources(self):
        """Update resource locations."""
        # Remove some resources
        if self.resources and np.random.random() < 0.3:
            self.resources.pop()
        
        # Add new resources
        if np.random.random() < 0.5:
            while True:
                x, y = np.random.randint(1, self.size-1, 2)
                if (self.grid[x, y] == CellType.EMPTY.value and 
                    not np.array_equal([x, y], self.agent_pos) and
                    (x, y) not in self.resources):
                    self.resources.add((x, y))
                    break
    
    def _render_ascii(self):
        """Render with resources marked as '*'."""
        super()._render_ascii()
        print("Resources:")
        for x, y in self.resources:
            print(f"  Resource at ({x}, {y})")