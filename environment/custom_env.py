import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from typing import Optional, Tuple, Dict, Any

class HeatwaveWarningEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Grid environment parameters - 10x10 representing regions of South Sudan
        self.grid_size = 10
        self.max_steps = 100
        self.current_step = 0
        
        # Agent position
        self.agent_pos = [0, 0]
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Issue Alert, 5=Do Nothing
        self.action_space = spaces.Discrete(6)
        
        # State: (x, y, temperature, humidity, vegetation, alert_status) as Box for SB3 compatibility
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 25, 10, 0, 0]),
            high=np.array([9, 9, 55, 100, 1, 1]),
            dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        
        # Initialize environment
        self.reset()
    
    def _initialize_grid(self):
        """Initialize 10x10 grid representing regions of South Sudan"""
        # Temperature grid (°C) - varies by region
        self.temperature_grid = np.random.uniform(30, 45, (self.grid_size, self.grid_size))
        # Add hotspots in certain regions (simulate arid areas)
        self.temperature_grid[2:4, 7:9] += 8  # Northern arid region
        self.temperature_grid[6:8, 1:3] += 6  # Western dry region
        
        # Humidity grid (%) - lower in arid regions
        self.humidity_grid = np.random.uniform(20, 80, (self.grid_size, self.grid_size))
        self.humidity_grid[2:4, 7:9] -= 30  # Arid regions have lower humidity
        self.humidity_grid[6:8, 1:3] -= 20
        self.humidity_grid = np.clip(self.humidity_grid, 10, 100)
        
        # Vegetation index (0-1) - lower in vulnerable areas
        self.vegetation_grid = np.random.uniform(0.2, 0.8, (self.grid_size, self.grid_size))
        self.vegetation_grid[2:4, 7:9] *= 0.3  # Sparse vegetation in arid areas
        self.vegetation_grid[6:8, 1:3] *= 0.5
        
        # Alert status grid (0=no alert, 1=alert issued)
        self.alert_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Heatwave threshold - cells above 40°C are heatwave zones
        self.heatwave_threshold = 40
        
    def _update_environmental_conditions(self):
        """Update temperature and humidity over time (temporal dynamics)"""
        # Simulate daily temperature variation and heatwave progression
        temp_change = np.random.normal(0, 1, (self.grid_size, self.grid_size))
        self.temperature_grid += temp_change * 0.5
        self.temperature_grid = np.clip(self.temperature_grid, 25, 55)
        
        # Update humidity with inverse correlation to temperature
        humidity_change = -temp_change * 0.8 + np.random.normal(0, 2, (self.grid_size, self.grid_size))
        self.humidity_grid += humidity_change
        self.humidity_grid = np.clip(self.humidity_grid, 10, 100)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset agent position
        self.agent_pos = [0, 0]
        self.current_step = 0
        
        # Initialize grid
        self._initialize_grid()
        
        # Reset tracking variables
        self.alerts_issued = 0
        self.correct_alerts = 0
        self.missed_alerts = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation as array"""
        x, y = self.agent_pos
        return np.array([
            x, y,
            self.temperature_grid[x, y],
            self.humidity_grid[x, y],
            self.vegetation_grid[x, y],
            self.alert_grid[x, y]
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = -1  # Per move step penalty
        x, y = self.agent_pos
        
        # Execute action
        if action == 0:  # Move Up
            if x > 0:
                self.agent_pos[0] -= 1
        elif action == 1:  # Move Down
            if x < self.grid_size - 1:
                self.agent_pos[0] += 1
        elif action == 2:  # Move Left
            if y > 0:
                self.agent_pos[1] -= 1
        elif action == 3:  # Move Right
            if y < self.grid_size - 1:
                self.agent_pos[1] += 1
        elif action == 4:  # Issue Alert
            if self.alert_grid[x, y] == 0:  # No previous alert
                self.alert_grid[x, y] = 1
                self.alerts_issued += 1
                
                # +10 → Correct early alert (before heatwave threshold)
                if self.temperature_grid[x, y] >= self.heatwave_threshold:
                    reward += 10  # Correct early alert
                    self.correct_alerts += 1
        elif action == 5:  # Do Nothing
            # -10 → Missed alert in heatwave zone
            if (self.temperature_grid[x, y] >= self.heatwave_threshold and 
                self.alert_grid[x, y] == 0):
                reward -= 10  # Missed alert in heatwave zone
                self.missed_alerts += 1
        
        # Update environmental conditions
        self._update_environmental_conditions()
        self.current_step += 1
        
        # Done condition: Fixed number of steps or all zones covered
        terminated = (self.current_step >= self.max_steps or 
                     self._all_heatwave_zones_covered())
        
        # Get new observation
        observation = self._get_observation()
        
        info = {
            "alerts_issued": self.alerts_issued,
            "correct_alerts": self.correct_alerts,
            "missed_alerts": self.missed_alerts,
            "heatwave_zones": np.sum(self.temperature_grid >= self.heatwave_threshold)
        }
        
        return observation, reward, terminated, False, info
    
    def _all_heatwave_zones_covered(self) -> bool:
        """Check if all heatwave zones have been alerted"""
        heatwave_zones = self.temperature_grid >= self.heatwave_threshold
        alerted_heatwave_zones = heatwave_zones & (self.alert_grid == 1)
        return np.sum(alerted_heatwave_zones) == np.sum(heatwave_zones)
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        pygame.init()
        pygame.font.init()
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("South Sudan Heatwave Early Warning System")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * self.cell_size
                y = i * self.cell_size
                
                # Color based on temperature (risk level)
                temp = self.temperature_grid[i, j]
                if temp > 45:
                    color = (139, 0, 0)  # Dark red - extreme risk
                elif temp > 42:
                    color = (255, 0, 0)  # Red - high risk
                elif temp > 38:
                    color = (255, 165, 0)  # Orange - medium risk
                else:
                    color = (144, 238, 144)  # Light green - low risk
                
                pygame.draw.rect(canvas, color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(canvas, (0, 0, 0), (x, y, self.cell_size, self.cell_size), 1)
                
                # 🔵 Zones already alerted (blue circles)
                if self.alert_grid[i, j] == 1:
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    pygame.draw.circle(canvas, (0, 0, 255), center, self.cell_size // 4)
                
                # 🔴 Heatwave areas (red border)
                if self.temperature_grid[i, j] >= self.heatwave_threshold:
                    pygame.draw.rect(canvas, (255, 0, 0), (x, y, self.cell_size, self.cell_size), 3)
        
        # 🟢 Agent position (green circle)
        agent_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        agent_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(canvas, (0, 255, 0), (agent_x, agent_y), self.cell_size // 3)
        
        # Draw info panel
        font = pygame.font.Font(None, 24)
        info_texts = [
            f"Step: {self.current_step}",
            f"Alerts Issued: {self.alerts_issued}",
            f"Correct Alerts: {self.correct_alerts}",
            f"Missed Alerts: {self.missed_alerts}",
            f"Current Temp: {self.temperature_grid[self.agent_pos[0], self.agent_pos[1]]:.1f}°C",
            f"Heatwave Zones: {np.sum(self.temperature_grid >= self.heatwave_threshold)}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            canvas.blit(text_surface, (10, 10 + i * 25))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# Register environment
from gymnasium.envs.registration import register

register(
    id='HeatwaveWarning-v0',
    entry_point='environment.custom_env:HeatwaveWarningEnv',
    max_episode_steps=100,
)