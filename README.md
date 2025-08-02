# South Sudan Heatwave Early Warning System

## Overview of Project and Problem

This project simulates a reinforcement learning agent that learns to issue heatwave alerts across a grid-based representation of South Sudan. The agent must minimize heatwave impact by acting early based on climate variables including temperature, humidity, and vegetation indices.

The system addresses the critical need for early warning systems in South Sudan, where extreme heat events pose significant risks to vulnerable populations. By using reinforcement learning, the agent learns optimal policies for when and where to issue alerts to maximize protection while minimizing false alarms.

## Environment Design

### HeatwaveWarningEnv

**Grid Structure**: 10x10 grid representing regions of South Sudan
- Each cell represents a geographic area with distinct climate characteristics
- Agent navigates the grid and makes alert decisions for each location

**State Space**: Dictionary containing:
- `x`: Agent x-coordinate (0-9)
- `y`: Agent y-coordinate (0-9)  
- `temperature`: Current temperature in °C (25-55)
- `humidity`: Humidity percentage (10-100)
- `vegetation`: Vegetation index (0-1, lower values indicate vulnerability)
- `alert_status`: Whether cell has been alerted (0=no, 1=yes)

**Environmental Dynamics**:
- Temperature and humidity change over time to simulate weather progression
- Certain regions (northern and western areas) have higher baseline temperatures
- Heatwave zones are defined as areas with temperature ≥ 40°C

## Action/Reward Definition

### Action Space (Discrete 6):
- **0**: Move Up - Navigate north
- **1**: Move Down - Navigate south  
- **2**: Move Left - Navigate west
- **3**: Move Right - Navigate east
- **4**: Issue Alert - Alert current community
- **5**: Do Nothing - Stay in current position

### Reward Structure:
- **+10**: Correct early alert (issued before heatwave threshold in heatwave zone)
- **-10**: Missed alert in heatwave zone (doing nothing when alert needed)
- **-1**: Per move step penalty (encourages efficiency)

### Termination Conditions:
- Fixed number of steps (100) reached
- All heatwave zones have been covered with alerts

## How to Run Training and Render Simulations

### Installation
```bash
pip install -r requirements.txt
```

### Usage Examples

**Run Random Agent and Create GIF:**
```bash
python main.py random
```
This creates `media/random_agent.gif` showing baseline random behavior.

**Train Models:**
```bash
python main.py train --algorithm dqn      # Train DQN
python main.py train --algorithm ppo      # Train PPO  
python main.py train --algorithm a2c      # Train A2C
python main.py train --algorithm reinforce # Train REINFORCE
```

**Evaluate Trained Models:**
```bash
python main.py evaluate --algorithm dqn
python main.py evaluate --algorithm ppo
python main.py evaluate --algorithm a2c
python main.py evaluate --algorithm reinforce
```

**Visualize Environment:**
```bash
python main.py visualize
```
Opens pygame window showing real-time environment visualization.

### Training Details

**Models are saved to:**
- DQN models: `models/dqn/`
- Policy Gradient models: `models/pg/` (PPO, A2C, REINFORCE)

**TensorBoard Logging:**
- DQN logs: `logs/dqn/`
- PPO logs: `logs/ppo/`  
- A2C logs: `logs/a2c/`

**View training progress:**
```bash
tensorboard --logdir logs/
```

## Performance Results Summary

### Evaluation Metrics

Each model is evaluated on:
- **Mean Reward**: Average episode reward over 10 test episodes
- **Episode Length**: Average number of steps per episode
- **Alerts Sent**: Average number of alerts issued per episode
- **Heatwave Zones Missed**: Average number of unalerted heatwave zones
- **Alert Efficiency**: Ratio of alerts sent to total zones needing alerts

### Algorithm Characteristics

**DQN (Deep Q-Network)**:
- Value-based learning with experience replay
- Good exploration-exploitation balance
- Stable convergence with target networks

**PPO (Proximal Policy Optimization)**:
- Policy gradient method with clipped objectives
- Most stable training among policy methods
- Good sample efficiency

**A2C (Actor-Critic)**:
- Combines value and policy learning
- Lower variance than pure policy gradient
- Faster convergence than REINFORCE

**REINFORCE**:
- Pure policy gradient method
- Simple but high variance
- Direct policy optimization

### Visualization Features

**Environment Rendering**:
- 🟢 Green circle: Agent position
- 🔵 Blue circles: Zones already alerted  
- 🔴 Red borders: Heatwave areas (temp ≥ 40°C)
- Color coding: Green=cool, Orange=warm, Red=hot, Dark Red=extreme

**Training Curves**: Each algorithm generates plots showing:
- Mean reward progression
- Episode length trends
- Alert efficiency over time
- Comparison of alerts sent vs zones missed

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py          # HeatwaveWarningEnv implementation
│   └── rendering.py           # Pygame visualization and GIF creation
├── training/
│   ├── dqn_training.py        # DQN with Stable Baselines3
│   ├── pg_training.py         # PPO and REINFORCE training
│   └── actor_critic_training.py # A2C implementation
├── models/
│   ├── dqn/                   # Saved DQN models and metrics
│   └── pg/                    # Saved policy gradient models
├── media/
│   └── random_agent.gif       # Random agent visualization
├── main.py                    # CLI interface
├── requirements.txt           # Dependencies
└── README.md                  # This documentation
```

## Technical Implementation

- **Framework**: Gymnasium for environment, Stable Baselines3 for algorithms
- **Visualization**: Pygame for real-time rendering, imageio for GIF creation
- **Logging**: TensorBoard integration for training metrics
- **Model Persistence**: Automatic saving of trained models and performance plots

This system demonstrates the application of modern reinforcement learning techniques to critical humanitarian challenges, providing a foundation for real-world early warning system deployment.