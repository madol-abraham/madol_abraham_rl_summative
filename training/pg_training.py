import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import HeatwaveWarningEnv

class PPOMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.alerts_sent = []
        self.heatwave_zones_missed = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            episode_length = self.locals.get('episode_length', 0)
            episode_reward = sum(self.locals.get('rewards', [0]))
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.alerts_sent.append(info.get('alerts_issued', 0))
            
            heatwave_zones = info.get('heatwave_zones', 0)
            correct_alerts = info.get('correct_alerts', 0)
            missed = max(0, heatwave_zones - correct_alerts)
            self.heatwave_zones_missed.append(missed)
            
        return True

def train_ppo(total_timesteps=50000):
    """Train PPO model using Stable Baselines3"""
    env = HeatwaveWarningEnv()
    
    # Create models directory
    os.makedirs("models/pg", exist_ok=True)
    
    # PPO model with TensorBoard logging
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/ppo/"
    )
    
    callback = PPOMetricsCallback()
    
    print("Training PPO for Heatwave Warning System...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save trained model
    model.save("models/pg/ppo_heatwave_model")
    print("PPO model saved to models/pg/ppo_heatwave_model.zip")
    
    # Plot training curves
    plot_training_curves(callback, "PPO", "models/pg/")
    
    return model, callback

class REINFORCEPolicy(nn.Module):
    """Simple policy network for REINFORCE"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(REINFORCEPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)

class REINFORCEAgent:
    """REINFORCE agent implementation"""
    def __init__(self, env, lr=0.001, gamma=0.99):
        self.env = env
        self.gamma = gamma
        
        # Convert dict observation to flat vector for neural network
        self.state_dim = 6  # x, y, temp, humidity, vegetation, alert_status
        self.action_dim = env.action_space.n
        
        self.policy = REINFORCEPolicy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.alerts_sent = []
        self.heatwave_zones_missed = []
        
    def select_action(self, state_array):
        state = torch.FloatTensor(state_array).unsqueeze(0)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def train_episode(self):
        state, _ = self.env.reset()
        log_probs = []
        rewards = []
        
        while True:
            action, log_prob = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if terminated or truncated:
                break
            state = next_state
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Track metrics
        episode_reward = sum(rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(rewards))
        self.alerts_sent.append(info.get('alerts_issued', 0))
        
        heatwave_zones = info.get('heatwave_zones', 0)
        correct_alerts = info.get('correct_alerts', 0)
        missed = max(0, heatwave_zones - correct_alerts)
        self.heatwave_zones_missed.append(missed)
        
        return episode_reward, info

def train_reinforce(num_episodes=1000):
    """Train REINFORCE agent"""
    env = HeatwaveWarningEnv()
    agent = REINFORCEAgent(env)
    
    # Create models directory
    os.makedirs("models/pg", exist_ok=True)
    
    print("Training REINFORCE for Heatwave Warning System...")
    
    for episode in range(num_episodes):
        reward, info = agent.train_episode()
        
        if episode % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}")
    
    # Save model
    torch.save(agent.policy.state_dict(), "models/pg/reinforce_policy.pth")
    print("REINFORCE model saved to models/pg/reinforce_policy.pth")
    
    # Plot training curves
    plot_reinforce_curves(agent, "models/pg/")
    
    return agent

def plot_training_curves(callback, algorithm_name, save_dir):
    """Plot training performance curves for PPO"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(callback.episode_rewards)
    axes[0,0].set_title(f'{algorithm_name} - Mean Reward')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True)
    
    axes[0,1].plot(callback.episode_lengths)
    axes[0,1].set_title(f'{algorithm_name} - Episode Length')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Steps')
    axes[0,1].grid(True)
    
    axes[1,0].plot(callback.alerts_sent, label='Alerts Sent')
    axes[1,0].plot(callback.heatwave_zones_missed, label='Zones Missed')
    axes[1,0].set_title(f'{algorithm_name} - Alerts vs Missed')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    efficiency = [s/(s+m) if (s+m)>0 else 0 for s,m in zip(callback.alerts_sent, callback.heatwave_zones_missed)]
    axes[1,1].plot(efficiency)
    axes[1,1].set_title(f'{algorithm_name} - Alert Efficiency')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Efficiency')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}ppo_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PPO training curves saved to {save_dir}ppo_training_curves.png")

def plot_reinforce_curves(agent, save_dir):
    """Plot training curves for REINFORCE"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(agent.episode_rewards)
    axes[0,0].set_title('REINFORCE - Mean Reward')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True)
    
    axes[0,1].plot(agent.episode_lengths)
    axes[0,1].set_title('REINFORCE - Episode Length')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Steps')
    axes[0,1].grid(True)
    
    axes[1,0].plot(agent.alerts_sent, label='Alerts Sent')
    axes[1,0].plot(agent.heatwave_zones_missed, label='Zones Missed')
    axes[1,0].set_title('REINFORCE - Alerts vs Missed')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    efficiency = [s/(s+m) if (s+m)>0 else 0 for s,m in zip(agent.alerts_sent, agent.heatwave_zones_missed)]
    axes[1,1].plot(efficiency)
    axes[1,1].set_title('REINFORCE - Alert Efficiency')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Efficiency')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}reinforce_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"REINFORCE training curves saved to {save_dir}reinforce_training_curves.png")

if __name__ == "__main__":
    # Train PPO
    print("Training PPO...")
    ppo_model, ppo_callback = train_ppo(total_timesteps=30000)
    
    # Train REINFORCE
    print("\nTraining REINFORCE...")
    reinforce_agent = train_reinforce(num_episodes=500)