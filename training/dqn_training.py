import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import HeatwaveWarningEnv

class DQNMetricsCallback(BaseCallback):
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
            
            # Calculate episode reward from info or track manually
            episode_reward = sum(self.locals.get('rewards', [0]))
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.alerts_sent.append(info.get('alerts_issued', 0))
            
            # Calculate missed heatwave zones
            heatwave_zones = info.get('heatwave_zones', 0)
            correct_alerts = info.get('correct_alerts', 0)
            missed = max(0, heatwave_zones - correct_alerts)
            self.heatwave_zones_missed.append(missed)
            
        return True

def train_dqn(total_timesteps=50000):
    """Train DQN model and save under models/dqn/"""
    env = HeatwaveWarningEnv()
    
    # Create models directory
    os.makedirs("models/dqn", exist_ok=True)
    
    # DQN model with TensorBoard logging
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/dqn/"
    )
    
    # Callback for metrics tracking
    callback = DQNMetricsCallback()
    
    print("Training DQN for Heatwave Warning System...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save trained model
    model.save("models/dqn/dqn_heatwave_model")
    print("DQN model saved to models/dqn/dqn_heatwave_model.zip")
    
    # Plot and save training performance curves
    plot_training_curves(callback, "DQN", "models/dqn/")
    
    return model, callback

def plot_training_curves(callback, algorithm_name, save_dir):
    """Plot training performance curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mean reward
    axes[0,0].plot(callback.episode_rewards)
    axes[0,0].set_title(f'{algorithm_name} - Mean Reward')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True)
    
    # Episode length
    axes[0,1].plot(callback.episode_lengths)
    axes[0,1].set_title(f'{algorithm_name} - Episode Length')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Steps')
    axes[0,1].grid(True)
    
    # Alerts sent vs heatwave zones missed
    axes[1,0].plot(callback.alerts_sent, label='Alerts Sent')
    axes[1,0].plot(callback.heatwave_zones_missed, label='Heatwave Zones Missed')
    axes[1,0].set_title(f'{algorithm_name} - Alerts vs Missed Zones')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Alert efficiency (alerts sent / (alerts sent + missed zones))
    efficiency = []
    for sent, missed in zip(callback.alerts_sent, callback.heatwave_zones_missed):
        total = sent + missed
        eff = sent / total if total > 0 else 0
        efficiency.append(eff)
    
    axes[1,1].plot(efficiency)
    axes[1,1].set_title(f'{algorithm_name} - Alert Efficiency')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Efficiency Ratio')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}dqn_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}dqn_training_curves.png")

def evaluate_dqn(model_path="models/dqn/dqn_heatwave_model.zip", num_episodes=10):
    """Evaluate trained DQN model"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    env = HeatwaveWarningEnv()
    model = DQN.load(model_path)
    
    total_rewards = []
    episode_lengths = []
    alerts_sent = []
    zones_missed = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        alerts_sent.append(info.get('alerts_issued', 0))
        
        heatwave_zones = info.get('heatwave_zones', 0)
        correct_alerts = info.get('correct_alerts', 0)
        missed = max(0, heatwave_zones - correct_alerts)
        zones_missed.append(missed)
    
    results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_alerts_sent': np.mean(alerts_sent),
        'mean_zones_missed': np.mean(zones_missed),
        'alert_efficiency': np.mean(alerts_sent) / (np.mean(alerts_sent) + np.mean(zones_missed)) if (np.mean(alerts_sent) + np.mean(zones_missed)) > 0 else 0
    }
    
    print(f"\nDQN Evaluation Results ({num_episodes} episodes):")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f}")
    print(f"Mean Alerts Sent: {results['mean_alerts_sent']:.1f}")
    print(f"Mean Zones Missed: {results['mean_zones_missed']:.1f}")
    print(f"Alert Efficiency: {results['alert_efficiency']:.2%}")
    
    return results

if __name__ == "__main__":
    # Train DQN
    model, callback = train_dqn(total_timesteps=30000)
    
    # Evaluate trained model
    evaluate_dqn()