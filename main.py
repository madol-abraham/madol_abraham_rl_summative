#!/usr/bin/env python3
"""
Main CLI script for South Sudan Heatwave Early Warning System
Allows user to run random agent, train models, and evaluate performance
"""

import os
import sys
import argparse
from environment.custom_env import HeatwaveWarningEnv
from environment.rendering import create_random_agent_gif, visualize_pygame

def run_random_agent():
    """Run a random agent in the environment and render for GIF"""
    print("Running random agent and creating GIF...")
    create_random_agent_gif()
    print("Random agent GIF created at media/random_agent.gif")

def train_model(algorithm):
    """Train a selected RL model"""
    print(f"Training {algorithm} model...")
    
    if algorithm.lower() == 'dqn':
        from training.dqn_training import train_dqn
        model, callback = train_dqn(total_timesteps=30000)
        print("DQN training completed!")
        
    elif algorithm.lower() == 'ppo':
        from training.pg_training import train_ppo
        model, callback = train_ppo(total_timesteps=30000)
        print("PPO training completed!")
        
    elif algorithm.lower() == 'reinforce':
        from training.pg_training import train_reinforce
        agent = train_reinforce(num_episodes=500)
        print("REINFORCE training completed!")
        
    elif algorithm.lower() == 'a2c':
        from training.actor_critic_training import train_actor_critic
        model, callback = train_actor_critic(total_timesteps=30000)
        print("A2C training completed!")
        
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: dqn, ppo, reinforce, a2c")

def evaluate_model(algorithm):
    """Evaluate a trained model's performance"""
    print(f"Evaluating {algorithm} model...")
    
    if algorithm.lower() == 'dqn':
        from training.dqn_training import evaluate_dqn
        results = evaluate_dqn()
        if results:
            print("DQN evaluation completed!")
        
    elif algorithm.lower() == 'ppo':
        from stable_baselines3 import PPO
        model_path = "models/pg/ppo_heatwave_model.zip"
        if os.path.exists(model_path):
            evaluate_sb3_model(PPO, model_path, "PPO")
        else:
            print(f"PPO model not found at {model_path}")
            
    elif algorithm.lower() == 'a2c':
        from training.actor_critic_training import evaluate_a2c
        results = evaluate_a2c()
        if results:
            print("A2C evaluation completed!")
            
    elif algorithm.lower() == 'reinforce':
        evaluate_reinforce_model()
        
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: dqn, ppo, reinforce, a2c")

def evaluate_sb3_model(model_class, model_path, algorithm_name):
    """Evaluate Stable Baselines3 model"""
    import numpy as np
    
    env = HeatwaveWarningEnv()
    model = model_class.load(model_path)
    
    total_rewards = []
    episode_lengths = []
    alerts_sent = []
    zones_missed = []
    
    for episode in range(10):
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
    
    print(f"\n{algorithm_name} Evaluation Results (10 episodes):")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Mean Alerts Sent: {np.mean(alerts_sent):.1f}")
    print(f"Mean Zones Missed: {np.mean(zones_missed):.1f}")
    efficiency = np.mean(alerts_sent) / (np.mean(alerts_sent) + np.mean(zones_missed)) if (np.mean(alerts_sent) + np.mean(zones_missed)) > 0 else 0
    print(f"Alert Efficiency: {efficiency:.2%}")

def evaluate_reinforce_model():
    """Evaluate REINFORCE model"""
    import torch
    import numpy as np
    from training.pg_training import REINFORCEAgent
    
    model_path = "models/pg/reinforce_policy.pth"
    if not os.path.exists(model_path):
        print(f"REINFORCE model not found at {model_path}")
        return
    
    env = HeatwaveWarningEnv()
    agent = REINFORCEAgent(env)
    agent.policy.load_state_dict(torch.load(model_path))
    
    total_rewards = []
    episode_lengths = []
    alerts_sent = []
    zones_missed = []
    
    for episode in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action, _ = agent.select_action(obs)
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
    
    print(f"\nREINFORCE Evaluation Results (10 episodes):")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Mean Alerts Sent: {np.mean(alerts_sent):.1f}")
    print(f"Mean Zones Missed: {np.mean(zones_missed):.1f}")
    efficiency = np.mean(alerts_sent) / (np.mean(alerts_sent) + np.mean(zones_missed)) if (np.mean(alerts_sent) + np.mean(zones_missed)) > 0 else 0
    print(f"Alert Efficiency: {efficiency:.2%}")

def show_visualization():
    """Show pygame visualization"""
    print("Starting pygame visualization...")
    visualize_pygame()

def main():
    parser = argparse.ArgumentParser(description='South Sudan Heatwave Early Warning System')
    parser.add_argument('mode', choices=['random', 'train', 'evaluate', 'visualize'], 
                       help='Mode to run: random agent, train model, evaluate model, or visualize')
    parser.add_argument('--algorithm', '-a', choices=['dqn', 'ppo', 'reinforce', 'a2c'],
                       help='Algorithm to train or evaluate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOUTH SUDAN HEATWAVE EARLY WARNING SYSTEM")
    print("=" * 60)
    
    if args.mode == 'random':
        run_random_agent()
        
    elif args.mode == 'train':
        if not args.algorithm:
            print("Please specify an algorithm to train with --algorithm")
            print("Available: dqn, ppo, reinforce, a2c")
            return
        train_model(args.algorithm)
        
    elif args.mode == 'evaluate':
        if not args.algorithm:
            print("Please specify an algorithm to evaluate with --algorithm")
            print("Available: dqn, ppo, reinforce, a2c")
            return
        evaluate_model(args.algorithm)
        
    elif args.mode == 'visualize':
        show_visualization()

if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("South Sudan Heatwave Early Warning System")
        print("\nUsage examples:")
        print("  python main.py random                    # Run random agent and create GIF")
        print("  python main.py train --algorithm dqn     # Train DQN model")
        print("  python main.py train --algorithm ppo     # Train PPO model")
        print("  python main.py train --algorithm a2c     # Train A2C model")
        print("  python main.py train --algorithm reinforce # Train REINFORCE model")
        print("  python main.py evaluate --algorithm dqn  # Evaluate DQN model")
        print("  python main.py visualize                 # Show pygame visualization")
        print("\nAvailable algorithms: dqn, ppo, reinforce, a2c")
    else:
        main()