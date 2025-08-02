import pygame
import numpy as np
import imageio
import os
from environment.custom_env import HeatwaveWarningEnv

def create_random_agent_gif():
    """Create GIF of random agent in environment and store in /media/random_agent.gif"""
    # Create media directory if it doesn't exist
    os.makedirs("media", exist_ok=True)
    
    # Create environment with rgb_array rendering for GIF capture
    env = HeatwaveWarningEnv(render_mode="rgb_array")
    frames = []
    
    # Run random agent for a few steps
    obs, _ = env.reset()
    frames.append(env.render())
    
    for step in range(30):  # Capture 30 steps
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if terminated or truncated:
            obs, _ = env.reset()
            frames.append(env.render())
    
    # Save GIF using imageio
    if frames:
        imageio.mimsave("media/random_agent.gif", frames, duration=0.5)
        print("Random agent GIF saved to media/random_agent.gif")
    
    env.close()

def visualize_pygame():
    """Use pygame to visualize the environment with human rendering"""
    pygame.init()
    pygame.display.init()
    
    env = HeatwaveWarningEnv(render_mode="human")
    
    print("Pygame visualization started. Close window to exit.")
    print("Legend:")
    print("🟢 Green circle: Agent position")
    print("🔵 Blue circles: Zones already alerted")
    print("🔴 Red borders: Heatwave areas")
    print("Color coding: Green=cool, Orange=warm, Red=hot, Dark Red=extreme")
    print("\n" + "="*50)
    print("REAL-TIME SIMULATION OUTPUT:")
    print("="*50)
    
    running = True
    obs, _ = env.reset()
    step_count = 0
    episode = 1
    max_episodes = 3  # Limit to 3 episodes
    
    action_names = ["Move Up", "Move Down", "Move Left", "Move Right", "Issue Alert", "Do Nothing"]
    
    print(f"\nEPISODE {episode}/3 STARTED")
    print(f"Initial Position: ({obs[0]:.0f}, {obs[1]:.0f}) | Temp: {obs[2]:.1f}°C | Humidity: {obs[3]:.1f}%")
    
    try:
        while running and episode <= max_episodes:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print real-time info
            print(f"Step {step_count:3d}: {action_names[action]:<12} | Pos: ({obs[0]:.0f},{obs[1]:.0f}) | Temp: {obs[2]:.1f}°C | Reward: {reward:+4.1f} | Alerts: {info['alerts_issued']}")
            
            # Render environment
            env.render()
            
            if terminated or truncated:
                print(f"\nEPISODE {episode}/3 COMPLETED:")
                print(f"  Total Steps: {step_count}")
                print(f"  Alerts Issued: {info['alerts_issued']}")
                print(f"  Correct Alerts: {info['correct_alerts']}")
                print(f"  Missed Alerts: {info['missed_alerts']}")
                print(f"  Heatwave Zones: {info['heatwave_zones']}")
                print("-" * 50)
                
                episode += 1
                if episode <= max_episodes:
                    obs, _ = env.reset()
                    step_count = 0
                    print(f"\nEPISODE {episode}/3 STARTED")
                    print(f"Initial Position: ({obs[0]:.0f}, {obs[1]:.0f}) | Temp: {obs[2]:.1f}°C | Humidity: {obs[3]:.1f}%")
                else:
                    print("\n🎯 All 3 episodes completed! Closing visualization...")
                    running = False
            
            # Faster delay
            pygame.time.wait(500)  # Reduced from 1500ms to 500ms
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user")
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    print("Creating random agent GIF...")
    create_random_agent_gif()
    
    print("\nStarting pygame visualization...")
    visualize_pygame()