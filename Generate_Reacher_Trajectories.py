import gymnasium as gym
import numpy as np

def generate_trajectories(n, env_name='Reacher-v5', gamma=0.99, max_steps=200):
    env = gym.make(env_name)
    trajectories = []
    
    for _ in range(n):
        obs, _ = env.reset()
        trajectory = []
        rewards = []
        timesteps = []
        
        for t in range(max_steps):
            action = env.action_space.sample()  # Random action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((obs, action, t))
            rewards.append(reward)
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Compute returns (discounted sum of rewards)
        returns = np.zeros(len(rewards))
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G
        
        # Append trajectory with returns and timesteps
        trajectory = [(r, s, a, t) for r, (s, a, t) in zip(returns, trajectory)]
        trajectories.append(trajectory)
    
    env.close()
    return trajectories

def max_return(trajectories):
    return max(max(r for r, _, _, _ in traj) for traj in trajectories)

