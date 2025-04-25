

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998
TAU = 0.01  # Soft update coefficient
MEMORY_SIZE = 100_000
NUM_EPISODES = 2000
FEATURE_DIM = 64
GRAD_CLIP = 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# LFA-DQN with a ReLU layer
class LFA_DQN(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=FEATURE_DIM):
        super(LFA_DQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(feature_dim, action_dim, bias=False)

    def forward(self, x):
        features = self.feature_layer(x)
        return self.q_head(features)

# Epsilon-greedy policy
def select_action(q_network, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_network(state_tensor)
        return q_values.argmax().item()

# Soft update target network
def soft_update(target, source, tau=TAU):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Training setup
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = LFA_DQN(state_dim, action_dim).to(device)
target_network = LFA_DQN(state_dim, action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=LR)
replay_buffer = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START
returns = []
losses = []

best_return = -float("inf")
best_model_state = None

# Training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(q_network, state, epsilon, action_dim)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Clip reward to stabilize training
        clipped_reward = np.clip(reward, -1.0, 1.0)
        replay_buffer.push(state, action, clipped_reward, next_state, float(done))

        state = next_state
        total_reward += reward

        if total_reward > best_return:
            best_return = total_reward
            best_model_state = q_network.state_dict()

        # Learning step
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_network(next_states).max(1)[0]
                targets = rewards + (1 - dones) * GAMMA * next_q_values

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), GRAD_CLIP)
            optimizer.step()
            soft_update(target_network, q_network)

            losses.append(loss.item())

    # Decay epsilon
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    returns.append(total_reward)

    print(f"Episode {episode}, Return: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# Load best model
q_network.load_state_dict(best_model_state)

# Create env with visual render
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_network(state_tensor)
        action = q_values.argmax().item()
    
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Visualized Episode Return: {total_reward:.2f}")
env.close()

# Create env with video recording
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
env = RecordVideo(
    gym.make("CartPole-v1", render_mode="rgb_array"),
    video_folder="./videos/",
    episode_trigger=lambda episode_id: True  # Record every episode
)

# Load your trained Q-network
# Assume `q_network` is already trained and loaded
q_network.eval()

state, _ = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = q_network(state_tensor).argmax().item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

env.close()
print("Video saved to ./videos/")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(returns)
plt.title("Episode Return")
plt.xlabel("Episode")
plt.ylabel("Return")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Loss Over Time")
plt.xlabel("Training Step")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()
