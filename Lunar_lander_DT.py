# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random
# from decision_transformer import DecisionTransformer
# from trainer_discrete import get_second_last_layer_output  # Assuming it's in the same folder or properly imported


# class LFA_DQN(nn.Module):
#     def __init__(self, input_dim, num_actions):
#         super(LFA_DQN, self).__init__()
#         self.linear = nn.Linear(input_dim, num_actions)

#     def forward(self, x):
#         return self.linear(x)


# def train_lfa_dqn(env_name='LunarLander-v3', episodes=500, gamma=0.99, lr=1e-3, epsilon_start=1.0,
#                   epsilon_end=0.05, epsilon_decay=0.995):

#     env = gym.make(env_name)
#     num_actions = env.action_space.n

#     # Load pretrained decision transformer
#     state_dim = env.observation_space.shape[0]
#     dt_model = DecisionTransformer(
#         state_dim=state_dim,
#         act_dim=num_actions,
#         hidden_size=256,
#         max_length=1000,
#         action_tanh=False
#     )
#     dt_model.embed_action = torch.nn.Embedding(num_actions, dt_model.hidden_size)
#     dt_model.load_state_dict(torch.load("decision_transformer_mountaincar.pth",weights_only=True))
#     dt_model.eval()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dt_model.to(device)

#     feature_dim = 256  # Same as hidden size of the DT
#     q_network = LFA_DQN(feature_dim, num_actions).to(device)
#     optimizer = optim.Adam(q_network.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     epsilon = epsilon_start
#     replay_buffer = deque(maxlen=10000)

#     for episode in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         done = False
#         timestep = 0
#         return_to_go = 200.0  # Arbitrary return target for embedding

#         while not done:
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#             timestep_tensor = torch.tensor([timestep], dtype=torch.long).to(device)
#             #timestep_tensor=timestep_tensor
#             feature = get_second_last_layer_output(dt_model, state_tensor, return_to_go, timestep_tensor)

#             if random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 with torch.no_grad():
#                     q_values = q_network(feature)
#                     action = q_values.argmax().item()

#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             replay_buffer.append((state, action, reward, next_state, done, timestep))
#             state = next_state
#             timestep += 1
#             total_reward += reward

#             if len(replay_buffer) >= 64:
#                 batch = random.sample(replay_buffer, 64)
#                 batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_timesteps = zip(*batch)

#                 batch_features = torch.cat([
#                     get_second_last_layer_output(
#                         dt_model,
#                         torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device),
#                         return_to_go,
#                         torch.tensor([t], dtype=torch.long).to(device)

#                     ) for s, t in zip(batch_states, batch_timesteps)
#                 ])

#                 batch_next_features = torch.cat([
#                     get_second_last_layer_output(
#                         dt_model,
#                         torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(device),
#                         return_to_go,
#                         torch.tensor(t + 1, dtype=torch.long).to(device)
#                     ) for ns, t in zip(batch_next_states, batch_timesteps)
#                 ])

#                 batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(device)
#                 batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(device)
#                 batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(device)

#                 q_values = q_network(batch_features).gather(1, batch_actions_tensor)
#                 with torch.no_grad():
#                     max_next_q_values = q_network(batch_next_features).max(1)[0].unsqueeze(1)
#                     target_q_values = batch_rewards_tensor + (1 - batch_dones_tensor) * gamma * max_next_q_values

#                 loss = criterion(q_values, target_q_values)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         epsilon = max(epsilon_end, epsilon_decay * epsilon)
#         print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

#     env.close()


# if __name__ == "__main__":
#     train_lfa_dqn()
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from decision_transformer import DecisionTransformer
from trainer_discrete import get_second_last_layer_output  # Assuming it's in the same folder or properly imported


class LFA_DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(LFA_DQN, self).__init__()
        self.linear = nn.Linear(input_dim, num_actions)

    def forward(self, x):
        return self.linear(x)


def train_lfa_dqn(env_name='LunarLander-v3', episodes=500, gamma=0.99, lr=1e-3, epsilon_start=1.0,
                  epsilon_end=0.05, epsilon_decay=0.995):

    env = gym.make(env_name)
    num_actions = env.action_space.n

    # Load pretrained decision transformer
    state_dim = env.observation_space.shape[0]
    dt_model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=num_actions,
        hidden_size=256,
        max_length=1000,
        action_tanh=False
    )
    dt_model.embed_action = torch.nn.Embedding(num_actions, dt_model.hidden_size)
    dt_model.load_state_dict(torch.load("decision_transformer_mountaincar.pth",weights_only=True))
    dt_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_model.to(device)

    feature_dim = 256  # Same as hidden size of the DT
    q_network = LFA_DQN(feature_dim, num_actions).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epsilon = epsilon_start
    replay_buffer = deque(maxlen=10000)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        timestep = 0
        return_to_go = 6000.0  # Arbitrary return target for embedding

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            timestep_tensor = torch.tensor([timestep], dtype=torch.long).to(device)
            timestep_tensor=timestep_tensor.unsqueeze(0)
            feature = get_second_last_layer_output(dt_model, state_tensor, return_to_go, timestep_tensor)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(feature)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.append((state, action, reward, next_state, done, timestep))
            state = next_state
            timestep += 1
            total_reward += reward

            if len(replay_buffer) >= 64:
                batch = random.sample(replay_buffer, 64)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_timesteps = zip(*batch)

                batch_features = torch.cat([
                    get_second_last_layer_output(
                        dt_model,
                        torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device),
                        return_to_go,
                        torch.tensor([t], dtype=torch.long).unsqueeze(0).to(device)
                    ) for s, t in zip(batch_states, batch_timesteps)
                ])

                batch_next_features = torch.cat([
                    get_second_last_layer_output(
                        dt_model,
                        torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(device),
                        return_to_go,
                        torch.tensor([t + 1], dtype=torch.long).unsqueeze(0).to(device)
                    ) for ns, t in zip(batch_next_states, batch_timesteps)
                ])

                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = q_network(batch_features).gather(1, batch_actions_tensor)
                with torch.no_grad():
                    max_next_q_values = q_network(batch_next_features).max(1)[0].unsqueeze(1)
                    target_q_values = batch_rewards_tensor + (1 - batch_dones_tensor) * gamma * max_next_q_values

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()


if __name__ == "__main__":
    train_lfa_dqn(env_name='CartPole-v1')
