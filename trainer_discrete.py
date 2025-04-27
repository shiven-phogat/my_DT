import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Generate_Reacher_Trajectories import generate_trajectories
from decision_transformer import DecisionTransformer
import pickle


# Prepare dataset for discrete actions
def prepare_dataset(trajectories):
    returns, states, actions, timesteps = [], [], [], []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            r, s, a, t = traj[i]
            returns.append([r])
            states.append(s)
            actions.append([int(a)])  # Store as index
            timesteps.append([t])
    return (
        torch.tensor(returns, dtype=torch.float32),
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(timesteps, dtype=torch.long)
    )

# def prepare_dataset(trajectories):
#     returns, states, actions, timesteps = [], [], [], []
#     for traj in trajectories:
#         rewards = [step[0] for step in traj]
#         total_steps = len(traj)
#         for i in range(total_steps):
#             rtg = sum(rewards[i:])  # Return-to-go
#             _, s, a, t = traj[i]
#             returns.append([rtg])
#             states.append(s)
#             actions.append([int(a)])
#             timesteps.append([t])
#     return (
#         torch.tensor(returns, dtype=torch.float32),
#         torch.tensor(states, dtype=torch.float32),
#         torch.tensor(actions, dtype=torch.long),
#         torch.tensor(timesteps, dtype=torch.long)
#     )

# def prepare_dataset(trajectories):
#     returns, states, actions, timesteps = [], [], [], []
#     for traj in trajectories:
#         rewards = [step[0] for step in traj]
#         total_return = sum(rewards)
#         print(total_return)
#         cumulative = 0.0
#         for i, (r, s, a, t) in enumerate(traj):
#             rtg = total_return - cumulative
#             print(len(traj))
#             returns.append([rtg])
#             states.append(s)
#             actions.append([int(a)])
#             timesteps.append([t])
#             cumulative += r  # Accumulate reward to subtract in next step

#     return (
#         torch.tensor(returns, dtype=torch.float32),
#         torch.tensor(states, dtype=torch.float32),
#         torch.tensor(actions, dtype=torch.long),
#         torch.tensor(timesteps, dtype=torch.long)
#     )


# Training function
def train_model(model, dataloader, epochs=50, lr=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for returns, states, actions, timesteps in dataloader:
            optimizer.zero_grad()

            if states.dim() == 2:
                states = states.unsqueeze(1)
                actions = actions.unsqueeze(1)
                returns = returns.unsqueeze(1)
                timesteps = timesteps.unsqueeze(1)

            _, action_logits, _ = model(states, actions, None, returns, timesteps)

            # Reshape for CrossEntropyLoss: (batch * seq_len, num_actions) vs (batch * seq_len)
            action_logits = action_logits.view(-1, action_logits.shape[-1])
            actions = actions.view(-1)

            # print("Actions unique values:", torch.unique(actions))
            # print("Action logits shape:", action_logits.shape)
            # print("Action logits sample:", action_logits[0].detach().cpu().numpy())



            loss = criterion(action_logits, actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# Extract second-last transformer layer output


# def get_second_last_layer_output(model, state, max_return, timestep):
#     state = state.view(1, 1, -1)  # [1, 1, state_dim]
#     dummy_action = torch.tensor([[0]], dtype=torch.long)  # [1, 1]
#     max_return = torch.tensor([[max_return]], dtype=torch.float32)  # [1, 1]
#     timestep = timestep.view(1, 1)  # [1, 1]

#     print(state.device, max_return.device, timestep.device, model.embed_return.weight.device)


#     # Get embeddings: [1, 1, hidden_size]
#     state_emb = model.embed_state(state)
#     action_emb = model.embed_action(dummy_action)
#     return_emb = model.embed_return(max_return.unsqueeze(-1))  # [1, 1, 1] → [1, 1, 256]
#     timestep_emb = model.embed_timestep(timestep)  # ✅ just pass as [1, 1]

#     # Confirm all shapes match
#     assert state_emb.shape == action_emb.shape == return_emb.shape == timestep_emb.shape, \
#         f"Embedding shape mismatch: {state_emb.shape}, {action_emb.shape}, {return_emb.shape}, {timestep_emb.shape}"

#     # Create token sequence
#     token_sequence = torch.cat([state_emb, action_emb, return_emb, timestep_emb], dim=1)  # [1, 4, 256]
#     inputs_embeds = model.embed_ln(token_sequence)

#     with torch.no_grad():
#         transformer_outputs = model.transformer(
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=True
#         )

#     if transformer_outputs.hidden_states is None:
#         raise ValueError("Hidden states not returned. Check GPT2Config.")

#     return transformer_outputs.hidden_states[-2]  # second-last layer

def get_second_last_layer_output(model, state, return_to_go, timestep):
    device = next(model.parameters()).device

    if not isinstance(return_to_go, torch.Tensor):
        return_to_go = torch.tensor([return_to_go], dtype=torch.float32)
    return_to_go = return_to_go.to(device)

    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], dtype=torch.long)
    timestep = timestep.to(device)

    state = state.unsqueeze(0).to(device)  # [1, state_dim]
    actions = torch.zeros((1, 1,1), dtype=torch.long, device=device)  # dummy action
    rewards = torch.zeros((1, 1), dtype=torch.float32, device=device)  # dummy reward
    returns = return_to_go.unsqueeze(0).unsqueeze(0)
    timesteps = timestep.unsqueeze(0)

    # Hook to capture hidden state
    second_last_hidden = {}

    # GPT2-style transformer blocks are under `transformer.h`
    handle = model.transformer.h[-2].register_forward_hook(
        lambda module, input, output: second_last_hidden.update({'output': output[0]})
    )

    with torch.no_grad():
        _ = model(
            states=state,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns,
            timesteps=timesteps,
        )

    handle.remove()

    # output is [1, seq_len, hidden_dim]; we want last token from second-last layer
    return second_last_hidden['output'][:, -1]  # shape: [1, hidden_dim]



def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for returns, states, actions, timesteps in dataloader:
            if states.dim() == 2:
                states = states.unsqueeze(1)
                actions = actions.unsqueeze(1)
                returns = returns.unsqueeze(1)
                timesteps = timesteps.unsqueeze(1)

            _, action_logits, _ = model(states, actions, None, returns, timesteps)

            predicted_actions = torch.argmax(action_logits, dim=-1)  # [batch, seq]
            print(predicted_actions.shape)
            correct += (predicted_actions.view(-1) == actions.view(-1)).sum().item()
            total += actions.numel()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

import gym
import d4rl
import torch
import numpy as np

def generate_trajectories_from_d4rl(env_name='CartPole-v1', gamma=0.99):
    env = gym.make(env_name)
    dataset = env.get_dataset()

    trajectories = []
    traj = {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 'timesteps': []}
    
    N = dataset['observations'].shape[0]
    timestep = 0

    for i in range(N):
        traj['observations'].append(dataset['observations'][i])
        traj['actions'].append(dataset['actions'][i])
        traj['rewards'].append(dataset['rewards'][i])
        done_bool = bool(dataset['terminals'][i]) or bool(dataset['timeouts'][i])
        traj['dones'].append(done_bool)
        traj['timesteps'].append(timestep)

        timestep += 1

        if done_bool:
            # Now compute discounted returns
            rewards = traj['rewards']
            returns = np.zeros(len(rewards))
            G = 0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                returns[t] = G

            trajectory = []
            for r, s, a, t in zip(returns, traj['observations'], traj['actions'], traj['timesteps']):
                trajectory.append((r, s, a, t))

            trajectories.append(trajectory)

            # reset for next episode
            traj = {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 'timesteps': []}
            timestep = 0

    env.close()
    return trajectories


def convert_to_trajectory_format(loaded_trajectories, gamma=1):
    converted_trajectories = []
    
    for trajectory in loaded_trajectories:
        # Assuming the format is (state, action, reward, timestep)
        rewards,states, actions, timesteps = zip(*trajectory)
        
        # Compute returns (discounted sum of rewards)
        returns = np.zeros(len(rewards))
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G
        
        # Create new trajectory with returns, states, actions, and timesteps
        converted_trajectory = [(r, s, a, t) for r, (s, a, t) in zip(returns, zip(states, actions, timesteps))]
        converted_trajectories.append(converted_trajectory)
    
    return converted_trajectories







# Main
if __name__ == "__main__":
    # Load the trajectories from the file
    with open("cartpole_trajectories.pkl", "rb") as f:
        loaded_trajectories = pickle.load(f)

# Check the loaded data
    print(f"Loaded {len(loaded_trajectories)} trajectories.")

# Optionally, inspect the first trajectory
    #print("First trajectory data:", loaded_trajectories[0])
    trajectories=convert_to_trajectory_format(loaded_trajectories)
    env_name = 'CartPole-v1'
    n = 25000
    #trajectories = generate_trajectories(n, env_name=env_name)   #original code to generate random traj
    #trajectories = generate_trajectories_from_d4rl()               #code to generate d4rl trajectories
    returns, states, actions, timesteps = prepare_dataset(trajectories)

    # from sklearn.model_selection import train_test_split

    # train_idx, val_idx = train_test_split(np.arange(len(returns)), test_size=0.2, random_state=42)

    # train_dataset = TensorDataset(
    #     returns[train_idx], states[train_idx], actions[train_idx], timesteps[train_idx]
    # )
    # val_dataset = TensorDataset(
    #     returns[val_idx], states[val_idx], actions[val_idx], timesteps[val_idx]
    # )

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    dataset = TensorDataset(returns, states, actions, timesteps)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # dataloader=train_loader

    state_dim = states.shape[-1]
    num_actions = 2  # for cart pole

    print("state_dim:", state_dim)
    print("num_actions:", num_actions)

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=num_actions,
        hidden_size=128,
        max_length=1000,
        action_tanh=False  # Not needed for discrete actions
    )

    # Override action embedding for discrete action space
    model.embed_action = torch.nn.Embedding(num_actions, model.hidden_size)

    train_model(model, dataloader,epochs=5)
    #evaluate_model(model,val_loader)

    torch.save(model.state_dict(), "decision_transformer_mountaincar.pth")

    #max_return = torch.max(returns).item()
    max_return = returns.max().item()
    example_state = states[0].unsqueeze(0)
    example_timestep = timesteps[0].unsqueeze(0)
    print(timesteps[0])
    print(example_timestep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert to tensors and move to the appropriate device
    # max_return = torch.tensor(max_return, dtype=torch.float32).to(device)
    # example_state = torch.tensor(example_state, dtype=torch.float32).to(device)
    # example_timestep = torch.tensor(example_timestep, dtype=torch.long).to(device)

    print("max return shape",max_return)
    print("example_state shape",example_state)
    print("example_timestep shape",example_timestep.shape)

    output = get_second_last_layer_output(model, example_state, max_return, example_timestep)
    print("Second last layer output:", output)
    print(output.shape())
