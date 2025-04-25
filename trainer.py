import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Generate_Reacher_Trajectories import generate_trajectories  # Import the function
from decision_transformer import DecisionTransformer  # Import Decision Transformer

# Prepare dataset
def prepare_dataset(trajectories):
    returns, states, actions, timesteps = [], [], [], []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            r, s, a, t = traj[i]
            returns.append([r])
            states.append(s)
            actions.append(a)
            timesteps.append([t])
    return torch.tensor(returns, dtype=torch.float32), torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), torch.tensor(timesteps, dtype=torch.long)

# Training function
def train_model(model, dataloader, epochs=50, lr=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for returns, states, actions, timesteps in dataloader:
            optimizer.zero_grad()

            # Ensure correct batch and sequence dimensions
            if states.dim() == 2:  # If shape is (batch, feature_dim), add sequence dimension
                states = states.unsqueeze(1)
                actions = actions.unsqueeze(1)
                returns = returns.unsqueeze(1)
                timesteps = timesteps.unsqueeze(1)

            _, action_preds, _ = model(states, actions, None, returns, timesteps)  

            loss = criterion(action_preds, actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# Function to get second last layer output
def get_second_last_layer_output(model, state, max_return, timestep):
    state = state.reshape(1, 1, -1)  # Add batch and sequence dimensions
    max_return = torch.tensor([[max_return]], dtype=torch.float32)
    timestep = torch.tensor([[timestep]], dtype=torch.long)

    state_embedding = model.embed_state(state)
    return_embedding = model.embed_return(max_return)
    timestep_embedding = model.embed_timestep(timestep)

    input_embedding = state_embedding + return_embedding + timestep_embedding
    input_embedding = model.embed_ln(input_embedding)

    # Ensure transformer outputs hidden states
    transformer_outputs = model.transformer(inputs_embeds=input_embedding)

    if transformer_outputs.hidden_states is None:
        raise ValueError("Hidden states were not returned. Ensure 'output_hidden_states=True' in GPT2Config.")

    second_last_layer_output = transformer_outputs.hidden_states[-2]

    return second_last_layer_output


# Main Execution
if __name__ == "__main__":
    n = 100  # Number of trajectories
    trajectories = generate_trajectories(n,env_name='MountainCar-v0')
    returns, states, actions, timesteps = prepare_dataset(trajectories)
    dataset = TensorDataset(returns, states, actions, timesteps)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    print("state_dim",state_dim)
    print("action_dim",action_dim)
    
    model = DecisionTransformer(state_dim=state_dim, act_dim=action_dim, hidden_size=256, max_length=1000)
    
    train_model(model, dataloader)
    
    torch.save(model.state_dict(), "decision_transformer_reacher.pth")
    
     # Example usage of second last layer output
    max_return = torch.max(returns).item()
    example_state = states[0].unsqueeze(0)  # Ensure batch dimension
    example_timestep = timesteps[0].unsqueeze(0)  # Ensure batch dimension
    
    output = get_second_last_layer_output(model, example_state, max_return, example_timestep)
    print("Second last layer output:", output)

