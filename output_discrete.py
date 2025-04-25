import torch
from decision_transformer import DecisionTransformer
from trainer_discrete import get_second_last_layer_output, prepare_dataset
from Generate_Reacher_Trajectories import generate_trajectories

if __name__ == "__main__":
    env_name = 'LunarLander-v3'
    n = 10  # Use fewer trajectories for quick inspection
    trajectories = generate_trajectories(n, env_name=env_name)

    # Prepare dataset
    returns, states, actions, timesteps = prepare_dataset(trajectories)
    state_dim = states.shape[-1]
    num_actions = 4

    # Initialize model and override embedding for discrete actions
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=num_actions,
        hidden_size=256,
        max_length=1000,
        action_tanh=False
    )
    model.embed_action = torch.nn.Embedding(num_actions, model.hidden_size)

    # Load weights
    model.load_state_dict(torch.load("decision_transformer_mountaincar.pth",weights_only=True))
    model.eval()

    # Choose example input
    max_return = torch.max(returns).item()
    example_state = states[0].unsqueeze(0)  # shape [1, state_dim]
    example_timestep = timesteps[0].unsqueeze(0)  # shape [1]

    # Get transformer embedding
    output = get_second_last_layer_output(model, example_state, max_return, example_timestep)
    print("Second last layer output shape:", output.shape)

    # Save embedding to file
    torch.save(output, "second_last_layer_embedding.pt")
    print("Embedding saved to second_last_layer_embedding.pt")
