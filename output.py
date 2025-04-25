import torch
from decision_transformer import DecisionTransformer

# Function to get second last layer output
def get_second_last_layer_output(model, state, max_return, timestep):
    state = state.reshape(1, 1, -1)  # (batch, seq_len, state_dim)
    max_return = torch.tensor([[max_return]], dtype=torch.float32)
    timestep = torch.tensor([[timestep]], dtype=torch.long)

    state_embedding = model.embed_state(state)
    return_embedding = model.embed_return(max_return)
    timestep_embedding = model.embed_timestep(timestep)

    input_embedding = state_embedding + return_embedding + timestep_embedding
    input_embedding = model.embed_ln(input_embedding)

    # Forward pass with hidden states
    transformer_outputs = model.transformer(inputs_embeds=input_embedding, output_hidden_states=True)

    if transformer_outputs.hidden_states is None:
        raise ValueError("Hidden states were not returned. Ensure 'output_hidden_states=True' in transformer config.")

    second_last_layer_output = transformer_outputs.hidden_states[-2]
    return second_last_layer_output

# Example usage
if __name__ == "__main__":
    state_dim = 2  # Adjust if needed
    action_dim = 19900
    model = DecisionTransformer(state_dim=state_dim, act_dim=action_dim, hidden_size=256, max_length=1000)

    model.load_state_dict(torch.load("decision_transformer_mountaincar.pth"))
    model.eval()

    # Example input
    example_state = torch.tensor([[-0.5, 0.02]], dtype=torch.float32)  # shape: (1, state_dim)
    max_return = 1.0
    timestep = 10

    output = get_second_last_layer_output(model, example_state, max_return, timestep)
    print("Second last layer output:", output)
