import torch
from decision_transformer import DecisionTransformer
from trainer_discrete import get_second_last_layer_output, prepare_dataset
from Generate_Reacher_Trajectories import generate_trajectories

import gym
import numpy as np

def evaluate_decision_transformer(model, env_name, num_episodes=10, max_ep_len=500):
    env = gym.make(env_name)
    model.eval()

    returns = []
    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # For Gymnasium compatibility

        #print("state is",state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
        #print("new state is",state)
        
        states = state  # [1, state_dim]
        
        # ❗ Important: initialize action and reward with dummy values
        actions = torch.zeros((1, 1), dtype=torch.long)  # Discrete action (not one-hot yet)
        rewards = torch.zeros((1, 1), dtype=torch.float32)
        
        timesteps = torch.tensor([[0]], dtype=torch.long)

        ep_return = 0
        for t in range(max_ep_len):
            rtg = torch.tensor([[max(0, max_ep_len - ep_return)]], dtype=torch.float32)

            with torch.no_grad():
                
                state_preds, action_preds, reward_preds = model(
                    states=state.unsqueeze(0),      # add batch dim
                    actions=actions.unsqueeze(0),
                    rewards=rewards.unsqueeze(0),
                    returns_to_go=rtg.unsqueeze(0),
                    timesteps=timesteps.unsqueeze(0),
                )
                # print(action_preds)
                action_logits = action_preds[0, -1]
                action = action_logits.argmax().item()

            next_state, reward, done, *_ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            ep_return += reward

            # Update buffers
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            # states_buffer = torch.cat([states_buffer, next_state], dim=0)

            # actions_buffer = torch.cat([actions_buffer, torch.tensor([[action]], dtype=torch.long)], dim=0)
            # rewards_buffer = torch.cat([rewards_buffer, torch.tensor([[reward]], dtype=torch.float32)], dim=0)
            # timesteps_buffer = torch.cat([timesteps_buffer, torch.tensor([[t+1]], dtype=torch.long)], dim=0)
            state=next_state
            timesteps=torch.tensor([[t+1]], dtype=torch.long)
            

            if done:
                break
        
        returns.append(ep_return)
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"Average Return over {num_episodes} episodes: {avg_return:.2f} ± {std_return:.2f}")
    return avg_return





if __name__ == "__main__":
    env_name = 'CartPole-v1'
    n = 10  # Use fewer trajectories for quick inspection
    trajectories = generate_trajectories(n, env_name=env_name)

    # Prepare dataset
    returns, states, actions, timesteps = prepare_dataset(trajectories)
    state_dim = states.shape[-1]
    num_actions = 2 # 2 for cart pole

    # Initialize model and override embedding for discrete actions
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=num_actions,
        hidden_size=128,
        max_length=1000,
        action_tanh=False
    )
    model.embed_action = torch.nn.Embedding(num_actions, model.hidden_size)

    # Load weights
    model.load_state_dict(torch.load("decision_transformer_mountaincar.pth",weights_only=True))
    model.eval()

    # Choose example input
    #max_return = torch.max(returns).item()
    max_return = 500
    example_state = states[0].unsqueeze(0)  # shape [1, state_dim]
    example_timestep = timesteps[0].unsqueeze(0)  # shape [1]

    # Get transformer embedding
    output = get_second_last_layer_output(model, example_state, max_return, example_timestep)
    print("Second last layer output shape:", output.shape)
    print("second last layer output",output)

    # Save embedding to file
    # torch.save(output, "second_last_layer_embedding.pt")
    # print("Embedding saved to second_last_layer_embedding.pt")

    evaluate_decision_transformer(model, env_name)
