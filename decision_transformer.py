import torch
import torch.nn as nn
import transformers

from transformers import GPT2Model, GPT2Config

class DecisionTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, timestep_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        config = GPT2Config(
            vocab_size=1,  # Doesn't matter -- we don't use the vocab
            output_hidden_states=True,
            n_embd=hidden_size,
            n_head=16,  # Choose a divisor of 256
            **kwargs
        )

        # Use Hugging Face's latest GPT2 model
        self.transformer = GPT2Model(config)

        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        #self.embed_action = nn.Linear(act_dim, hidden_size)                #THIS IS FOR CONTINOUS
        self.embed_action = nn.Embedding(act_dim, hidden_size)         #THIS IS FOR DISCRETE ACTION
        self.embed_timestep = nn.Embedding(max_length, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        actions = actions.long().squeeze(-1)   #ONLY FOR DISCRETE LIKE MOUNTAIN CAR COMMENT FOR REACHER
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        timestep_embeddings = self.embed_timestep(timesteps).squeeze(-2)

        # print("timestep_embeddings", timestep_embeddings.shape)
        # print("action_embeddings shape",action_embeddings.shape)
        # print("state_embeddings shape",state_embeddings.shape)
        # print("returns_embeddings shape",returns_embeddings.shape)

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings, timestep_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 4 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs.last_hidden_state

        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 3])
        state_preds = self.predict_state(x[:, 3])
        action_preds = self.predict_action(x[:, 2])

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, states.shape[-1])
        actions = actions.reshape(1, -1, actions.shape[-1])
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        attention_mask = torch.ones(states.shape[:-1], dtype=torch.long, device=states.device)

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]
