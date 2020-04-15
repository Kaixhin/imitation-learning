import torch
from torch import nn
from torch.distributions import Categorical


class Actor(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size))

  def forward(self, state):
    policy = Categorical(logits=self.actor(state))
    return policy


class Critic(nn.Module):
  def __init__(self, state_size, hidden_size):
    super().__init__()
    self.critic = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

  def forward(self, state):
    value = self.critic(state).squeeze(dim=1)
    return value


class ActorCritic(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super().__init__()
    self.actor = Actor(state_size, action_size, hidden_size)
    self.critic = Critic(state_size, hidden_size)

  def forward(self, state):
    policy, value = self.actor(state), self.critic(state)
    return policy, value

  def log_prob(self, state, action):
    return self.actor(state).log_prob(action)


class AddLinearEmbed(nn.Module):
  def __init__(self, linear_size, embedding_size, output_size):
    super().__init__()
    self.linear = nn.Linear(linear_size, output_size)
    self.embedding = nn.Embedding(embedding_size, output_size)

  def forward(self, x):
    return self.linear(x[0]) + self.embedding(x[1])


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, state_only=True):
    super().__init__()
    self.state_only = state_only
    input_layer = nn.Linear(state_size, hidden_size) if state_only else AddLinearEmbed(state_size, action_size, hidden_size)
    self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid())

  def forward(self, state, action):
    D = self.discriminator(state if self.state_only else (state, action)).squeeze(dim=1)
    return D
  
  def predict_rewards(self, state, action):
    with torch.no_grad():
      D = self.forward(state, action)
      return torch.log(D) - torch.log1p(-D)


class AIRLDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, discount, state_only=False):
    super().__init__()
    self.state_only = state_only
    self.discount = discount
    self.g = nn.Linear(state_size, 1) if state_only else AddLinearEmbed(state_size, action_size, 1)  # Reward function r
    self.h = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))  # Shaping function Φ

  def get_reward(self, state_action):
    return self.g(state_action[0] if self.state_only else state_action).squeeze(dim=1)

  def get_value(self, state):
    return self.h(state).squeeze(dim=1)

  def forward(self, state, action, next_state, policy):
    f = self.get_reward((state, ) if self.state_only else (state, action)) + self.discount * self.get_value(next_state) - self.get_value(state)
    f_exp = f.exp()
    return f_exp / (f_exp + policy)

  def predict_rewards(self, state, action, next_state, policy):
    with torch.no_grad():
      D = self.forward(state, action, next_state, policy)
      return torch.log(D) - torch.log1p(-D)
