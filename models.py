import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


# Concatenates the state and one-hot version of an action
def _join_state_action(state, action, action_size):
  return torch.cat([state, F.one_hot(action, action_size).to(dtype=torch.float32)], dim=1)


# Computes the scaled squared distance between two sets of vectors
def _squared_distance(X, Y, lengthscale=1):
  X, Y = X / lengthscale, Y / lengthscale
  XX = X.pow(2).sum(1, keepdim=True)
  YY = Y.pow(2).sum(1, keepdim=True)
  XY = X @ Y.t()
  return torch.clamp(XX - 2 * XY + YY.t(), min=0)


# Gaussian/radial basis function/exponentiated quadratic kernel
def _gaussian_kernel(X, Y, variance=1, lengthscale=1):
  return variance * torch.exp(-0.5 * _squared_distance(X, Y, lengthscale=lengthscale))


class Actor(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(
        hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size))

  def forward(self, state):
    policy = Categorical(logits=self.actor(state))
    return policy


class Critic(nn.Module):
  def __init__(self, state_size, hidden_size):
    super().__init__()
    self.critic = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(
    ), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

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

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.actor(state).log_prob(action)


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, state_only=False):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    input_layer = nn.Linear(
        state_size if state_only else state_size + action_size, hidden_size)
    self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(
        hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid())

  def forward(self, state, action):
    D = self.discriminator(state if self.state_only else _join_state_action(
        state, action, self.action_size)).squeeze(dim=1)
    return D

  def predict_reward(self, state, action):
    D = self.forward(state, action)
    return torch.log(D) - torch.log1p(-D)


class GMMILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, state_only=True):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only

  def predict_reward(self, state, action, expert_state, expert_action):
    state_action = state if self.state_only else _join_state_action(
        state, action, self.action_size)
    expert_state_action = expert_state if self.state_only else _join_state_action(
        expert_state, expert_action, self.action_size)
    # TODO: Use median heuristics to select 2 data-dependent bandwidths
    # Return maximum mean discrepancy
    return _gaussian_kernel(state_action, expert_state_action).mean(dim=1)


class AIRLDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, discount, state_only=False):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.discount = discount
    self.g = nn.Linear(state_size if state_only else state_size +
                       action_size, 1)  # Reward function r
    self.h = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(
        hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))  # Shaping function Φ

  def reward(self, state, action):
    return self.g(state if self.state_only else _join_state_action(state, action, self.action_size)).squeeze(dim=1)

  def value(self, state):
    return self.h(state).squeeze(dim=1)

  def forward(self, state, action, next_state, policy):
    f = self.reward(state, action) + self.discount * \
        self.value(next_state) - self.value(state)
    f_exp = f.exp()
    return f_exp / (f_exp + policy)

  def predict_reward(self, state, action, next_state, policy):
    D = self.forward(state, action, next_state, policy)
    return torch.log(D) - torch.log1p(-D)
