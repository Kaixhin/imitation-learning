import torch
from torch import nn
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform
from torch.nn import functional as F

import numpy as np


def create_fcnn_layer(state_size, hidden_size, output_size, activation_function, dropout=0, ortho_init=True):
  network_dims = (state_size, hidden_size, hidden_size)
  network_layers = []

  for n in range(len(network_dims) - 1):
    layer = nn.Linear(network_dims[n], network_dims[n + 1])
    if ortho_init:
      nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function._get_name().lower()))
    network_layers.append(layer)
    if dropout > 0:
      network_layers.append(nn.Dropout(p=dropout))
    network_layers.append(activation_function)
  final_layer = nn.Linear(network_dims[-1], output_size)
  if ortho_init:
    nn.init.orthogonal_(final_layer.weight, gain=nn.init.calculate_gain('linear'))
  network_layers.append(final_layer)
  return nn.Sequential(*network_layers)


# Concatenates the state and action (previously one-hot discrete version)
def _join_state_action(state, action, action_size):
    return torch.cat([state, action], dim=1)


# Computes the squared distance between two sets of vectors
def _squared_distance(x, y):
  n_1, n_2, d = x.size(0), y.size(0), x.size(1)
  tiled_x, tiled_y = x.view(n_1, 1, d).expand(n_1, n_2, d), y.view(1, n_2, d).expand(n_1, n_2, d)
  return (tiled_x - tiled_y).pow(2).mean(dim=2)


# Gaussian/radial basis function/exponentiated quadratic kernel
def _gaussian_kernel(x, y, gamma=1):
  return torch.exp(-gamma * _squared_distance(x, y))


class Actor(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, dropout=0, action_scale=1.0, action_loc=-1.0, log_std_init=-0.5, activation_function=nn.Tanh(), ortho_init=True):
    super().__init__()

    self.actor = create_fcnn_layer(state_size, hidden_size, output_size=action_size, activation_function=activation_function, ortho_init=ortho_init, dropout=dropout)

    log_std = log_std_init * np.ones(action_size, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.loc = action_loc
    self.scale = action_scale
    #_orthogonal_initialization(self)

  def forward(self, state):
    std = torch.exp(self.log_std)
    mu = self.actor(state)
    normal = Independent(Normal(mu, std), 1)
    #policy = TransformedDistribution(normal, [TanhTransform(), AffineTransform(loc=self.loc, scale=self.scale)])
    return normal
    #normal.entropy() # Needs double-check with Kai-san can we just take normals entropy term for this?

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.forward(state).log_prob(action)

  def _get_action_uncertainty(self, state, action):
    ensemble_policies = []
    for _ in range(5):  # Perform Monte-Carlo dropout for an implicit ensemble
      ensemble_policies.append(self.log_prob(state, action).exp())
    return torch.stack(ensemble_policies).var(dim=0)

  # Set uncertainty threshold at the 98th quantile of uncertainty costs calculated over the expert data
  def set_uncertainty_threshold(self, expert_state, expert_action):
    self.q = torch.quantile(self._get_action_uncertainty(expert_state, expert_action), 0.98).item()

  def predict_reward(self, state, action):
    # Calculate (raw) uncertainty cost
    uncertainty_cost = self._get_action_uncertainty(state, action)
    # Calculate clipped uncertainty cost
    neg_idxs = uncertainty_cost.less_equal(self.q)
    uncertainty_cost[neg_idxs] = -1
    uncertainty_cost[~neg_idxs] = 1
    return -uncertainty_cost


class Critic(nn.Module):
  def __init__(self, state_size, hidden_size, activation_function=nn.Tanh(), ortho_init=True):
    super().__init__()
    self.critic = create_fcnn_layer(state_size, hidden_size, output_size=1, activation_function=activation_function, ortho_init=ortho_init)

  def forward(self, state):
    value = self.critic(state).squeeze(dim=1)
    return value


class ActorCritic(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, action_scale=1.0, action_loc=0.0, log_std_init=-0.5, ortho_init=True):
    super().__init__()
    self.activation_function = nn.Tanh()
    self.actor = Actor(state_size, action_size, hidden_size, action_scale=action_scale, action_loc=action_loc, log_std_init=log_std_init, activation_function=self.activation_function, ortho_init=ortho_init)
    self.critic = Critic(state_size, hidden_size, activation_function=self.activation_function, ortho_init=ortho_init)

  def forward(self, state):
    policy, value = self.actor(state), self.critic(state)
    return policy, value

  def greedy_action(self, state):
    policy = self.actor(state)
    return policy.mean

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.actor.log_prob(state, action)


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, state_only=False, forward_kl=False):
    super().__init__()
    self.action_size, self.state_only, self.forward_kl = action_size, state_only, forward_kl
    input_layer = nn.Linear(state_size if state_only else state_size + action_size, hidden_size)
    self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid())

  def forward(self, state, action):
    D = self.discriminator(state if self.state_only else _join_state_action(state, action, self.action_size)).squeeze(dim=1)
    return D
  
  def predict_reward(self, state, action):
    D = self.forward(state, action)
    h = torch.log(D) - torch.log1p(-D)
    return torch.exp(h * -h) if self.forward_kl else h


class GMMILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, state_only=True):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.gamma_1, self.gamma_2 = None, None

  def predict_reward(self, state, action, expert_state, expert_action):
    state_action = state if self.state_only else _join_state_action(state, action, self.action_size)
    expert_state_action = expert_state if self.state_only else _join_state_action(expert_state, expert_action, self.action_size)
    
    # Use median heuristics to set data-dependent bandwidths
    if self.gamma_1 is None:
      self.gamma_1 = 1 / _squared_distance(state_action, expert_state_action).median().item()
      self.gamma_2 = 1 / _squared_distance(expert_state_action, expert_state_action).median().item()

    # Calculate negative of witness function (based on kernel mean embeddings)
    return (_gaussian_kernel(expert_state_action, state_action, gamma=self.gamma_1).mean(dim=0) + _gaussian_kernel(expert_state_action, state_action, gamma=self.gamma_2).mean(dim=0)) # - (_gaussian_kernel(state_action, state_action, gamma=self.gamma_1).mean(dim=0) + _gaussian_kernel(state_action, state_action, gamma=self.gamma_2).mean(dim=0))  TODO: GMMIL seems to fail with the self-similarity terms (commented out)? 


class AIRLDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, discount, state_only=False, ):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.discount = discount
    self.g = nn.Linear(state_size if state_only else state_size + action_size, 1)  # Reward function r
    self.h = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))  # Shaping function Φ

  def reward(self, state, action):
    if self.state_only:
      return self.g(state).squeeze(dim=1)
    else:
      return self.g(_join_state_action(state, action, self.action_size)).squeeze(dim=1)

  def value(self, state):
    return self.h(state).squeeze(dim=1)

  def forward(self, state, action, next_state, policy, terminal):
    f = self.reward(state, action) + (1 - terminal) * (self.discount * self.value(next_state) - self.value(state))
    f_exp = f.exp()
    return f_exp / (f_exp + policy)

  def predict_reward(self, state, action, next_state, policy, terminal):
    D = self.forward(state, action, next_state, policy, terminal)
    return torch.log(D) - torch.log1p(-D)


class EmbeddingNetwork(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.embedding = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, input_size))

  def forward(self, input):
    return self.embedding(input)


class REDDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, state_only=False):
    super().__init__()
    self.action_size, self.state_only = action_size, state_only
    self.gamma = None
    self.predictor = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size)
    self.target = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size)
    for param in self.target.parameters():
      param.requires_grad = False

  def forward(self, state, action):
    state_action = state if self.state_only else _join_state_action(state, action, self.action_size)
    prediction, target = self.predictor(state_action), self.target(state_action)
    return prediction, target

  def predict_reward(self, state, action, sigma=1):  # TODO: Set sigma based such that r(s, a) from expert demonstrations ≈ 1
    prediction, target = self.forward(state, action)
    return _gaussian_kernel(prediction, target, gamma=sigma).mean(dim=1)
