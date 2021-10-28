import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import Parameter, functional as F
from torch.nn.utils import parametrizations

ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}


# Concatenates the state and action
def _join_state_action(state, action):
    return torch.cat([state, action], dim=1)


# Computes the squared distance between two sets of vectors
def _squared_distance(x, y):
  n_1, n_2, d = x.size(0), y.size(0), x.size(1)
  tiled_x, tiled_y = x.view(n_1, 1, d).expand(n_1, n_2, d), y.view(1, n_2, d).expand(n_1, n_2, d)
  return (tiled_x - tiled_y).pow(2).mean(dim=2)


# Gaussian/radial basis function/exponentiated quadratic kernel
def _gaussian_kernel(x, y, gamma=1):
  return torch.exp(-gamma * _squared_distance(x, y))


# Creates a sequential fully-connected network
def _create_fcnn(input_size, hidden_size, output_size, activation_function, dropout=0, final_gain=1, spectral_norm=False):
  assert activation_function in ACTIVATION_FUNCTIONS.keys()
  
  network_dims, layers = (input_size, hidden_size, hidden_size), []

  for l in range(len(network_dims) - 1):
    layer = nn.Linear(network_dims[l], network_dims[l + 1])
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function))
    nn.init.constant_(layer.bias, 0)
    if spectral_norm: layer = parametrizations.spectral_norm(layer)
    layers.append(layer)
    if dropout > 0: layers.append(nn.Dropout(p=dropout))
    layers.append(ACTIVATION_FUNCTIONS[activation_function]())

  final_layer = nn.Linear(network_dims[-1], output_size)
  nn.init.orthogonal_(final_layer.weight, gain=final_gain)
  nn.init.constant_(final_layer.bias, 0)
  if spectral_norm: final_layer = parametrizations.spectral_norm(final_layer)
  layers.append(final_layer)

  return nn.Sequential(*layers)


def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network, target_network, polyak_factor):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data.mul_(polyak_factor).add_((1 - polyak_factor) * param.data)


# Creates a batch of training data made from a mix of expert and policy data; rewrites transitions in-place
def sqil_sample(transitions, expert_transitions, batch_size):
  transitions['states'][:batch_size // 2], transitions['actions'][:batch_size // 2], transitions['next_states'][:batch_size // 2], transitions['terminals'][:batch_size // 2], transitions['weights'][:batch_size // 2], transitions['absorbing'][:batch_size // 2]  = expert_transitions['states'][:batch_size // 2], expert_transitions['actions'][:batch_size // 2], expert_transitions['next_states'][:batch_size // 2], expert_transitions['terminals'][:batch_size // 2], expert_transitions['weights'][:batch_size // 2], expert_transitions['absorbing'][:batch_size // 2]  # Replace half of the batch with expert data
  transitions['rewards'][:batch_size // 2], transitions['rewards'][batch_size // 2:] = 1, 0  # Set a constant +1 reward for expert data and 0 for policy data


class SoftActor(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, activation_function, dropout=0):
    super().__init__()
    self.actor = _create_fcnn(state_size, hidden_size, output_size=2 * action_size, activation_function=activation_function, dropout=dropout)

  def forward(self, state):
    mean, log_std_dev = self.actor(state).chunk(2, dim=1)
    policy = TransformedDistribution(Independent(Normal(mean, F.softplus(log_std_dev) + 0.001), 1), TanhTransform(cache_size=1))  # Restrict action range to (-1, 1)
    return policy

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.forward(state).log_prob(action)

  def get_greedy_action(self, state):
    return torch.tanh(self.forward(state).base_dist.mean)

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
  def __init__(self, state_size, action_size, hidden_size, activation_function):
    super().__init__()
    self.critic = _create_fcnn(state_size + action_size, hidden_size, output_size=1, activation_function=activation_function)

  def forward(self, state, action):
    value = self.critic(_join_state_action(state, action)).squeeze(dim=1)
    return value


class TwinCritic(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, activation_function):
    super().__init__()
    self.critic_1 = Critic(state_size, action_size, hidden_size, activation_function=activation_function)
    self.critic_2 = Critic(state_size, action_size, hidden_size, activation_function=activation_function)

  def forward(self, state, action):
    value_1, value_2 = self.critic_1(state, action), self.critic_2(state, action)
    return value_1, value_2


  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.actor.log_prob(state, action)


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, activation_function, state_only=False, forward_kl=False, spectral_norm=False):
    super().__init__()
    self.state_only, self.forward_kl = state_only, forward_kl
    self.discriminator = _create_fcnn(state_size if state_only else state_size + action_size, hidden_size, 1, activation_function, spectral_norm=spectral_norm)

  def forward(self, state, action):
    D = self.discriminator(state if self.state_only else _join_state_action(state, action)).squeeze(dim=1)
    return D
  
  def predict_reward(self, state, action):
    D = torch.sigmoid(self.forward(state, action))
    h = torch.log(D + 1e-6) - torch.log1p(-D + 1e-6) # Add epsilon to improve numerical stability given limited floating point precision
    return torch.exp(h) * -h if self.forward_kl else h


class AIRLDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, discount, activation_function, state_only=False, spectral_norm=False):
    super().__init__()
    self.state_only = state_only
    self.discount = discount
    self.g = nn.Linear(state_size if state_only else state_size + action_size, 1)  # Reward function r
    if spectral_norm: self.g = parametrizations.spectral_norm(self.g)
    self.h = _create_fcnn(state_size, hidden_size, 1, activation_function, spectral_norm=spectral_norm)  # Shaping function Φ

  def reward(self, state, action):
    if self.state_only:
      return self.g(state).squeeze(dim=1)
    else:
      return self.g(_join_state_action(state, action)).squeeze(dim=1)

  def value(self, state):
    return self.h(state).squeeze(dim=1)

  def forward(self, state, action, next_state, log_policy, terminal):
    f = self.reward(state, action) + (1 - terminal) * (self.discount * self.value(next_state) - self.value(state))
    return f - log_policy  # Note that this is equivalent to sigmoid^-1(e^f / (e^f + π))

  def predict_reward(self, state, action, next_state, log_policy, terminal):
    D = torch.sigmoid(self.forward(state, action, next_state, log_policy, terminal))
    return torch.log(D + 1e-6) - torch.log1p(-D + 1e-6) # Add epsilon to improve numerical stability given limited floating point precision


class GMMILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, self_similarity=True, state_only=True):
    super().__init__()
    self.state_only = state_only
    self.gamma_1, self.gamma_2, self.self_similarity = None, None, self_similarity

  def predict_reward(self, state, action, expert_state, expert_action):
    state_action = state if self.state_only else _join_state_action(state, action)
    expert_state_action = expert_state if self.state_only else _join_state_action(expert_state, expert_action)
    
    # Use median heuristics to set data-dependent bandwidths
    if self.gamma_1 is None:
      self.gamma_1 = 1 / (_squared_distance(state_action, expert_state_action).median().item() + 1e-8)  # Add epsilon for numerical stability (if distance is zero)
      self.gamma_2 = 1 / (_squared_distance(expert_state_action.transpose(0, 1), expert_state_action.transpose(0, 1)).median().item() + 1e-8)  # Add epsilon for numerical stability (if distance is zero)

    # Calculate negative of witness function (based on kernel mean embeddings)
    similarity = (_gaussian_kernel(expert_state_action, state_action, gamma=self.gamma_1).mean(dim=0) + _gaussian_kernel(expert_state_action, state_action, gamma=self.gamma_2).mean(dim=0))
    return similarity - (_gaussian_kernel(state_action, state_action, gamma=self.gamma_1).mean(dim=0) + _gaussian_kernel(state_action, state_action, gamma=self.gamma_2).mean(dim=0)) if self.self_similarity else similarity


class EmbeddingNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, activation_function):
    super().__init__()
    self.embedding = _create_fcnn(input_size, hidden_size, input_size, activation_function)

  def forward(self, input):
    return self.embedding(input)


class REDDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, activation_function, state_only=False):
    super().__init__()
    self.state_only = state_only
    self.sigma_1 = None
    self.predictor = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size, activation_function)
    self.target = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size, activation_function)
    for param in self.target.parameters():
      param.requires_grad = False

  def forward(self, state, action):
    state_action = state if self.state_only else _join_state_action(state, action)
    prediction, target = self.predictor(state_action), self.target(state_action)
    return prediction, target

  # Originally, sets σ based such that r(s, a) from expert demonstrations ≈ 1; instead this uses kernel median heuristic (same as GMMIL)
  def set_sigma(self, expert_state, expert_action):
    prediction, target = self.forward(expert_state, expert_action)
    self.sigma_1 = 1 / _squared_distance(prediction.transpose(0, 1), target.transpose(0, 1)).median().item()

  def predict_reward(self, state, action):
    prediction, target = self.forward(state, action)
    return _gaussian_kernel(prediction, target, gamma=self.sigma_1).mean(dim=1)
