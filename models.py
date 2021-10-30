import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import Parameter, functional as F
from torch.nn.utils import parametrizations
from omegaconf import DictConfig

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
def _create_fcnn(input_size, hidden_size, depth, output_size, activation_function, input_dropout=0, dropout=0, final_gain=1, spectral_norm=False):
  assert activation_function in ACTIVATION_FUNCTIONS.keys()
  
  network_dims, layers = (input_size, *[hidden_size] * depth), []
  if input_dropout > 0:
    layers.append(nn.Dropout(p=input_dropout))
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
  def __init__(self, state_size, action_size, model_cfg: DictConfig):
    super().__init__()
    self.log_std_dev_min, self.log_std_dev_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    self.actor = _create_fcnn(state_size, model_cfg.hidden_size, model_cfg.depth, output_size=2 * action_size, activation_function=model_cfg.activation, input_dropout=model_cfg.get('input_dropout', 0), dropout=model_cfg.get('dropout', 0))

  def forward(self, state):
    mean, log_std_dev = self.actor(state).chunk(2, dim=1)
    log_std_dev = torch.clamp(log_std_dev, min=self.log_std_dev_min, max=self.log_std_dev_max)
    policy = TransformedDistribution(Independent(Normal(mean, log_std_dev.exp()), 1), TanhTransform(cache_size=1))  # Restrict action range to (-1, 1)
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
  def __init__(self, state_size, action_size, model_cfg: DictConfig):
    super().__init__()
    self.critic = _create_fcnn(state_size + action_size, model_cfg.hidden_size, model_cfg.depth, output_size=1, activation_function=model_cfg.activation)

  def forward(self, state, action):
    value = self.critic(_join_state_action(state, action)).squeeze(dim=1)
    return value


class TwinCritic(nn.Module):
  def __init__(self, state_size, action_size, model_cfg: DictConfig):
    super().__init__()
    self.critic_1 = Critic(state_size, action_size, model_cfg)
    self.critic_2 = Critic(state_size, action_size, model_cfg)

  def forward(self, state, action):
    value_1, value_2 = self.critic_1(state, action), self.critic_2(state, action)
    return value_1, value_2


  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state, action):
    return self.actor.log_prob(state, action)


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, imitation_cfg: DictConfig, discount):
    super().__init__()
    model_cfg = imitation_cfg.model
    self.discount, self.state_only, self.reward_shaping, self.reward_function = discount, imitation_cfg.state_only, model_cfg.reward_shaping, model_cfg.reward_function
    if self.reward_shaping:
      self.g = nn.Linear(state_size if self.state_only else state_size + action_size, 1)  # Reward function r
      if imitation_cfg.spectral_norm: self.g = parametrizations.spectral_norm(self.g)
      self.h = _create_fcnn(state_size, model_cfg.hidden_size, model_cfg.depth, 1, activation_function=model_cfg.activation, spectral_norm=imitation_cfg.spectral_norm)  # Shaping function Φ
    else:
      self.g = _create_fcnn(state_size if self.state_only else state_size + action_size, model_cfg.hidden_size, model_cfg.depth, 1, activation_function=model_cfg.activation, spectral_norm=imitation_cfg.spectral_norm)

  def _reward(self, state, action):
    if self.state_only:
      return self.g(state).squeeze(dim=1)
    else:
      return self.g(_join_state_action(state, action)).squeeze(dim=1)

  def _value(self, state):
    return self.h(state).squeeze(dim=1)

  def forward(self, state, action, next_state=None, log_policy=None, terminal=None):
    if self.reward_shaping:
      f = self._reward(state, action) + (1 - terminal) * (self.discount * self._value(next_state) - self._value(state))
      return f - log_policy  # Note that this is equivalent to sigmoid^-1(e^f / (e^f + π))
    else:
      return self.g(state if self.state_only else _join_state_action(state, action)).squeeze(dim=1)
  
  def predict_reward(self, state, action, next_state=None, log_policy=None, terminal=None):
    D = torch.sigmoid(self.forward(state, action, next_state=next_state, log_policy=log_policy, terminal=terminal))
    h = -torch.log1p(-D + 1e-6) if self.reward_function == 'GAIL' else torch.log(D + 1e-6) - torch.log1p(-D + 1e-6)  # Add epsilon to improve numerical stability given limited floating point precision
    return torch.exp(h) * -h if self.reward_function == 'FAIRL' else h  # FAIRL reward function is based on AIRL reward function


class GMMILDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, imitation_cfg: DictConfig):
    super().__init__()
    self.state_only = imitation_cfg.state_only
    self.gamma_1, self.gamma_2, self.self_similarity = None, None, imitation_cfg.self_similarity

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
  def __init__(self, input_size, model_cfg: DictConfig):
    super().__init__()
    self.embedding = _create_fcnn(input_size, model_cfg.hidden_size, model_cfg.depth, input_size, model_cfg.activation)

  def forward(self, input):
    return self.embedding(input)


class REDDiscriminator(nn.Module):
  def __init__(self, state_size, action_size, imitation_cfg: DictConfig):
    super().__init__()
    self.state_only = imitation_cfg.state_only
    self.sigma_1 = None
    self.predictor = EmbeddingNetwork(state_size if self.state_only else state_size + action_size, imitation_cfg.model)
    self.target = EmbeddingNetwork(state_size if self.state_only else state_size + action_size, imitation_cfg.model)
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
