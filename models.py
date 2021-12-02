import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import Parameter, functional as F
from torch.nn.utils import parametrizations
from omegaconf import DictConfig

ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}


# Concatenates the state and action
def _join_state_action(state: Tensor, action: Tensor) -> Tensor:
    return torch.cat([state, action], dim=1)


# Computes the squared distance between two sets of vectors
def _squared_distance(x: Tensor, y: Tensor) -> Tensor:
  n_1, n_2, d = x.size(0), y.size(0), x.size(1)
  tiled_x, tiled_y = x.view(n_1, 1, d).expand(n_1, n_2, d), y.view(1, n_2, d).expand(n_1, n_2, d)
  return (tiled_x - tiled_y).pow(2).mean(dim=2)


# Gaussian/radial basis function/exponentiated quadratic kernel
def _gaussian_kernel(x: Tensor, gamma: float=1) -> Tensor:
  return torch.exp(-gamma * x)


def _weighted_similarity(XY: Tensor, w_x: Tensor, w_y: Tensor, gamma: float=1) -> Tensor:
  return torch.einsum('i,ij,j->i', [w_x, _gaussian_kernel(XY, gamma=gamma), w_y])


def _weighted_median(x: Tensor, weights: Tensor) -> Tensor:
  x_sorted, indices = torch.sort(x.flatten())
  weights_norm_sorted = (weights.flatten() / weights.sum())[indices]  # Normalise and rearrange weights according to sorting
  median_index = torch.min((torch.cumsum(weights_norm_sorted, dim=0) >= 0.5).nonzero())
  return x_sorted[median_index]


# Creates a sequential fully-connected network
def _create_fcnn(input_size: int, hidden_size: int, depth: int, output_size: int, activation_function: str, input_dropout: float=0, dropout: float=0, final_gain: float=1, spectral_norm: bool=False) -> nn.Module:
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


def create_target_network(network: nn.Module) -> nn.Module:
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network: nn.Module, target_network: nn.Module, polyak_factor: float):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data.mul_(polyak_factor).add_((1 - polyak_factor) * param.data)


class SoftActor(nn.Module):
  def __init__(self, state_size: int, action_size: int, model_cfg: DictConfig):
    super().__init__()
    self.log_std_dev_min, self.log_std_dev_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    self.actor = _create_fcnn(state_size, model_cfg.hidden_size, model_cfg.depth, output_size=2 * action_size, activation_function=model_cfg.activation, input_dropout=model_cfg.get('input_dropout', 0), dropout=model_cfg.get('dropout', 0))

  def forward(self, state: Tensor) -> Distribution:
    mean, log_std_dev = self.actor(state).chunk(2, dim=1)
    log_std_dev = torch.clamp(log_std_dev, min=self.log_std_dev_min, max=self.log_std_dev_max)
    policy = TransformedDistribution(Independent(Normal(mean, log_std_dev.exp()), 1), TanhTransform(cache_size=1))  # Restrict action range to (-1, 1)
    return policy

  # Calculates the log probability of an action a with the policy π(·|s) given state s
  def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
    return self.forward(state).log_prob(action)

  def get_greedy_action(self, state: Tensor) -> Tensor:
    return torch.tanh(self.forward(state).base_dist.mean)

  def _get_action_uncertainty(self, state: Tensor, action: Tensor) -> Tensor:
    state, action = torch.repeat_interleave(state, 5, dim=0), torch.repeat_interleave(action, 5, dim=0)  # Repeat state and actions x ensemble size
    prob = self.log_prob(state, action).exp()  # Perform Monte-Carlo dropout for an implicit ensemble; PyTorch implementation does not share masks across a batch (all independent)
    return prob.view(-1, 5).var(dim=1)  # Resized tensor is batch size x ensemble size

  # Set uncertainty threshold at the 98th quantile of uncertainty costs calculated over the expert data
  def set_uncertainty_threshold(self, expert_state: Tensor, expert_action: Tensor):
    self.q = torch.quantile(self._get_action_uncertainty(expert_state, expert_action), 0.98).item()

  def predict_reward(self, state: Tensor, action: Tensor) -> Tensor:
    # Calculate (raw) uncertainty cost
    uncertainty_cost = self._get_action_uncertainty(state, action)
    # Calculate clipped uncertainty cost
    neg_idxs = uncertainty_cost.less_equal(self.q)
    uncertainty_cost[neg_idxs] = -1
    uncertainty_cost[~neg_idxs] = 1
    return -uncertainty_cost


class Critic(nn.Module):
  def __init__(self, state_size: int, action_size: int, model_cfg: DictConfig):
    super().__init__()
    self.critic = _create_fcnn(state_size + action_size, model_cfg.hidden_size, model_cfg.depth, output_size=1, activation_function=model_cfg.activation)

  def forward(self, state: Tensor, action: Tensor) -> Tensor:
    value = self.critic(_join_state_action(state, action)).squeeze(dim=1)
    return value


class TwinCritic(nn.Module):
  def __init__(self, state_size: int, action_size: int, model_cfg: DictConfig):
    super().__init__()
    self.critic_1 = Critic(state_size, action_size, model_cfg)
    self.critic_2 = Critic(state_size, action_size, model_cfg)

  def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
    value_1, value_2 = self.critic_1(state, action), self.critic_2(state, action)
    return value_1, value_2


class GAILDiscriminator(nn.Module):
  def __init__(self, state_size: int, action_size: int, imitation_cfg: DictConfig, discount):
    super().__init__()
    model_cfg = imitation_cfg.model
    self.discount, self.state_only, self.reward_shaping, self.reward_function = discount, imitation_cfg.state_only, model_cfg.reward_shaping, model_cfg.reward_function
    if self.reward_shaping:
      self.g = nn.Linear(state_size if self.state_only else state_size + action_size, 1)  # Reward function r
      if imitation_cfg.spectral_norm: self.g = parametrizations.spectral_norm(self.g)
      self.h = _create_fcnn(state_size, model_cfg.hidden_size, model_cfg.depth, 1, activation_function=model_cfg.activation, spectral_norm=imitation_cfg.spectral_norm)  # Shaping function Φ
    else:
      self.g = _create_fcnn(state_size if self.state_only else state_size + action_size, model_cfg.hidden_size, model_cfg.depth, 1, activation_function=model_cfg.activation, spectral_norm=imitation_cfg.spectral_norm)

  def _reward(self, state: Tensor, action: Tensor) -> Tensor:
    if self.state_only:
      return self.g(state).squeeze(dim=1)
    else:
      return self.g(_join_state_action(state, action)).squeeze(dim=1)

  def _value(self, state: Tensor) -> Tensor:
    return self.h(state).squeeze(dim=1)

  def forward(self, state: Tensor, action: Tensor, next_state: Optional[Tensor]=None, log_policy: Optional[Tensor]=None, terminal: Optional[Tensor]=None) -> Tensor:
    if self.reward_shaping:
      f = self._reward(state, action) + (1 - terminal) * (self.discount * self._value(next_state) - self._value(state))
      return f - log_policy  # Note that this is equivalent to sigmoid^-1(e^f / (e^f + π))
    else:
      return self.g(state if self.state_only else _join_state_action(state, action)).squeeze(dim=1)
  
  def predict_reward(self, state: Tensor, action: Tensor, next_state: Optional[Tensor]=None, log_policy: Optional[Tensor]=None, terminal: Optional[Tensor]=None) -> Tensor:
    D = torch.sigmoid(self.forward(state, action, next_state=next_state, log_policy=log_policy, terminal=terminal))
    h = -torch.log1p(-D + 1e-6) if self.reward_function == 'GAIL' else torch.log(D + 1e-6) - torch.log1p(-D + 1e-6)  # Add epsilon to improve numerical stability given limited floating point precision
    return torch.exp(h) * -h if self.reward_function == 'FAIRL' else h  # FAIRL reward function is based on AIRL reward function


class GMMILDiscriminator(nn.Module):
  def __init__(self, state_size: int, action_size: int, imitation_cfg: DictConfig):
    super().__init__()
    self.state_only = imitation_cfg.state_only
    self.gamma_1, self.gamma_2, self.self_similarity = None, None, imitation_cfg.self_similarity

  def predict_reward(self, state: Tensor, action: Tensor, expert_state: Tensor, expert_action: Tensor, weight: Tensor, expert_weight: Tensor) -> Tensor:
    state_action = state if self.state_only else _join_state_action(state, action)
    expert_state_action = expert_state if self.state_only else _join_state_action(expert_state, expert_action)
    # Use median heuristics to set data-dependent bandwidths
    if self.gamma_1 is None:
      self.gamma_1 = 1 / (_weighted_median(_squared_distance(state_action, expert_state_action), torch.outer(weight, expert_weight)).item() + 1e-8)
      self.gamma_2 = 1 / (_weighted_median(_squared_distance(expert_state_action, expert_state_action), torch.outer(expert_weight, expert_weight)).item() + 1e-8)  # Add epsilon for numerical stability (if distance is zero)
    # Calculate negative of witness function (based on kernel mean embeddings)
    weight_norm, exp_weight_norm  = weight / weight.sum(), expert_weight / expert_weight.sum()
    s_a_e_s_a_sq_dist = _squared_distance(state_action, expert_state_action)
    similarity = _weighted_similarity(s_a_e_s_a_sq_dist, weight_norm, exp_weight_norm, gamma=self.gamma_1) + _weighted_similarity(s_a_e_s_a_sq_dist, weight_norm, exp_weight_norm, gamma=self.gamma_2)
    if self.self_similarity:
      s_a_s_a_sq_dist = _squared_distance(state_action, state_action)
      self_similarity = _weighted_similarity(s_a_s_a_sq_dist, weight_norm, weight_norm, gamma=self.gamma_1) + _weighted_similarity(s_a_s_a_sq_dist, weight_norm, weight_norm, gamma=self.gamma_2)
    return similarity - self_similarity if self.self_similarity else similarity


class EmbeddingNetwork(nn.Module):
  def __init__(self, input_size: int, model_cfg: DictConfig):
    super().__init__()
    self.embedding = _create_fcnn(input_size, model_cfg.hidden_size, model_cfg.depth, input_size, model_cfg.activation)

  def forward(self, input: Tensor) -> Tensor:
    return self.embedding(input)


class REDDiscriminator(nn.Module):
  def __init__(self, state_size: int, action_size: int, imitation_cfg: DictConfig):
    super().__init__()
    self.state_only = imitation_cfg.state_only
    self.sigma_1 = None
    self.predictor = EmbeddingNetwork(state_size if self.state_only else state_size + action_size, imitation_cfg.model)
    self.target = EmbeddingNetwork(state_size if self.state_only else state_size + action_size, imitation_cfg.model)
    for param in self.target.parameters():
      param.requires_grad = False

  def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
    state_action = state if self.state_only else _join_state_action(state, action)
    prediction, target = self.predictor(state_action), self.target(state_action)
    return prediction, target

  # Originally, sets σ based such that r(s, a) from expert demonstrations ≈ 1; instead this uses kernel median heuristic (same as GMMIL)
  def set_sigma(self, expert_state: Tensor, expert_action: Tensor):
    prediction, target = self.forward(expert_state, expert_action)
    self.sigma_1 = 1 / _squared_distance(prediction, target).median().item()

  def predict_reward(self, state: Tensor, action: Tensor) -> Tensor:
    prediction, target = self.forward(state, action)
    return torch.exp(-self.sigma_1 * (prediction - target).pow(2).mean(dim=1))
