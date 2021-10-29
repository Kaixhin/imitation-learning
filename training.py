import numpy as np
import torch
from torch import autograd
from torch.distributions import Beta
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from models import update_target_network


# Replay memory returns transition tuples of the form (s, a, r, s', terminal)
class ReplayMemory(Dataset):
  def __init__(self, size, state_size, action_size, absorbing, transitions=None):
    super().__init__()
    self.size, self.idx, self.full = size, 0, False
    self.absorbing = absorbing
    self.states, self.actions, self.rewards, self.next_states, self.terminals, self.weights = torch.empty(size, state_size), torch.empty(size, action_size), torch.empty(size), torch.empty(size, state_size), torch.empty(size), torch.empty(size)
    if transitions is not None:
      trans_size = min(transitions['states'].size(0), size)  # Take data up to size of replay
      self.states[:trans_size], self.actions[:trans_size], self.rewards[:trans_size], self.next_states[:trans_size], self.terminals[:trans_size], self.weights[:trans_size] = transitions['states'], transitions['actions'], transitions['rewards'], transitions['next_states'], transitions['terminals'], transitions['weights']
      self.idx = trans_size % self.size
      self.full = self.idx == 0 and trans_size > 0  # Replay is full if index has wrapped around (but not if there was no data)

  # Allows string-based access for entire data of one type, or int-based access for single transition
  def __getitem__(self, idx):
    if isinstance(idx, str):
      if idx == 'states':
        return self.states
      elif idx == 'actions':
        return self.actions
      elif idx == 'terminals':
        return self.terminals
    else:
      return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.next_states[idx], terminals=self.terminals[idx], weights=self.weights[idx])

  def __len__(self):
    return self.terminals.size(0)

  def append(self, state, action, reward, next_state, terminal):
    self.states[self.idx], self.actions[self.idx], self.rewards[self.idx], self.next_states[self.idx], self.terminals[self.idx], self.weights[self.idx] = state, action, reward, next_state, terminal, 1
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0

  # Returns a uniformly sampled valid transition index
  def _sample_idx(self):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - 1)
      valid_idx = idx != (self.idx - 1) % self.size  # Make sure data does not cross the memory index
    return idx

  def sample(self, n):
    idxs = [self._sample_idx() for _ in range(n)]
    transitions = [self[idx] for idx in idxs]
    transitions = dict(states=torch.stack([t['states'] for t in transitions]), actions=torch.stack([t['actions'] for t in transitions]), rewards=torch.stack([t['rewards'] for t in transitions]), next_states=torch.stack([t['next_states'] for t in transitions]), terminals=torch.stack([t['terminals'] for t in transitions]), weights=torch.stack([t['weights'] for t in transitions]))  # Note that stack creates new memory so SQIL does not overwrite original data
    transitions['absorbing'] = transitions['states'][:, -1] if self.absorbing else torch.zeros_like(transitions['terminals'])  # Indicate absorbing states if absorbing env
    return transitions

  def wrap_for_absorbing_states(self):
    absorbing_state = torch.cat([torch.zeros(self.states.size(1) - 1), torch.ones(1)], dim=0)
    self.next_states[(self.idx - 1) % self.size], self.terminals[(self.idx - 1) % self.size] = absorbing_state, False  # Replace terminal state with absorbing state and remove terminal
    self.append(absorbing_state, torch.zeros(self.actions.size(1)), 0, absorbing_state, False)  # Add absorbing state pair as next transition


# Performs one SAC update
def sac_update(actor, critic, log_alpha, target_critic, transitions, actor_optimiser, critic_optimiser, temperature_optimiser, discount, entropy_target, polyak_factor, max_grad_norm=0):
  states, actions, rewards, next_states, terminals, weights, absorbing = transitions['states'], transitions['actions'], transitions['rewards'], transitions['next_states'], transitions['terminals'], transitions['weights'], transitions['absorbing']
  alpha = log_alpha.exp()
  
  # Compute value function loss
  with torch.no_grad():
    new_next_policies = actor(next_states)
    new_next_actions = new_next_policies.sample()
    new_next_log_probs = new_next_policies.log_prob(new_next_actions)  # Log prob calculated before absorbing state rewrite; these are masked out of target values, but tends to result in NaNs as the policy might be strange over the all-zeros "absorbing action", and NaNs propagate into the target values, so we just avoid it in the first place
    new_next_actions = (1 - absorbing.unsqueeze(dim=1)) * new_next_actions  # If current state is absorbing, manually overwrite with absorbing state action
    target_values = torch.min(*target_critic(next_states, new_next_actions)) - (1 - absorbing) * alpha * new_next_log_probs  # Agent has no control at absorbing state, therefore do not maximise entropy on these
    target_values = rewards + (1 - terminals) * discount * target_values
  values_1, values_2 = critic(states, actions)
  value_loss = (weights * (values_1 - target_values).pow(2)).mean() + (weights * (values_2 - target_values).pow(2)).mean()
  # Update critic
  critic_optimiser.zero_grad(set_to_none=True)
  value_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(critic.parameters(), max_grad_norm)
  critic_optimiser.step()

  # Compute policy loss
  new_policies = actor(states)
  new_actions = new_policies.rsample()
  new_log_probs = new_policies.log_prob(new_actions)
  new_values = torch.min(*critic(states, new_actions))
  policy_loss = (weights * (1 - absorbing) * alpha.detach() * new_log_probs - new_values).mean()  # Do not update actor on absorbing states (no control)
  # Update actor
  actor_optimiser.zero_grad(set_to_none=True)
  policy_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(actor.parameters(), max_grad_norm)
  actor_optimiser.step()

  # Compute temperature loss
  temperature_loss = -(weights * (1 - absorbing) * alpha * (new_log_probs.detach() + entropy_target)).mean()  # Do not update temperature on absorbing states (no control)
  # Update temperature
  temperature_optimiser.zero_grad(set_to_none=True)
  temperature_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(log_alpha, max_grad_norm)
  temperature_optimiser.step()

  # Update target critic
  update_target_network(critic, target_critic, polyak_factor)


# Performs a behavioural cloning update
def behavioural_cloning_update(actor, expert_transition, actor_optimiser, max_grad_norm=0):
  expert_state, expert_action, weight = expert_transition['states'], expert_transition['actions'], expert_transition['weights']
  expert_action = expert_action.clamp(min=-1 + 1e-6, max=1 - 1e-6)  # Clamp expert actions to (-1, 1)

  actor_optimiser.zero_grad(set_to_none=True)
  behavioural_cloning_loss = (weight * -actor.log_prob(expert_state, expert_action)).mean()  # Maximum likelihood objective
  behavioural_cloning_loss.backward()
  if max_grad_norm > 0: clip_grad_norm_(actor.parameters(), max_grad_norm)
  actor_optimiser.step()


# Performs a target estimation update
def target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, batch_size):
  expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

  for expert_transition in expert_dataloader:
    expert_state, expert_action, weight = expert_transition['states'], expert_transition['actions'], expert_transition['weights']

    discriminator_optimiser.zero_grad(set_to_none=True)
    prediction, target = discriminator(expert_state, expert_action)
    regression_loss = (weight * (prediction - target).pow(2).mean(dim=1)).mean()
    regression_loss.backward()
    discriminator_optimiser.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(algorithm, actor, discriminator, transitions, expert_transitions, discriminator_optimiser, grad_penalty=1, mixup_alpha=0, pos_class_prior=1, nonnegative_margin=0):
  expert_state, expert_action, expert_next_state, expert_terminal, expert_weight = expert_transitions['states'], expert_transitions['actions'], expert_transitions['next_states'], expert_transitions['terminals'], expert_transitions['weights']
  state, action, next_state, terminal, weight = transitions['states'], transitions['actions'], transitions['next_states'], transitions['terminals'], transitions['weights']

  if algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
    D_expert = discriminator(expert_state, expert_action)
    D_policy = discriminator(state, action)
  elif algorithm == 'AIRL':
    with torch.no_grad():
      expert_log_prob = actor.log_prob(expert_state, expert_action)
      log_prob = actor.log_prob(state, action)
    D_expert = discriminator(expert_state, expert_action, expert_next_state, expert_log_prob, expert_terminal)
    D_policy = discriminator(state, action, next_state, log_prob, terminal)

  # Binary logistic regression
  discriminator_optimiser.zero_grad(set_to_none=True)
  if mixup_alpha > 0:
    batch_size = state.size(0)
    eps = Beta(torch.full((batch_size, ), 1.), torch.full((batch_size, ), 1.)).sample()  # Sample ε ∼ Beta(α, α)  # TODO: Make alpha a hyperparam
    eps_2d = eps.unsqueeze(dim=1)  # Expand weights for broadcasting
    mix_state, mix_action, mix_weight = eps_2d * expert_state + (1 - eps_2d) * state, eps_2d * expert_action + (1 - eps_2d) * action, eps * expert_weight + (1 - eps) * weight  # Create convex combination of expert and policy data  # TODO: Adapt for AIRL
    D_mix = discriminator(mix_state, mix_action)
    mix_loss = eps * F.binary_cross_entropy_with_logits(D_mix, torch.ones_like(D_mix), weight=mix_weight, reduction='none') + (1 - eps) * F.binary_cross_entropy_with_logits(D_mix, torch.zeros_like(D_mix), weight=mix_weight, reduction='none') 
    mix_loss.mean(dim=0).backward()
  else:
    expert_loss = (pos_class_prior if algorithm == 'PUGAIL' else 1) * F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert), weight=expert_weight)  # Loss on "real" (expert) data
    expert_loss.backward()

    if algorithm == 'PUGAIL':
      policy_loss = torch.clamp(pos_class_prior * F.binary_cross_entropy_with_logits(D_expert, torch.zeros_like(D_expert), weight=expert_weight) - F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy), weight=weight), min=-nonnegative_margin)  # Loss on "real" and "unlabelled" (policy) data
    else:
      policy_loss = F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy), weight=weight)  # Loss on "fake" (policy) data
    policy_loss.backward()
  
  if grad_penalty > 0:
    eps = torch.rand_like(D_expert)  # Sample ε ∼ U(0, 1)
    eps_2d = eps.unsqueeze(dim=1)  # Expand weights for broadcasting
    mix_state, mix_action, mix_weight = eps_2d * expert_state + (1 - eps_2d) * state, eps_2d * expert_action + (1 - eps_2d) * action, eps * expert_weight + (1 - eps) * weight  # Create convex combination of expert and policy data  # TODO: Adapt for AIRL
    mix_state.requires_grad_()
    mix_action.requires_grad_()
    D_mix = discriminator(mix_state, mix_action)
    grads = autograd.grad(D_mix, (mix_state, mix_action), torch.ones_like(D_mix), create_graph=True)  # Calculate gradients wrt inputs (does not accumulate parameter gradients)
    grad_penalty_loss = grad_penalty * mix_weight * sum([grad.norm(2, dim=1) ** 2 for grad in grads])  # Penalise norm of input gradients (assumes 1D inputs)
    grad_penalty_loss.mean(dim=0).backward()

  discriminator_optimiser.step()
