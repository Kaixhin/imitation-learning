from typing import Dict, Tuple

from omegaconf import DictConfig
import torch
from torch import Tensor, autograd
from torch.distributions import Beta, Bernoulli
from torch.optim import Optimizer
from torch.nn import functional as F

from models import GAILDiscriminator, REDDiscriminator, SoftActor, TwinCritic, make_gail_input, update_target_network


# Performs one SAC update
def sac_update(actor: SoftActor, critic: TwinCritic, log_alpha: Tensor, target_critic, transitions: Dict[str, Tensor], actor_optimiser: Optimizer, critic_optimiser: Optimizer, temperature_optimiser: Optimizer, discount: float, entropy_target: float, polyak_factor: float) -> Tuple[Tensor, Tensor]:
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
  actor_optimiser.step()

  # Compute temperature loss
  temperature_loss = -(weights * (1 - absorbing) * alpha * (new_log_probs.detach() + entropy_target)).mean()  # Do not update temperature on absorbing states (no control)
  # Update temperature
  temperature_optimiser.zero_grad(set_to_none=True)
  temperature_loss.backward()
  temperature_optimiser.step()

  # Update target critic
  update_target_network(critic, target_critic, polyak_factor)

  return new_log_probs.detach(), torch.min(values_1, values_2).detach()

# Performs a behavioural cloning update
def behavioural_cloning_update(actor: SoftActor, expert_transition: Dict[str, Tensor], actor_optimiser: Optimizer):
  expert_state, expert_action, weight = expert_transition['states'], expert_transition['actions'], expert_transition['weights']
  expert_action = expert_action.clamp(min=-1 + 1e-6, max=1 - 1e-6)  # Clamp expert actions to (-1, 1)

  actor_optimiser.zero_grad(set_to_none=True)
  behavioural_cloning_loss = (weight * -actor.log_prob(expert_state, expert_action)).mean()  # Maximum likelihood objective
  behavioural_cloning_loss.backward()
  actor_optimiser.step()


# Performs a target estimation update
def target_estimation_update(discriminator: REDDiscriminator, expert_transition: Dict[str, Tensor], discriminator_optimiser: Optimizer):
  expert_state, expert_action, weight = expert_transition['states'], expert_transition['actions'], expert_transition['weights']

  discriminator_optimiser.zero_grad(set_to_none=True)
  prediction, target = discriminator(expert_state, expert_action)
  regression_loss = (weight * (prediction - target).pow(2).mean(dim=1)).mean()
  regression_loss.backward()
  discriminator_optimiser.step()


# Creates a convex combination of 2 variables (e.g. state 1, state 2)
def _mix_vars(x_1: Tensor, x_2: Tensor, eps: Tensor) -> Tensor:
  mix = eps.unsqueeze(dim=1) if x_1.ndim == 2 else eps  # Assumes variables are either 1D or 2D
  return mix * x_1 + (1 - mix) * x_2  # Mix variables with broadcasting weights


# Performs an adversarial imitation learning update
def adversarial_imitation_update(actor: SoftActor, discriminator: GAILDiscriminator, transitions: Dict[str, Tensor], expert_transitions: Dict[str, Tensor], discriminator_optimiser: Optimizer, imitation_cfg: DictConfig):
  reward_shaping, subtract_log_policy, loss_function, grad_penalty, mixup_alpha, entropy_bonus, pos_class_prior, nonnegative_margin = imitation_cfg.discriminator.reward_shaping, imitation_cfg.discriminator.subtract_log_policy, imitation_cfg.loss_function, imitation_cfg.grad_penalty, imitation_cfg.mixup_alpha, imitation_cfg.entropy_bonus, imitation_cfg.pos_class_prior, imitation_cfg.nonnegative_margin

  expert_state, expert_action, expert_next_state, expert_terminal, expert_weight = expert_transitions['states'], expert_transitions['actions'], expert_transitions['next_states'], expert_transitions['terminals'], expert_transitions['weights']
  state, action, next_state, terminal, weight = transitions['states'], transitions['actions'], transitions['next_states'], transitions['terminals'], transitions['weights']

  # Discriminator training objective
  discriminator_optimiser.zero_grad(set_to_none=True)
  if loss_function in ['BCE', 'PUGAIL']:
    with torch.no_grad(): policy_input, expert_input = make_gail_input(state, action, next_state, terminal, actor, reward_shaping, subtract_log_policy), make_gail_input(expert_state, expert_action, expert_next_state, expert_terminal, actor, reward_shaping, subtract_log_policy)
    D_policy, D_expert = discriminator(**policy_input), discriminator(**expert_input)

    if loss_function == 'BCE':
      expert_loss = F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert), weight=expert_weight)  # Loss on "real" (expert) data
      policy_loss = F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy), weight=weight)  # Loss on "fake" (policy) data
    else:
      expert_loss = pos_class_prior * F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert), weight=expert_weight)  # Loss on "real" (expert) data
      policy_loss = torch.clamp(pos_class_prior * F.binary_cross_entropy_with_logits(D_expert, torch.zeros_like(D_expert), weight=expert_weight) - F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy), weight=weight), min=-nonnegative_margin)  # Loss on "real" and "unlabelled" (policy) data
    (expert_loss + policy_loss).backward(retain_graph=True)
    entropy_Ds, entropy_weights = [D_expert, D_policy], [expert_weight, weight]
  elif loss_function == 'Mixup':
    batch_size = state.size(0)
    eps = Beta(torch.full((batch_size, ), float(mixup_alpha)), torch.full((batch_size, ), float(mixup_alpha))).sample()  # Sample ε ∼ Beta(α, α)
    mix_state, mix_action, mix_next_state, mix_terminal, mix_weight = _mix_vars(expert_state, state, eps), _mix_vars(expert_action, action, eps), _mix_vars(expert_next_state, next_state, eps), _mix_vars(expert_terminal, terminal, eps), _mix_vars(expert_weight, weight, eps)  # Create convex combination of expert and policy data
    with torch.no_grad(): mix_input = make_gail_input(mix_state, mix_action, mix_next_state, mix_terminal, actor, reward_shaping, subtract_log_policy)
    D_mix = discriminator(**mix_input)

    mix_loss = eps * F.binary_cross_entropy_with_logits(D_mix, torch.ones_like(D_mix), weight=mix_weight, reduction='none') + (1 - eps) * F.binary_cross_entropy_with_logits(D_mix, torch.zeros_like(D_mix), weight=mix_weight, reduction='none') 
    mix_loss.mean(dim=0).backward(retain_graph=True)
    entropy_Ds, entropy_weights = [D_mix], [mix_weight]

  # Gradient penalty
  if grad_penalty > 0:
    eps = torch.rand_like(terminal)  # Sample ε ∼ U(0, 1)
    eps_2d = eps.unsqueeze(dim=1)  # Expand weights for broadcasting
    mix_state, mix_action, mix_next_state, mix_terminal, mix_weight = _mix_vars(expert_state, state, eps), _mix_vars(expert_action, action, eps), _mix_vars(expert_next_state, next_state, eps), _mix_vars(expert_terminal, terminal, eps), _mix_vars(expert_weight, weight, eps)  # Create convex combination of expert and policy data
    mix_state.requires_grad_()
    mix_action.requires_grad_()
    with torch.no_grad(): mix_input = make_gail_input(mix_state, mix_action, mix_next_state, mix_terminal, actor, reward_shaping, subtract_log_policy)
    D_mix = discriminator(**mix_input)
    grads = autograd.grad(D_mix, (mix_state, mix_action), torch.ones_like(D_mix), create_graph=True)  # Calculate gradients wrt inputs (does not accumulate parameter gradients)
    grad_penalty_loss = grad_penalty * mix_weight * sum([grad.norm(2, dim=1) ** 2 for grad in grads])  # Penalise norm of input gradients (assumes 1D inputs)
    grad_penalty_loss.mean(dim=0).backward()

  # Entropy bonus
  if entropy_bonus > 0:
    entropy_bonus_loss = -entropy_bonus * (sum([w * Bernoulli(logits=l).entropy() for l, w in zip(entropy_Ds, entropy_weights)])).mean()  # Maximise entropy
    entropy_bonus_loss.backward()

  discriminator_optimiser.step()
