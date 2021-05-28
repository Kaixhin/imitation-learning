import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


# Indicate absorbing states
def indicate_absorbing(states, actions, terminals, next_states=None):
  absorbing_idxs = terminals.to(dtype=torch.bool)
  abs_states = torch.cat([states, torch.zeros(states.size(0), 1)], axis=1)
  abs_states[absorbing_idxs] = 0
  abs_states[absorbing_idxs, -1] = 1
  abs_actions = actions.clone()
  abs_actions[absorbing_idxs] = 0
  if next_states is not None:
    abs_next_states = torch.cat([next_states, torch.zeros(next_states.size(0), 1)], axis=1)
    return abs_states, abs_actions, abs_next_states
  else:
    return abs_states, abs_actions


# Dataset that returns transition tuples of the form (s, a, r, s', terminal)
class TransitionDataset(Dataset):
  def __init__(self, transitions):
    super().__init__()
    self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['rewards'], transitions['terminals']  # Detach actions

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
      return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.states[idx + 1], terminals=self.terminals[idx])

  def __len__(self):
    return self.terminals.size(0) - 1  # Need to return state and next state


# Computes and stores generalised advantage estimates ψ in the set of trajectories
def compute_advantages_(trajectories, next_value, discount, trace_decay):
  reward_to_go, advantage = torch.zeros(1), torch.zeros(1)
  trajectories['rewards_to_go'], trajectories['advantages'] = torch.empty_like(trajectories['rewards']), torch.empty_like(trajectories['rewards'])
  for t in reversed(range(trajectories['states'].size(0))):
    reward_to_go = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * (discount * reward_to_go)  # Reward-to-go/value R
    trajectories['rewards_to_go'][t] = reward_to_go
    td_error = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * discount * next_value - trajectories['values'][t]  # TD-error δ
    advantage = td_error + (1 - trajectories['terminals'][t]) * discount * trace_decay * advantage  # Generalised advantage estimate ψ
    trajectories['advantages'][t] = advantage
    next_value = trajectories['values'][t]
  # Normalise the advantages
  trajectories['advantages'] = (trajectories['advantages'] - trajectories['advantages'].mean()) / (trajectories['advantages'].std() + 1e-8)


# Performs one PPO update (includes GAE re-estimation)
def ppo_update(agent, trajectories, next_state, agent_optimiser, discount, trace_decay, ppo_clip, value_loss_coeff=1, entropy_loss_coeff=1, max_grad_norm=1):
  policy, trajectories['values'] = agent(trajectories['states'])
  trajectories['log_prob_actions'] = policy.log_prob(trajectories['actions'])
  with torch.no_grad():  # Do not differentiate through advantage calculation
    next_value = agent(next_state)[1]
    compute_advantages_(trajectories, next_value, discount, trace_decay)  # Recompute rewards-to-go R and generalised advantage estimates ψ based on the current value function V

  policy_ratio = (trajectories['log_prob_actions'] - trajectories['old_log_prob_actions']).exp()
  policy_loss = -torch.min(policy_ratio * trajectories['advantages'], torch.clamp(policy_ratio, min=1 - ppo_clip, max=1 + ppo_clip) * trajectories['advantages']).mean()  # Update the policy by maximising the clipped PPO objective
  value_loss = F.mse_loss(trajectories['values'], trajectories['rewards_to_go'])  # Fit value function by regression on mean squared error
  entropy_loss = -policy.entropy().mean()  # Add entropy regularisation
  
  agent_optimiser.zero_grad(set_to_none=True)
  (policy_loss + value_loss_coeff * value_loss + entropy_loss_coeff * entropy_loss).backward()
  clip_grad_norm_(agent.parameters(), max_grad_norm)  # Clamp norm of gradients
  agent_optimiser.step()


# Performs a behavioural cloning update
def behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, batch_size):
  expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

  for expert_transition in expert_dataloader:
    expert_state, expert_action = expert_transition['states'], expert_transition['actions']

    agent_optimiser.zero_grad(set_to_none=True)
    behavioural_cloning_loss = -agent.log_prob(expert_state, expert_action).mean()  # Maximum likelihood objective
    behavioural_cloning_loss.backward()
    agent_optimiser.step()


# Performs a target estimation update
def target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, batch_size, absorbing):
  expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

  for expert_transition in expert_dataloader:
    expert_state, expert_action = expert_transition['states'], expert_transition['actions']
    if absorbing: expert_state, expert_action = indicate_absorbing(expert_state, expert_action, expert_transition['terminals'])

    discriminator_optimiser.zero_grad(set_to_none=True)
    prediction, target = discriminator(expert_state, expert_action)
    regression_loss = F.mse_loss(prediction, target)
    regression_loss.backward()
    discriminator_optimiser.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(algorithm, agent, discriminator, expert_trajectories, policy_trajectories, discriminator_optimiser, batch_size, absorbing=False, r1_reg_coeff=1, pos_class_prior=1, nonnegative_margin=0):
  expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
  policy_dataloader = DataLoader(policy_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

  # Iterate over mininum of expert and policy data
  for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
    expert_state, expert_action, expert_next_state, expert_terminal = expert_transition['states'], expert_transition['actions'], expert_transition['next_states'], expert_transition['terminals']
    policy_state, policy_action, policy_next_state, policy_terminal = policy_transition['states'], policy_transition['actions'], policy_transition['next_states'], policy_transition['terminals']

    if algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
      if absorbing: expert_state, expert_action, policy_state, policy_action = *indicate_absorbing(expert_state, expert_action, expert_terminal), *indicate_absorbing(policy_state, policy_action, policy_terminal)
      D_expert = discriminator(expert_state, expert_action)
      D_policy = discriminator(policy_state, policy_action)
    elif algorithm == 'AIRL':
      with torch.no_grad():
        expert_data_log_policy = agent.log_prob(expert_state, expert_action)
        policy_data_log_policy = agent.log_prob(policy_state, policy_action)
      if absorbing: expert_state, expert_action, expert_next_state, policy_state, policy_action, policy_next_state = *indicate_absorbing(expert_state, expert_action, expert_terminal, expert_next_state), *indicate_absorbing(policy_state, policy_action, policy_terminal, policy_next_state)
      D_expert = discriminator(expert_state, expert_action, expert_next_state, expert_data_log_policy, expert_terminal)
      D_policy = discriminator(policy_state, expert_action, policy_next_state, policy_data_log_policy, policy_terminal)
 
    # Binary logistic regression
    discriminator_optimiser.zero_grad(set_to_none=True)
    expert_loss = (pos_class_prior if algorithm == 'PUGAIL' else 1) * F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert))  # Loss on "real" (expert) data
    autograd.backward(expert_loss, create_graph=True)
    r1_reg = 0
    for param in discriminator.parameters():
      r1_reg += param.grad.norm()  # R1 gradient penalty
    if algorithm == 'PUGAIL':
      policy_loss = torch.clamp(F.binary_cross_entropy_with_logits(D_expert, torch.zeros_like(D_expert)) - pos_class_prior * F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy)), min=-nonnegative_margin)  # Loss on "real" and "unlabelled" (policy) data
    else:
      policy_loss = F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy))  # Loss on "fake" (policy) data
    (policy_loss + r1_reg_coeff * r1_reg).backward()
    discriminator_optimiser.step()
