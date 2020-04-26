import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


# Dataset that returns transition tuples of the form (s, a, r, s', terminal)
class TransitionDataset(Dataset):
  def __init__(self, transitions):
    super().__init__()
    self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(
    ), transitions['rewards'], transitions['terminals']

  # Allows string-based access for entire data of one type, or int-based access for single transition
  def __getitem__(self, idx):
    if isinstance(idx, str):
      if idx == 'states':
        return self.states
      elif idx == 'actions':
        return self.actions
    else:
      return self.states[idx], self.actions[idx], self.rewards[idx], self.states[idx + 1], self.terminals[idx]

  def __len__(self):
    return self.terminals.size(0) - 1  # Need to return state and next state


# Computes and stores generalized advantage estimates ψ in the set of trajectories
def compute_advantages(trajectories, next_value, discount, trace_decay):
  with torch.no_grad():  # Do not differentiate through advantage calculation
    reward_to_go, advantage = torch.tensor([0.]), torch.tensor([0.])
    trajectories['rewards_to_go'], trajectories['advantages'] = torch.empty_like(
        trajectories['rewards']), torch.empty_like(trajectories['rewards'])
    for t in reversed(range(trajectories['states'].size(0))):
      reward_to_go = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * (
          discount * reward_to_go)  # Reward-to-go/value R
      trajectories['rewards_to_go'][t] = reward_to_go
      td_error = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * \
          discount * next_value - trajectories['values'][t]  # TD-error δ
      # Generalized advantage estimate ψ
      advantage = td_error + \
          (1 - trajectories['terminals'][t]) * \
          discount * trace_decay * advantage
      trajectories['advantages'][t] = advantage
      next_value = trajectories['values'][t]


# Performs one PPO update (assumes trajectories for first epoch are attached to agent)
def ppo_update(agent, trajectories, agent_optimizer, ppo_clip, epoch, value_loss_coeff=1, entropy_reg_coeff=1):
  # Recalculate outputs for subsequent iterations
  if epoch > 0:
    policy, trajectories['values'] = agent(trajectories['states'])
    trajectories['log_prob_actions'], trajectories['entropies'] = policy.log_prob(
        trajectories['actions'].detach()), policy.entropy()

  policy_ratio = (trajectories['log_prob_actions'] -
                  trajectories['old_log_prob_actions']).exp()
  policy_loss = -torch.min(policy_ratio * trajectories['advantages'], torch.clamp(policy_ratio, min=1 - ppo_clip,
                                                                                  max=1 + ppo_clip) * trajectories['advantages']).mean()  # Update the policy by maximizing the clipped PPO objective
  # Fit value function by regression on mean squared error
  value_loss = F.mse_loss(
      trajectories['values'], trajectories['rewards_to_go'])
  entropy_reg = -trajectories['entropies'].mean()  # Add entropy regularization

  agent_optimizer.zero_grad()
  (policy_loss + value_loss_coeff * value_loss +
   entropy_reg_coeff * entropy_reg).backward()
  clip_grad_norm_(agent.parameters(), 1)  # Clamp norm of gradients
  agent_optimizer.step()


# Performs a behavioral cloning update
def behavioral_cloning_update(agent, expert_trajectories, agent_optimizer, batch_size):
  expert_dataloader = DataLoader(
      expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)

  for expert_transition in expert_dataloader:
    expert_state, expert_action = expert_transition[0], expert_transition[1]

    agent_optimizer.zero_grad()
    # Maximum likelihood objective
    behavioral_cloning_loss = - \
        agent.log_prob(expert_state, expert_action).mean()
    behavioral_cloning_loss.backward()
    agent_optimizer.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(algorithm, agent, discriminator, expert_trajectories, policy_trajectories, discriminator_optimizer, batch_size, r1_reg_coeff=1):
  expert_dataloader = DataLoader(
      expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)
  policy_dataloader = DataLoader(
      policy_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)

  # Iterate over minimum of expert and policy data
  for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
    expert_state, expert_action, expert_next_state, policy_state, policy_action, policy_next_state = expert_transition[
        0], expert_transition[1], expert_transition[3], policy_transition[0], policy_transition[1], policy_transition[3]

    if algorithm == 'GAIL':
      D_expert = discriminator(expert_state, expert_action)
      D_policy = discriminator(policy_state, policy_action)
    elif algorithm == 'AIRL':
      with torch.no_grad():
        expert_data_policy = agent.log_prob(expert_state, expert_action).exp()
        policy_data_policy = agent.log_prob(policy_state, policy_action).exp()
      D_expert = discriminator(
          expert_state, expert_action, expert_next_state, expert_data_policy)
      D_policy = discriminator(
          policy_state, expert_action, policy_next_state, policy_data_policy)

    # Binary logistic regression
    discriminator_optimizer.zero_grad()
    expert_loss = F.binary_cross_entropy(
        D_expert, torch.ones_like(D_expert))  # Loss on "real" (expert) data
    autograd.backward(expert_loss, create_graph=True)
    r1_reg = 0
    for param in discriminator.parameters():
      r1_reg += param.grad.norm().mean()  # R1 gradient penalty
    policy_loss = F.binary_cross_entropy(
        D_policy, torch.zeros_like(D_policy))  # Loss on "fake" (policy) data
    (policy_loss + r1_reg_coeff * r1_reg).backward()
    discriminator_optimizer.step()
