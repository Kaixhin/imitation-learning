from logging import ERROR

import d4rl
import gym
import numpy as np
import torch

from training import ReplayMemory

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


class D4RLEnv():
  def __init__(self, env_name, absorbing):
    self.env = gym.make(env_name)
    self.dataset = d4rl.qlearning_dataset(self.env)  # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

    self.absorbing = absorbing
    if absorbing: self.env.observation_space.shape = (self.env.observation_space.shape[0] + 1, )  # Append absorbing indicator bit to state dimension (assumes 1D state space)

  def reset(self):
    state = self.env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state
    return state 

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
    return state, reward, terminal

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, trajectories=-1, subsample=20):
    # Extract data
    states = torch.as_tensor(self.dataset['observations'], dtype=torch.float32)
    actions = torch.as_tensor(self.dataset['actions'], dtype=torch.float32)
    next_states = torch.as_tensor(self.dataset['next_observations'], dtype=torch.float32)
    terminals = torch.as_tensor(self.dataset['terminals'], dtype=torch.float32)
    state_size, action_size = states.size(1), actions.size(1)
    # Split into separate trajectories
    states_list, actions_list, next_states_list, terminals_list = [], [], [], []
    terminal_idxs = np.concatenate([[[-1]], terminals.nonzero()], axis=0).squeeze()
    for i in range(len(terminal_idxs) - 1):
      states_list.append(states[terminal_idxs[i] + 1:terminal_idxs[i + 1] + 1])
      actions_list.append(actions[terminal_idxs[i] + 1:terminal_idxs[i + 1] + 1])
      next_states_list.append(next_states[terminal_idxs[i] + 1:terminal_idxs[i + 1] + 1])
      terminals_list.append(terminals[terminal_idxs[i] + 1:terminal_idxs[i + 1] + 1])
    # Pick number of trajectories
    if trajectories > -1:
      states_list = states_list[:trajectories]
      actions_list = actions_list[:trajectories]
      next_states_list = next_states_list[:trajectories]
      terminals_list = terminals_list[:trajectories]
    # Wrap for absorbing states
    if self.absorbing:  
      absorbing_state, absorbing_action = torch.cat([torch.zeros(1, state_size), torch.ones(1, 1)], dim=1), torch.zeros(1, action_size)  # Create absorbing state and absorbing action
      for i in range(len(states_list)):
        # Append absorbing indicator (zero)
        states_list[i] = torch.cat([states_list[i], torch.zeros(states_list[i].size(0), 1)], dim=1)
        next_states_list[i] = torch.cat([next_states_list[i], torch.zeros(next_states_list[i].size(0), 1)], dim=1)
        if True:  # TODO: Apply only if terminal state was not caused by a time limit?
          # Replace the final next state with the absorbing state and overwrite terminal status
          next_states_list[i][-1] = absorbing_state
          terminals_list[i][-1] = 0
          # Add absorbing state to absorbing state transition
          states_list[i] = torch.cat([states_list[i], absorbing_state], dim=0)
          actions_list[i] = torch.cat([actions_list[i], absorbing_action], dim=0)
          next_states_list[i] = torch.cat([next_states_list[i], absorbing_state], dim=0)
          terminals_list[i] = torch.cat([terminals_list[i], torch.zeros(1)], dim=0)
    # Subsample within trajectories
    if subsample > 0:
      for i in range(len(states_list)):
        rand_start_idx, T = np.random.choice(subsample), len(states_list[i])  # Subsample from random index in 0 to N-1 (procedure from original GAIL implementation)
        idxs = range(rand_start_idx, T, subsample)
        if self.absorbing: idxs = sorted(list(set(idxs) | set([T - 2, T - 1])))  # Subsample but keep absorbing state transitions
        states_list[i] = states_list[i][idxs]
        actions_list[i] = actions_list[i][idxs]
        next_states_list[i] = next_states_list[i][idxs]
        terminals_list[i] = terminals_list[i][idxs]

    transitions = {'states': torch.cat(states_list, dim=0), 'actions': torch.cat(actions_list, dim=0), 'next_states': torch.cat(next_states_list, dim=0), 'terminals': torch.cat(terminals_list, dim=0)}  # TODO: Weights?
    transitions['rewards'] = torch.zeros_like(transitions['terminals'])  # Pass 0 rewards to replay memory for interoperability
    return ReplayMemory(transitions['states'].size(0), state_size + (1 if self.absorbing else 0), action_size, transitions=transitions)
