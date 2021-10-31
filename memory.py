import numpy as np
import torch
from torch.utils.data import Dataset


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
