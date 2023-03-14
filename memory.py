from __future__ import annotations
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


# Replay memory returns transition tuples of the form (s, a, r, s', terminal, timeout, w)
class ReplayMemory(Dataset):
  def __init__(self, size: int, state_size: int, action_size: int, absorbing: bool, transitions: Optional[Dict[str, Union[Tensor, int]]]=None):
    super().__init__()
    self.size, self.num_trajectories, self.idx, self.full = size, 0, 0, False
    self.absorbing = absorbing
    self.step, self.states, self.actions, self.rewards, self.next_states, self.terminals, self.timeouts, self.weights = torch.empty(size), torch.empty(size, state_size), torch.empty(size, action_size), torch.empty(size), torch.empty(size, state_size), torch.empty(size), torch.empty(size), torch.empty(size)
    if transitions is not None:
      trans_size = min(transitions['states'].size(0), size)  # Take data up to size of replay
      self.step[:trans_size], self.states[:trans_size], self.actions[:trans_size], self.rewards[:trans_size], self.next_states[:trans_size], self.terminals[:trans_size], self.timeouts[:trans_size], self.weights[:trans_size] = torch.arange(1, size + 1, dtype=torch.float32), transitions['states'], transitions['actions'], transitions['rewards'], transitions['next_states'], transitions['terminals'], transitions['timeouts'], transitions['weights']
      self.num_trajectories = transitions['num_trajectories']  # Note that this assumes all trajectories fit into this memory!
      self.idx = trans_size % self.size
      self.full = self.idx == 0 and trans_size > 0  # Replay is full if index has wrapped around (but not if there was no data)

  # Allows string-based access for entire data of one type, or int-based access for single transition
  def __getitem__(self, idx: Union[int, str]) -> Union[Dict[str, Tensor], Tensor]:
    if isinstance(idx, str):
      if idx == 'states':
        return self.states
      elif idx == 'actions':
        return self.actions
      elif idx == 'terminals':
        return self.terminals
    else:
      return dict(step=self.step[idx], states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.next_states[idx], terminals=self.terminals[idx], timeouts=self.timeouts[idx], weights=self.weights[idx])

  def __len__(self) -> int:
    return self.terminals.size(0)

  def append(self, step: int, state: Tensor, action: Tensor, reward: float, next_state: Tensor, terminal: bool, timeout: bool):
    self.step[self.idx], self.states[self.idx], self.actions[self.idx], self.rewards[self.idx], self.next_states[self.idx], self.terminals[self.idx], self.timeouts[self.idx], self.weights[self.idx] = step, state, action, reward, next_state, terminal, timeout, 1
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    if terminal or timeout: self.num_trajectories += 1

  def transfer_transitions(self, memory: ReplayMemory):
    for transition in tqdm(memory, leave=False):
      self.append(*[val for key, val in transition.items() if key != 'weights'])

  # Returns a uniformly sampled valid transition index
  def _sample_idx(self) -> int:
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - 1)
      valid_idx = idx != (self.idx - 1) % self.size  # Make sure data does not cross the memory index
    return idx

  def sample(self, n: int) -> Dict[str, Tensor]:
    idxs = [self._sample_idx() for _ in range(n)]
    transitions = [self[idx] for idx in idxs]
    transitions = {key: torch.stack([t[key] for t in transitions]) for key in transitions[0].keys()}  # Note that stack creates new memory so mix_policy_expert_transitions does not overwrite original data
    transitions['absorbing'] = transitions['states'][:, -1] if self.absorbing else torch.zeros_like(transitions['terminals'])  # Indicate absorbing states if absorbing env
    return transitions

  def wrap_for_absorbing_states(self):
    absorbing_state = torch.cat([torch.zeros(self.states.size(1) - 1), torch.ones(1)], dim=0)
    self.next_states[(self.idx - 1) % self.size], self.terminals[(self.idx - 1) % self.size] = absorbing_state, False  # Replace terminal state with absorbing state and remove terminal
    self.append(self.step[(self.idx - 1) % self.size], absorbing_state, torch.zeros(self.actions.size(1)), 0, absorbing_state, False, False)  # Add absorbing state pair as next transition
