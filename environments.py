from logging import ERROR

import d4rl_pybullet
import gym
import numpy as np
import torch

from training import ReplayMemory

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger

D4RL_ENV_NAMES = ['ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0', 'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0']


# TODO: Apply only if terminal state was not caused by a time limit?
def wrap_for_absorbing_states(states, actions, rewards, next_states, terminals):
  # Add terminal indicator bit TODO: Implement fully
  states = torch.cat([states, torch.zeros(states.size(0), 1)], dim=1)
  next_states = torch.cat([next_states, torch.zeros(states.size(0), 1)], dim=1)
  return states, actions, rewards, next_states, terminals


class D4RLEnv():
  def __init__(self, env_name, absorbing=False):
    assert env_name in D4RL_ENV_NAMES
    
    self.env = gym.make(env_name)
    self.dataset = self.env.get_dataset() # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

    self.absorbing = absorbing
    if absorbing: self.env.observation_space.shape = (self.env.observation_space.shape[0] + 1, )  # Append absorbing indicator bit to state dimension (assumes 1D state space)

  def reset(self):
    state = self.env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing:
      state = torch.cat([state, torch.full((state.size(0), 1), 0)], dim=1)  # Add absorbing indicator to state
    return state 

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing:
      state = torch.zeros_like(state) if terminal else state  # Create all zero absorbing state for episodic environments
      state = torch.cat([state, torch.full((state.size(0), 1), 1 if terminal else 0)], dim=1)  # Add absorbing indicator to state
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

  def get_dataset(self, size=0, subsample=20):
    N = self.dataset['rewards'].shape[0]
    dataset_out = {'states': torch.as_tensor(self.dataset['observations'][:-1], dtype=torch.float32),
                   'actions': torch.as_tensor(self.dataset['actions'][:-1], dtype=torch.float32),
                   'rewards': torch.as_tensor(self.dataset['rewards'][:-1], dtype=torch.float32),
                   'next_states': torch.as_tensor(self.dataset['observations'][1:], dtype=torch.float32), 
                   'terminals': torch.as_tensor(self.dataset['terminals'][:-1], dtype=torch.float32)}
    # Postprocess
    if size > 0 and size < N:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0:size]
    if subsample > 0:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0::subsample]
    if self.absorbing:
      dataset_out['states'], dataset_out['actions'], dataset_out['rewards'], dataset_out['next_states'], dataset_out['terminals'] = wrap_for_absorbing_states(dataset_out['states'], dataset_out['actions'], dataset_out['rewards'], dataset_out['next_states'], dataset_out['terminals'])

    return ReplayMemory(dataset_out['states'].size(0), dataset_out['states'].size(1), dataset_out['actions'].size(1), transitions=dataset_out)


ENVS = {'ant': D4RLEnv, 'halfcheetah': D4RLEnv, 'hopper': D4RLEnv, 'walker2d': D4RLEnv}
