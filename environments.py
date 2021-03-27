from logging import ERROR

import gym
import torch
import d4rl_pybullet

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


class CartPoleEnv():
  def __init__(self):
    self.env = gym.make('CartPole-v1')

  def reset(self):
    state = self.env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

  def step(self, action):
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

  def seed(self, seed):
    return self.env.seed(seed)

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space


class D4RLEnv():
  def __init__(self, envname):
    self.env = gym.make(envname)

  def reset(self):
    state = self.env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

  def step(self, action):
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal  # Add batch dimension to state

  def seed(self, seed):
    return self.env.seed(seed)

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, dtype=torch.float):
    """ Also adds next_state in the dataset"""
    dataset = self.env.get_dataset()
    N = dataset['rewards'].shape[0]
    dataset_out = dict()
    dataset_out['rewards'] = torch.as_tensor(dataset['rewards'], dtype=dtype)
    dataset_out['observations'] = torch.as_tensor(dataset['observations'], dtype=dtype)
    
    dataset_out['actions'] = torch.as_tensor(dataset['actions'], dtype=dtype)


    return self.env.get_dataset()
