from logging import ERROR

import d4rl_pybullet
import gym
import numpy as np
import torch

from training import ReplayMemory

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger

D4RL_ENV_NAMES = ['ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0', 'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0']


class D4RLEnv():
  def __init__(self, env_name, absorbing):
    assert env_name in D4RL_ENV_NAMES
    
    self.env = gym.make(env_name)
    self.dataset = self.env.get_dataset()  # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
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
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state; if terminal replay memory will overwrite with absorbing state
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

  def get_dataset(self, size=-1, subsample=20):
    N = self.dataset['rewards'].shape[0]
    dataset_out = {'states': torch.as_tensor(self.dataset['observations'][:-1], dtype=torch.float32),
                   'actions': torch.as_tensor(self.dataset['actions'][:-1], dtype=torch.float32),
                   'rewards': torch.as_tensor(self.dataset['rewards'][:-1], dtype=torch.float32),
                   'next_states': torch.as_tensor(self.dataset['observations'][1:], dtype=torch.float32), 
                   'terminals': torch.as_tensor(self.dataset['terminals'][:-1], dtype=torch.float32)}
    # Postprocess
    if size > -1 and size < N:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][0:size]
    if self.absorbing:  # Wrap for absorbing states; note that subsampling after removes need for importance weighting (https://openreview.net/forum?id=Hk4fpoA5Km)
      absorbing_state, absorbing_action = torch.cat([torch.zeros(dataset_out['states'].shape[1]), torch.ones(1)]), torch.zeros(dataset_out['actions'].shape[1])  # Create absorbing state and absorbing action
      # Append absorbing indicator (zero)
      dataset_out['states'] = torch.cat([dataset_out['states'], torch.zeros(dataset_out['states'].size(0), 1)], dim=1)
      dataset_out['next_states'] = torch.cat([dataset_out['next_states'], torch.zeros(dataset_out['states'].size(0), 1)], dim=1)
      # Loop over episodes
      terminal_idxs = np.concatenate([[[0]], dataset_out['terminals'].nonzero()], axis=0).squeeze()
      states, actions, rewards, next_states = [], [], [], []
      for i in range(len(terminal_idxs) - 1):  # TODO: Apply only if terminal state was not caused by a time limit?
        # Extract all transitions within an episode
        states.append(dataset_out['states'][terminal_idxs[i]:terminal_idxs[i + 1]])
        actions.append(dataset_out['actions'][terminal_idxs[i]:terminal_idxs[i + 1]])
        next_states.append(dataset_out['next_states'][terminal_idxs[i]:terminal_idxs[i + 1]])
        rewards.append(dataset_out['rewards'][terminal_idxs[i]:terminal_idxs[i + 1]])
        # Replace the final next state with the absorbing state
        next_states[-1][-1] = absorbing_state
        # Add absorbing state to absorbing state transition
        states.append(absorbing_state.unsqueeze(dim=0))
        actions.append(absorbing_action.unsqueeze(dim=0))
        next_states.append(absorbing_state.unsqueeze(dim=0))
        rewards.append(torch.zeros(1))
      # Recreate memory with wrapped episodes
      dataset_out['states'], dataset_out['actions'], dataset_out['rewards'], dataset_out['next_states'] = torch.cat(states, dim=0), torch.cat(actions, dim=0), torch.cat(rewards, dim=0), torch.cat(next_states, dim=0)
      dataset_out['terminals'] = torch.zeros_like(dataset_out['rewards'])
    if subsample > 0:
      for key in dataset_out.keys():
        dataset_out[key] = dataset_out[key][np.random.choice(subsample)::subsample]  # Subsample from random index in 0 to N-1 (procedure from original GAIL implementation)


    return ReplayMemory(dataset_out['states'].size(0), dataset_out['states'].size(1), dataset_out['actions'].size(1), transitions=dataset_out)


ENVS = {'ant': D4RLEnv, 'halfcheetah': D4RLEnv, 'hopper': D4RLEnv, 'walker2d': D4RLEnv}
