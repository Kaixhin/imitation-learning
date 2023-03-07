from logging import ERROR
from typing import List, Tuple

import d4rl
import gym
from gym.spaces import Box, Space
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from memory import ReplayMemory

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


class D4RLEnv():
  def __init__(self, env_name: str, absorbing: bool, load_data: bool=False):
    self.env = gym.make(f'{env_name}-expert-v2')
    if load_data: self.dataset = self.env.get_dataset()  # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

    self.absorbing = absorbing
    if absorbing: self.env.observation_space = Box(low=np.concatenate([self.env.observation_space.low, np.zeros(1)]), high=np.concatenate([self.env.observation_space.high, np.ones(1)]))  # Append absorbing indicator bit to state dimension (assumes 1D state space)

  def reset(self) -> Tensor:
    state = self.env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state
    return state 

  def step(self, action: Tensor) -> Tuple[Tensor, float, bool]:
    action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
    if self.absorbing: state = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
    return state, reward, terminal

  def seed(self, seed: int) -> List[int]:
    return self.env.seed(seed)

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @property
  def max_episode_steps(self) -> int:
    return self.env._max_episode_steps

  def get_dataset(self, trajectories: int=0, subsample: int=1) -> ReplayMemory:
    # Extract data
    states = torch.as_tensor(self.dataset['observations'], dtype=torch.float32)
    actions = torch.as_tensor(self.dataset['actions'], dtype=torch.float32)
    next_states = torch.as_tensor(self.dataset['next_observations'], dtype=torch.float32)
    terminals = torch.as_tensor(self.dataset['terminals'], dtype=torch.float32)
    timeouts = torch.as_tensor(self.dataset['timeouts'], dtype=torch.float32)
    state_size, action_size = states.size(1), actions.size(1)
    # Split into separate trajectories
    states_list, actions_list, next_states_list, terminals_list, weights_list, timeouts_list = [], [], [], [], [], []
    terminal_idxs, timeout_idxs = terminals.nonzero().flatten(), timeouts.nonzero().flatten()
    ep_end_idxs = torch.sort(torch.cat([torch.tensor([-1]), terminal_idxs, timeout_idxs], dim=0))[0]
    for i in range(len(ep_end_idxs) - 1):
      states_list.append(states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      actions_list.append(actions[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      next_states_list.append(next_states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
      terminals_list.append(terminals[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Only store true terminations; timeouts should not be treated as such
      timeouts_list.append(timeouts[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Store if episode terminated due to timeout
      weights_list.append(torch.ones_like(terminals_list[-1]))  # Add an importance weight of 1 to every transition
    # Pick number of trajectories
    if trajectories > 0:
      states_list = states_list[:trajectories]
      actions_list = actions_list[:trajectories]
      next_states_list = next_states_list[:trajectories]
      terminals_list = terminals_list[:trajectories]
      timeouts_list = timeouts_list[:trajectories]
      weights_list = weights_list[:trajectories]
    num_trajectories = len(states_list)
    # Wrap for absorbing states TODO: Fix
    if self.absorbing:
      absorbing_state, absorbing_action = torch.cat([torch.zeros(1, state_size), torch.ones(1, 1)], dim=1), torch.zeros(1, action_size)  # Create absorbing state and absorbing action
      for i in range(len(states_list)):
        # Append absorbing indicator (zero)
        states_list[i] = torch.cat([states_list[i], torch.zeros(states_list[i].size(0), 1)], dim=1)
        next_states_list[i] = torch.cat([next_states_list[i], torch.zeros(next_states_list[i].size(0), 1)], dim=1)
        if not timeouts_list[i]:  # Apply for episodes that did not terminate due to time limits
          # Replace the final next state with the absorbing state and overwrite terminal status
          next_states_list[i][-1] = absorbing_state
          terminals_list[i][-1] = 0
          weights_list[i][-1] = 1 / subsample  # Importance weight absorbing state as kept during subsampling
          # Add absorbing state to absorbing state transition
          states_list[i] = torch.cat([states_list[i], absorbing_state], dim=0)
          actions_list[i] = torch.cat([actions_list[i], absorbing_action], dim=0)
          next_states_list[i] = torch.cat([next_states_list[i], absorbing_state], dim=0)
          terminals_list[i] = torch.cat([terminals_list[i], torch.zeros(1)], dim=0)
          weights_list[i] = torch.cat([weights_list[i], torch.full((1, ), 1 / subsample)], dim=0)  # Importance weight absorbing state as kept during subsampling
    # Subsample within trajectories
    if subsample > 1:
      for i in range(len(states_list)):
        rand_start_idx, T = np.random.choice(subsample), len(states_list[i])  # Subsample from random index in 0 to N-1 (procedure from original GAIL implementation)
        idxs = range(rand_start_idx, T, subsample)
        if self.absorbing: idxs = sorted(list(set(idxs) | set([T - 2, T - 1])))  # Subsample but keep absorbing state transitions
        states_list[i] = states_list[i][idxs]
        actions_list[i] = actions_list[i][idxs]
        next_states_list[i] = next_states_list[i][idxs]
        terminals_list[i] = terminals_list[i][idxs]
        timeouts_list[i] = timeouts_list[i][idxs]
        weights_list[i] = weights_list[i][idxs]

    transitions = {'states': torch.cat(states_list, dim=0), 'actions': torch.cat(actions_list, dim=0), 'next_states': torch.cat(next_states_list, dim=0), 'terminals': torch.cat(terminals_list, dim=0), 'timeouts': torch.cat(timeouts_list, dim=0), 'weights': torch.cat(weights_list, dim=0), 'num_trajectories': num_trajectories}
    transitions['rewards'] = torch.zeros_like(transitions['terminals'])  # Pass 0 rewards to replay memory for interoperability/make sure reward information is not leaked to IL algorithm when data comes from an offline RL dataset
    return ReplayMemory(transitions['states'].size(0), state_size + (1 if self.absorbing else 0), action_size, self.absorbing, transitions=transitions)


def _get_expert_baseline(env: gym.Env):
  dataset = env.get_dataset()
  rewards, terminals = dataset['rewards'], dataset['terminals'] + dataset['timeouts']  # D4RL separates terminals and timeouts 
  cum_rewards, terminal_idxs = np.cumsum(rewards), np.nonzero(terminals)
  terminal_cum_rewards = cum_rewards[terminal_idxs]
  trajectory_cum_rewards = terminal_cum_rewards - np.concatenate([np.array([0]), terminal_cum_rewards[:-1]])
  mean, std = np.mean(trajectory_cum_rewards), np.std(trajectory_cum_rewards)
  print(f'Expert demonstration returns: {mean} +/- {std}')
  num_episodes=terminal_idxs[0].shape[0]
  return mean, std, num_episodes


def _get_random_agent_baseline(env: gym.Env, num_episodes: int):
  env.seed(0)
  _, terminal, total_reward, returns, step_counter = env.reset(), False, 0, [], 0  # Step counter used for manual timeouts
  pbar = tqdm(range(1, num_episodes + 1), unit_scale=1, smoothing=0)
  for i in pbar:
    while not terminal or step_counter < env._max_episode_steps:
      _, reward, terminal, _ = env.step(env.action_space.sample())
      step_counter += 1
      total_reward += reward
    returns.append(total_reward)
    env.seed(i)
    _, terminal, total_reward, step_counter = env.reset(), False, 0, 0
  returns = np.array(returns)
  mean, std = np.mean(returns), np.std(returns)
  print(f'Random agent returns: {mean} +/- {std}')
  return mean, std


def _get_env_baseline(env: gym.Env):
  random_agent_mean, expert_mean = env.ref_min_score, env.ref_max_score
  return expert_mean, random_agent_mean


def get_all_env_baseline(envs: dict):
  data =  dict()
  for env_name in envs.keys():
    env = gym.make(envs[env_name])  # Skip using D4RL class because action_space.sample() does not exist
    print(f"For env: {env_name} with data: {envs[env_name]}")
    expert_mean, random_agent_mean = _get_env_baseline(env)
    data[env_name] = dict(expert_mean=expert_mean, random_agent_mean=random_agent_mean)
  return data


if __name__ == '__main__':
  import argparse
  supported_envs = dict(ant='ant-expert-v2', hopper='hopper-expert-v2', halfcheetah='halfcheetah-expert-v2', walker2d='walker2d-expert-v2')
  parser = argparse.ArgumentParser(description='Env baselines')
  parser.add_argument('--save-result', action='store_true', default=False)
  parser.add_argument('--env', type=str, default='all')
  args = parser.parse_args()
  assert args.env == 'all' or args.env in supported_envs.keys()

  if args.env == 'all':
    filename = 'normalization_data'
    data = get_all_env_baseline(supported_envs)
    print(data)
    if args.save_result: np.savez(filename, **data)
  else:
    env_name = args.env
    env = gym.make(supported_envs[env_name])  # Skip using D4RL class because action_space.sample() does not exist
    print(f"For env: {env_name} with data: {supported_envs[env_name]}")
    expert_mean, expert_std, random_agent_mean, random_agent_std = _get_env_baseline(env)
    if args.save_result: np.savez(env_name, expert_mean=expert_mean, expert_std=expert_std, random_agent_mean=random_agent_mean, random_agent_std=random_agent_std)
