from collections import deque
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch import optim
from tqdm import tqdm

from environments import ENVS
from evaluation import evaluate_agent
from models import AIRLDiscriminator, GAILDiscriminator, GMMILDiscriminator, REDDiscriminator, SoftActor, TwinCritic, create_target_network
from training import ReplayMemory, adversarial_imitation_update, behavioural_cloning_update, indicate_absorbing, sac_update, target_estimation_update
from utils import flatten_list_dicts, lineplot


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  # Configuration check
  assert cfg.env_type in ENVS.keys()
  assert cfg.imitation in ['AIRL', 'BC', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED', 'SAC']
  # General setup
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)

  # Set up environment
  env = ENVS[cfg.env_type](cfg.env_name)
  env.seed(cfg.seed)
  expert_trajectories = env.get_dataset()  # Load expert trajectories dataset
  state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]
  
  # Set up agent
  actor, critic = SoftActor(state_size, action_size, cfg.hidden_size), TwinCritic(state_size, action_size, cfg.hidden_size)
  target_critic = create_target_network(critic)
  actor_optimiser, critic_optimiser = optim.RMSprop(actor.parameters(), lr=cfg.agent_learning_rate, alpha=0.9), optim.RMSprop(critic.parameters(), lr=cfg.agent_learning_rate, alpha=0.9)  # TODO: actor_learning_rate, critic_learning_rate
  memory = ReplayMemory(int(1e6), state_size, action_size)  # TODO: Make replay size hyperparameter

  # Set up imitation learning components
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
    if cfg.algorithm == 'AIRL':
      discriminator = AIRLDiscriminator(state_size + (1 if cfg.imitation.absorbing else 0), action_size, cfg.model.hidden_size, cfg.reinforcement.discount, state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'DRIL':
      discriminator = Actor(state_size, action_size, cfg.model.hidden_size, dropout=0.1)
    elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
      discriminator = GAILDiscriminator(state_size + (1 if cfg.imitation.absorbing else 0), action_size, cfg.model.hidden_size, state_only=cfg.imitation.state_only, forward_kl=cfg.algorithm == 'FAIRL')
    elif cfg.algorithm == 'GMMIL':
      discriminator = GMMILDiscriminator(state_size + (1 if cfg.imitation.absorbing else 0), action_size, self_similarity=cfg.imitation.self_similarity, state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'RED':
      discriminator = REDDiscriminator(state_size + (1 if cfg.imitation.absorbing else 0), action_size, cfg.model.hidden_size, state_only=cfg.imitation.state_only)
    if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']:
      discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=cfg.imitation.learning_rate)

  # Metrics
  metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])
  recent_returns = deque(maxlen=cfg.evaluation.average_window)  # Stores most recent evaluation returns

  # Main training loop
  state, terminal, train_return = env.reset(), False, 0
  pbar = tqdm(range(1, cfg.steps + 1), unit_scale=1, smoothing=0)
  if cfg.check_time_usage: start_time = time.time()  # Performance tracking
  for step in pbar:
    # Perform initial training (if needed)
    if cfg.algorithm in ['BC', 'DRIL', 'RED']:
      if step == 1:
        for _ in tqdm(range(cfg.imitation.epochs), leave=False):
          if cfg.algorithm == 'BC':
            # Perform behavioural cloning updates offline
            behavioural_cloning_update(actor, expert_trajectories, actor_optimiser, cfg.training.batch_size)
          elif cfg.algorithm == 'DRIL':
            # Perform behavioural cloning updates offline on policy ensemble (dropout version)
            behavioural_cloning_update(discriminator, expert_trajectories, discriminator_optimiser, cfg.training.batch_size)
            with torch.no_grad():  # TODO: Check why inference mode fails?
              discriminator.set_uncertainty_threshold(expert_trajectories['states'], expert_trajectories['actions'])
          elif cfg.algorithm == 'RED':
            # Train predictor network to match random target network
            target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, cfg.training.batch_size, cfg.imitation.absorbing)
            with torch.inference_mode():
              discriminator.set_sigma(expert_trajectories['states'], expert_trajectories['actions'])

        if cfg.check_time_usage:
          metrics['pre_training_time'] = time.time() - start_time
          start_time = time.time()

    if cfg.algorithm != 'BC':
      # Collect set of trajectories by running policy π in the environment
      with torch.inference_mode():
        policy = actor(state)
        action = policy.sample()
        next_state, reward, terminal = env.step(action)
        train_return += reward
        memory.append(dict(states=state, actions=action, rewards=torch.tensor([reward], dtype=torch.float32), next_states=next_state, terminals=torch.tensor([terminal], dtype=torch.float32)))
        state = next_state

      if terminal:
        # Store metrics and reset environment
        metrics['train_steps'].append(step)
        metrics['train_returns'].append([train_return])
        pbar.set_description(f'Step: {step} | Return: {train_return}')
        state, train_return = env.reset(), 0

      # Update models
      if len(trajectories) >= cfg.training.batch_size:
        policy_trajectories = flatten_list_dicts(trajectories)  # Flatten policy trajectories (into a single batch for efficiency; valid for feedforward networks)

        if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
          # Train discriminator
          if cfg.algorithm in ['AIRL', 'FAIRL', 'GAIL', 'PUGAIL']:
            for _ in tqdm(range(cfg.imitation.epochs), leave=False):
              adversarial_imitation_update(cfg.algorithm, actor, discriminator, expert_trajectories, ReplayMemory(policy_trajectory_replays), discriminator_optimiser, cfg.training.batch_size, cfg.imitation.absorbing, cfg.imitation.r1_reg_coeff, cfg.get('pos_class_prior', 0.5), cfg.get('nonnegative_margin', 0))

          # Predict rewards
          states, actions, next_states, terminals = policy_trajectories['states'], policy_trajectories['actions'], torch.cat([policy_trajectories['states'][1:], next_state]), policy_trajectories['terminals']
          if cfg.imitation.absorbing: states, actions, next_states = indicate_absorbing(states, actions, terminals, next_states)
          with torch.inference_mode():
            if cfg.algorithm == 'AIRL':
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions, next_states, policy_trajectories['log_prob_actions'], terminals)
            elif cfg.algorithm == 'DRIL':
              # TODO: By default DRIL also includes behavioural cloning online?
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
            elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
            elif cfg.algorithm == 'GMMIL':
              expert_states, expert_actions = expert_trajectories['states'], expert_trajectories['actions']
              if cfg.imitation.absorbing: expert_states, expert_actions = indicate_absorbing(expert_states, expert_actions, expert_trajectories['terminals'])
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions, expert_states, expert_actions)
            elif cfg.algorithm == 'RED':
              policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)

        # Perform PPO updates (includes GAE re-estimation with updated value function)
        for _ in tqdm(range(cfg.reinforcement.ppo_epochs), leave=False):
          ppo_update(actor, policy_trajectories, next_state, actor_optimiser, cfg.reinforcement.discount, cfg.reinforcement.trace_decay, cfg.reinforcement.ppo_clip, cfg.reinforcement.value_loss_coeff, cfg.reinforcement.entropy_loss_coeff, cfg.reinforcement.max_grad_norm)
        trajectories, policy_trajectories = [], None
    
    
    # Evaluate agent and plot metrics
    if step % cfg.evaluation.interval == 0 and not cfg.check_time_usage:
      test_returns = evaluate_agent(actor, cfg.evaluation.episodes, ENVS[cfg.env_type], cfg.env_name, cfg.seed)
      recent_returns.append(sum(test_returns) / cfg.evaluation.episodes)
      metrics['test_steps'].append(step)
      metrics['test_returns'].append(test_returns)
      lineplot(metrics['test_steps'], metrics['test_returns'], 'test_returns')
      if cfg.algorithm == 'BC':
        lineplot(range(cfg.evaluation.interval, cfg.steps + 1, cfg.evaluation.interval), metrics['test_returns'] * (cfg.steps // cfg.evaluation.interval), 'test_returns')
        break
      elif len(metrics['train_returns']) > 0:  # Plot train returns if any
        lineplot(metrics['train_steps'], metrics['train_returns'], 'train_returns')
    elif cfg.algorithm == 'BC' and cfg.check_time_usage: break

  if cfg.check_time_usage:
    metrics['training_time'] = time.time() - start_time

  if cfg.save_trajectories:
    # Store trajectories from agent after training
    _, trajectories = evaluate_agent(actor, cfg.evaluation.episodes, ENVS[cfg.env_type], cfg.env_name, cfg.seed, return_trajectories=True, render=cfg.render)
    torch.save(trajectories, 'trajectories.pth')
  # Save agent and metrics
  torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict()), 'agent.pth')
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']: torch.save(discriminator.state_dict(), 'discriminator.pth')
  torch.save(metrics, 'metrics.pth')

  env.close()
  return sum(recent_returns) / float(1 if cfg.algorithm == 'BC' else cfg.evaluation.average_window)


if __name__ == '__main__':
  main()
