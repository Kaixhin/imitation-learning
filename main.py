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
from models import AIRLDiscriminator, GAILDiscriminator, GMMILDiscriminator, REDDiscriminator, SoftActor, TwinCritic, create_target_network, sqil_sample
from training import ReplayMemory, adversarial_imitation_update, behavioural_cloning_update, sac_update, target_estimation_update
from utils import flatten_list_dicts, lineplot


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  # Configuration check
  assert cfg.env_type in ENVS.keys()
  assert cfg.algorithm in ['AIRL', 'BC', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED', 'SAC', 'SQIL']
  # General setup
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)

  # Set up environment
  env = ENVS[cfg.env_type](cfg.env_name, cfg.imitation.absorbing)
  env.seed(cfg.seed)
  expert_trajectories = env.get_dataset()  # Load expert trajectories dataset
  state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]
  
  # Set up agent
  actor, critic, log_alpha = SoftActor(state_size, action_size, cfg.model.hidden_size, cfg.model.activation), TwinCritic(state_size, action_size, cfg.model.hidden_size, cfg.model.activation), torch.zeros(1, requires_grad=True)
  target_critic, entropy_target = create_target_network(critic), -action_size  # Entropy target heuristic from SAC paper for continuous action domains
  actor_optimiser, critic_optimiser, temperature_optimiser = optim.AdamW(actor.parameters(), lr=cfg.reinforcement.learning_rate, weight_decay=cfg.reinforcement.weight_decay), optim.Adam(critic.parameters(), lr=cfg.reinforcement.learning_rate, weight_decay=cfg.reinforcement.weight_decay), optim.Adam([log_alpha], lr=cfg.reinforcement.learning_rate, weight_decay=cfg.reinforcement.weight_decay)
  memory = ReplayMemory(cfg.replay.size, state_size, action_size)

  # Set up imitation learning components
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
    if cfg.algorithm == 'AIRL':
      discriminator = AIRLDiscriminator(state_size, action_size, cfg.model.hidden_size, cfg.reinforcement.discount, cfg.model.activation, state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'DRIL':
      discriminator = SoftActor(state_size, action_size, cfg.model.hidden_size, cfg.model.activation, dropout=0.1)
    elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
      discriminator = GAILDiscriminator(state_size, action_size, cfg.model.hidden_size, cfg.model.activation, state_only=cfg.imitation.state_only, forward_kl=cfg.algorithm == 'FAIRL')
    elif cfg.algorithm == 'GMMIL':
      discriminator = GMMILDiscriminator(state_size, action_size, self_similarity=cfg.imitation.self_similarity, state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'RED':
      discriminator = REDDiscriminator(state_size, action_size, cfg.model.hidden_size, cfg.model.activation, state_only=cfg.imitation.state_only)
    if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']:
      discriminator_optimiser = optim.AdamW(discriminator.parameters(), lr=cfg.imitation.learning_rate, weight_decay=cfg.imitation.weight_decay)

  # Metrics
  metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])
  recent_returns = deque(maxlen=cfg.evaluation.average_window)  # Stores most recent evaluation returns

  if cfg.check_time_usage: start_time = time.time()  # Performance tracking
  # Pre-training
  if cfg.algorithm in ['BC', 'DRIL', 'RED']:
    for _ in tqdm(range(cfg.imitation.pretraining_epochs), leave=False):
      if cfg.algorithm == 'BC':
        # Perform behavioural cloning updates offline
        behavioural_cloning_update(actor, expert_trajectories, actor_optimiser, cfg.training.batch_size, max_grad_norm=cfg.reinforcement.max_grad_norm)
      elif cfg.algorithm == 'DRIL':
        # Perform behavioural cloning updates offline on policy ensemble (dropout version)
        behavioural_cloning_update(discriminator, expert_trajectories, discriminator_optimiser, cfg.training.batch_size, max_grad_norm=cfg.reinforcement.max_grad_norm)
        with torch.no_grad():  # TODO: Check why inference mode fails?
          discriminator.set_uncertainty_threshold(expert_trajectories['states'], expert_trajectories['actions'])
      elif cfg.algorithm == 'RED':
        # Train predictor network to match random target network
        target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, cfg.training.batch_size)
        with torch.inference_mode():
          discriminator.set_sigma(expert_trajectories['states'], expert_trajectories['actions'])

    if cfg.check_time_usage:
      metrics['pre_training_time'] = time.time() - start_time
      start_time = time.time()

  # Training
  state, terminal, train_return = env.reset(), False, 0
  pbar = tqdm(range(1, cfg.steps + 1), unit_scale=1, smoothing=0)
  for step in pbar:
    if cfg.algorithm != 'BC':
      # Collect set of transitions by running policy π in the environment
      with torch.inference_mode():
        action = actor(state).sample()
        next_state, reward, terminal = env.step(action)
        train_return += reward
        memory.append(state, action, reward, terminal)
        state = next_state

      # Reset environment and track metrics on episode termination
      if terminal:
        # Store metrics and reset environment
        metrics['train_steps'].append(step)
        metrics['train_returns'].append([train_return])
        pbar.set_description(f'Step: {step} | Return: {train_return}')
        state, train_return = env.reset(), 0

      # Train agent and imitation learning component
      if step >= cfg.training.start and step % cfg.training.interval == 0:
        # Sample a batch of transitions
        transitions, expert_transitions = memory.sample(cfg.training.batch_size), expert_trajectories.sample(cfg.training.batch_size)

        if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED', 'SQIL']:
          # Train discriminator
          if cfg.algorithm in ['AIRL', 'FAIRL', 'GAIL', 'PUGAIL']:
            adversarial_imitation_update(cfg.algorithm, actor, discriminator, transitions, expert_transitions, discriminator_optimiser, cfg.imitation.r1_reg_coeff, cfg.imitation.get('pos_class_prior', 0.5), cfg.imitation.get('nonnegative_margin', 0))
          
          # Predict rewards
          states, actions, next_states, terminals = transitions['states'], transitions['actions'], transitions['next_states'], transitions['terminals']
          with torch.inference_mode():
            if cfg.algorithm == 'AIRL':
              transitions['rewards'] = discriminator.predict_reward(states, actions, next_states, actor.log_prob(states, actions), terminals)
            elif cfg.algorithm == 'DRIL':
              # TODO: By default DRIL also includes behavioural cloning online?
              transitions['rewards'] = discriminator.predict_reward(states, actions)
            elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
              transitions['rewards'] = discriminator.predict_reward(states, actions)
            elif cfg.algorithm == 'GMMIL':
              expert_states, expert_actions = expert_transitions['states'], expert_transitions['actions']  # Note that using the entire dataset is prohibitively slow in off-policy case
              transitions['rewards'] = discriminator.predict_reward(states, actions, expert_states, expert_actions)
            elif cfg.algorithm == 'RED':
              transitions['rewards'] = discriminator.predict_reward(states, actions)
            elif cfg.algorithm == 'SQIL':
              sqil_sample(transitions, expert_transitions, cfg.training.batch_size)  # Rewrites training transitions as a mix of expert and policy data with constant reward functions TODO: Add sampling ratio option?
        
        # Train agent using SAC
        sac_update(actor, critic, log_alpha, target_critic, transitions, actor_optimiser, critic_optimiser, temperature_optimiser, cfg.reinforcement.discount, entropy_target, cfg.reinforcement.polyak_factor, max_grad_norm=cfg.reinforcement.max_grad_norm)
    
    # Evaluate agent and plot metrics
    if step % cfg.evaluation.interval == 0 and not cfg.check_time_usage:
      test_returns = evaluate_agent(actor, cfg.evaluation.episodes, ENVS[cfg.env_type], cfg.env_name, cfg.imitation.absorbing, cfg.seed)
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
    _, trajectories = evaluate_agent(actor, cfg.evaluation.episodes, ENVS[cfg.env_type], cfg.env_name, cfg.imitation.absorbing, cfg.seed, return_trajectories=True, render=cfg.render)
    torch.save(trajectories, 'trajectories.pth')
  # Save agent and metrics
  torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), log_alpha=log_alpha), 'agent.pth')
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']: torch.save(discriminator.state_dict(), 'discriminator.pth')
  torch.save(metrics, 'metrics.pth')

  env.close()
  return sum(recent_returns) / float(1 if cfg.algorithm == 'BC' else cfg.evaluation.average_window)


if __name__ == '__main__':
  main()
