from collections import deque
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from environments import D4RLEnv
from evaluation import evaluate_agent
from models import AIRLDiscriminator, GAILDiscriminator, GMMILDiscriminator, REDDiscriminator, SoftActor, TwinCritic, create_target_network, sqil_sample
from training import ReplayMemory, adversarial_imitation_update, behavioural_cloning_update, sac_update, target_estimation_update
from utils import cycle, flatten_list_dicts, lineplot


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  # Configuration check
  assert cfg.algorithm in ['AIRL', 'BC', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED', 'SAC', 'SQIL']
  cfg.replay.size = min(cfg.steps, cfg.replay.size)  # Set max replay size to min of environment steps and replay size
  assert cfg.imitation.subsample >= 1
  assert not (cfg.algorithm == 'PUGAIL' and cfg.imitation.mixup_alpha > 0)
  # TODO: Check that both PUGAIL and Mixup are not selected at the same time
  # General setup
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)

  # Set up environment
  env = D4RLEnv(cfg.env_name, cfg.imitation.absorbing)
  env.seed(cfg.seed)
  expert_trajectories = env.get_dataset(trajectories=cfg.imitation.trajectories, subsample=cfg.imitation.subsample)  # Load expert trajectories dataset
  state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]
  
  # Set up agent
  actor, critic, log_alpha = SoftActor(state_size, action_size, cfg.reinforcement.model.actor), TwinCritic(state_size, action_size, cfg.reinforcement.model.critic), torch.zeros(1, requires_grad=True)  # TODO: Account for model hidden size and depth
  target_critic, entropy_target = create_target_network(critic), cfg.reinforcement.target_temperature * action_size  # Entropy target heuristic from SAC paper for continuous action domains
  actor_optimiser, critic_optimiser, temperature_optimiser = optim.AdamW(actor.parameters(), lr=cfg.reinforcement.learning_rate, weight_decay=cfg.reinforcement.weight_decay), optim.Adam(critic.parameters(), lr=cfg.reinforcement.learning_rate, weight_decay=cfg.reinforcement.weight_decay), optim.Adam([log_alpha], lr=cfg.reinforcement.learning_rate, weight_decay=cfg.reinforcement.weight_decay)
  memory = ReplayMemory(cfg.replay.size, state_size, action_size, cfg.imitation.absorbing)

  # Set up imitation learning components
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
    if cfg.algorithm == 'AIRL':
      discriminator = AIRLDiscriminator(state_size, action_size, cfg.imitation, cfg.reinforcement.discount)  # TODO: Account for model hidden size and depth
    elif cfg.algorithm == 'DRIL':
      discriminator = SoftActor(state_size, action_size, cfg.imitation.model.hidden_size, cfg.imitation.model.activation, dropout=0.1)  # TODO: Check model HPs
    elif cfg.algorithm in ['FAIRL', 'GAIL', 'PUGAIL']:
      discriminator = GAILDiscriminator(state_size, action_size, cfg.imitation, forward_kl=cfg.algorithm == 'FAIRL')  # TODO: Account for model hidden size and depth + Account for model hidden size and depth
    elif cfg.algorithm == 'GMMIL':
      discriminator = GMMILDiscriminator(state_size, action_size, cfg.imitation)
    elif cfg.algorithm == 'RED':
      discriminator = REDDiscriminator(state_size, action_size, cfg.imitation)  # TODO: Check model HPs 
    if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']:
      discriminator_optimiser = optim.AdamW(discriminator.parameters(), lr=cfg.imitation.learning_rate, weight_decay=cfg.imitation.weight_decay)

  # Metrics
  metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])
  recent_returns = deque(maxlen=cfg.evaluation.average_window)  # Stores most recent evaluation returns

  if cfg.check_time_usage: start_time = time.time()  # Performance tracking
  # Behavioural cloning pretraining
  if cfg.pretraining.iterations > 0:
    expert_dataloader = iter(cycle(DataLoader(expert_trajectories, batch_size=cfg.pretraining.batch_size, shuffle=True, drop_last=True, num_workers=4)))
    actor_pretrain_optimiser = optim.AdamW(actor.parameters(), lr=cfg.pretraining.learning_rate, weight_decay=cfg.pretraining.weight_decay)  # Create separate pretraining optimiser
    for _ in tqdm(range(cfg.pretraining.iterations), leave=False):
      expert_transition = next(expert_dataloader)
      behavioural_cloning_update(actor, expert_transition, actor_pretrain_optimiser)
    if cfg.algorithm == 'BC':
        test_returns = evaluate_agent(actor, cfg.evaluation.episodes, cfg.env_name, cfg.imitation.absorbing, cfg.seed)
        steps = [*range(0, cfg.steps, cfg.evaluation.interval)]
        metrics['test_steps'].append(0)
        metrics['test_returns'].append(test_returns)
        lineplot(steps, len(steps)*[test_returns], 'test_returns', algo=cfg.algorithm, env=cfg.env_name)
        reward_returns = sum(test_returns)/len(test_returns)
        return reward_returns 
  if cfg.algorithm in ['DRIL', 'RED']:
    for _ in tqdm(range(cfg.imitation.pretraining.iterations), leave=False):
      if cfg.algorithm == 'DRIL':
        raise NotImplementedError  # TODO: Adapt HPs to run for certain number of iterations
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
  t, state, terminal, train_return = 0, env.reset(), False, 0
  pbar = tqdm(range(1, cfg.steps + 1), unit_scale=1, smoothing=0)
  for step in pbar:
    # Collect set of transitions by running policy π in the environment
    with torch.inference_mode():
      action = actor(state).sample()
      next_state, reward, terminal = env.step(action)
      t += 1
      train_return += reward
      memory.append(state, action, reward, next_state, terminal and t != env.max_episode_steps)  # True reward stored for SAC, should be overwritten by IL algorithms; if env terminated due to a time limit then do not count as terminal
      state = next_state

    # Reset environment and track metrics on episode termination
    if terminal:
      if cfg.imitation.absorbing and t != env.max_episode_steps: memory.wrap_for_absorbing_states()  # Wrap for absorbing state if terminated without time limit
      # Store metrics and reset environment
      metrics['train_steps'].append(step)
      metrics['train_returns'].append([train_return])
      pbar.set_description(f'Step: {step} | Return: {train_return}')
      t, state, train_return = 0, env.reset(), 0

    # Train agent and imitation learning component
    if step >= cfg.training.start and step % cfg.training.interval == 0:
      # Sample a batch of transitions
      transitions, expert_transitions = memory.sample(cfg.training.batch_size), expert_trajectories.sample(cfg.training.batch_size)

      if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED', 'SQIL']:
        # Train discriminator
        if cfg.algorithm in ['AIRL', 'FAIRL', 'GAIL', 'PUGAIL']:
          adversarial_imitation_update(cfg.algorithm, actor, discriminator, transitions, expert_transitions, discriminator_optimiser, grad_penalty=cfg.imitation.grad_penalty, mixup_alpha=cfg.imitation.mixup_alpha, pos_class_prior=cfg.imitation.get('pos_class_prior', 0.5), nonnegative_margin=cfg.imitation.get('nonnegative_margin', 0))
        
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
            # TODO: Importance weight expert absorbing states?
          elif cfg.algorithm == 'RED':
            transitions['rewards'] = discriminator.predict_reward(states, actions)
          elif cfg.algorithm == 'SQIL':
            sqil_sample(transitions, expert_transitions, cfg.training.batch_size)  # Rewrites training transitions as a mix of expert and policy data with constant reward functions TODO: Add sampling ratio option?
      
      # Train agent using SAC
      sac_update(actor, critic, log_alpha, target_critic, transitions, actor_optimiser, critic_optimiser, temperature_optimiser, cfg.reinforcement.discount, entropy_target, cfg.reinforcement.polyak_factor, max_grad_norm=cfg.reinforcement.max_grad_norm)
  
    # Evaluate agent and plot metrics
    if step % cfg.evaluation.interval == 0 and not cfg.check_time_usage:
      test_returns = evaluate_agent(actor, cfg.evaluation.episodes, cfg.env_name, cfg.imitation.absorbing, cfg.seed)
      recent_returns.append(sum(test_returns) / cfg.evaluation.episodes)
      metrics['test_steps'].append(step)
      metrics['test_returns'].append(test_returns)
      lineplot(metrics['test_steps'], metrics['test_returns'], 'test_returns', algo=cfg.algorithm, env=cfg.env_name)
      if len(metrics['train_returns']) > 0:  # Plot train returns if any
        lineplot(metrics['train_steps'], metrics['train_returns'], 'train_returns', algo=cfg.algorithm, env=cfg.env_name)

  if cfg.check_time_usage:
    metrics['training_time'] = time.time() - start_time

  if cfg.save_trajectories:
    # Store trajectories from agent after training
    _, trajectories = evaluate_agent(actor, cfg.evaluation.episodes, cfg.env_name, cfg.imitation.absorbing, cfg.seed, return_trajectories=True, render=cfg.render)
    torch.save(trajectories, 'trajectories.pth')
  # Save agent and metrics
  torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), log_alpha=log_alpha), 'agent.pth')
  if cfg.algorithm in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']: torch.save(discriminator.state_dict(), 'discriminator.pth')
  torch.save(metrics, 'metrics.pth')

  env.close()
  return sum(recent_returns) / float(cfg.evaluation.average_window)


if __name__ == '__main__':
  main()
