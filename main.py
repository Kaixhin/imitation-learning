import argparse
from collections import deque
import os

import torch
from torch import optim
from tqdm import tqdm

from environments import CartPoleEnv, D4RLEnv
from evaluation import evaluate_agent
from models import Actor, ActorCritic, AIRLDiscriminator, GAILDiscriminator, GMMILDiscriminator, REDDiscriminator
from training import TransitionDataset, adversarial_imitation_update, behavioural_cloning_update, compute_advantages, indicate_absorbing, ppo_update, target_estimation_update
from utils import flatten_list_dicts, lineplot

import hydra
from omegaconf import DictConfig, OmegaConf


# Setup
"""
parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--steps', type=int, default=100000, metavar='T', help='Number of environment steps')
parser.add_argument('--hidden-size', type=int, default=32, metavar='H', help='Hidden size')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount')
parser.add_argument('--trace-decay', type=float, default=0.95, metavar='λ', help='GAE trace decay')
parser.add_argument('--ppo-clip', type=float, default=0.2, metavar='ε', help='PPO clip ratio')
parser.add_argument('--ppo-epochs', type=int, default=4, metavar='K', help='PPO epochs')
parser.add_argument('--value-loss-coeff', type=float, default=0.5, metavar='c1', help='Value loss coefficient')
parser.add_argument('--entropy-loss-coeff', type=float, default=0, metavar='c2', help='Entropy regularisation coefficient')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=2048, metavar='B', help='Minibatch size')
parser.add_argument('--max-grad-norm', type=float, default=1, metavar='N', help='Maximum gradient L2 norm')
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='EI', help='Evaluation interval')
parser.add_argument('--evaluation-episodes', type=int, default=50, metavar='EE', help='Evaluation episodes')
parser.add_argument('--save-trajectories', action='store_true', default=False, help='Store trajectories from agent after training')
parser.add_argument('--imitation', type=str, default='', choices=['AIRL', 'BC', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED'], metavar='I', help='Imitation learning algorithm')
parser.add_argument('--state-only', action='store_true', default=False, help='State-only imitation learning')
parser.add_argument('--absorbing', action='store_true', default=False, help='Indicate absorbing states')
parser.add_argument('--imitation-epochs', type=int, default=5, metavar='IE', help='Imitation learning epochs')
parser.add_argument('--imitation-batch-size', type=int, default=128, metavar='IB', help='Imitation learning minibatch size')
parser.add_argument('--imitation-replay-size', type=int, default=4, metavar='IRS', help='Imitation learning trajectory replay size')
parser.add_argument('--r1-reg-coeff', type=float, default=1, metavar='γ', help='R1 gradient regularisation coefficient')
parser.add_argument('--pos-class-prior', type=float, default=0.5, metavar='η', help='Positive class prior')
parser.add_argument('--nonnegative-margin', type=float, default=0, metavar='β', help='Non-negative margin')
#args = parser.parse_args()
"""

code_path = os.getcwd()
# Set up environment and models
@hydra.main(config_path='conf', config_name='config')
def main(args: DictConfig) -> None:
  os.makedirs('./results', exist_ok=True)
  print("Working directory for current run: " + os.getcwd())
  torch.manual_seed(args.seed)
  env = D4RLEnv(args.environment.env_name)
  env.seed(args.seed)
  agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_size)
  agent_optimiser = optim.RMSprop(agent.parameters(), lr=args.learning_rate)
  if args.imitation:
    # Set up expert trajectories dataset
    expert_trajectories = env.get_dataset()
    #expert_trajectories = TransitionDataset(flatten_list_dicts(torch.load(code_path+'/expert_trajectories.pth')))
    # Set up discriminator
    if args.imitation in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
      if args.imitation == 'AIRL':
        discriminator = AIRLDiscriminator(env.observation_space.shape[0] + (1 if args.absorbing else 0), env.action_space.n, args.hidden_size, args.discount, state_only=args.state_only)
      elif args.imitation == 'DRIL':
        discriminator = Actor(env.observation_space.shape[0], env.action_space.n, args.hidden_size, dropout=0.1)
      elif args.imitation in ['FAIRL', 'GAIL', 'PUGAIL']:
        discriminator = GAILDiscriminator(env.observation_space.shape[0] + (1 if args.absorbing else 0), env.action_space.n, args.hidden_size, state_only=args.state_only, forward_kl=args.imitation == 'FAIRL')
      elif args.imitation == 'GMMIL':
        discriminator = GMMILDiscriminator(env.observation_space.shape[0] + (1 if args.absorbing else 0), env.action_space.n, state_only=args.state_only)
      elif args.imitation == 'RED':
        discriminator = REDDiscriminator(env.observation_space.shape[0] + (1 if args.absorbing else 0), env.action_space.n, args.hidden_size, state_only=args.state_only)
      if args.imitation in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']:
        discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=args.learning_rate)
  # Metrics
  metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])


  # Main training loop
  state, terminal, episode_return, trajectories, policy_trajectory_replay_buffer = env.reset(), False, 0, [], deque(maxlen=args.imitation_replay_size)
  pbar = tqdm(range(1, args.steps + 1), unit_scale=1, smoothing=0)
  for step in pbar:
    if args.imitation in ['BC', 'DRIL', 'RED']:
      if step == 1:
        for _ in tqdm(range(args.imitation_epochs), leave=False):
          if args.imitation == 'BC':
            # Perform behavioural cloning updates offline
            behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, args.imitation_batch_size)
          elif args.imitation == 'DRIL':
            # Perform behavioural cloning updates offline on policy ensemble (dropout version)
            behavioural_cloning_update(discriminator, expert_trajectories, discriminator_optimiser, args.imitation_batch_size)
            with torch.no_grad():
              discriminator.set_uncertainty_threshold(expert_trajectories['states'], expert_trajectories['actions'])
          elif args.imitation == 'RED':
            # Train predictor network to match random target network
            target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, args.imitation_batch_size, args.absorbing)

    if args.imitation != 'BC':
      # Collect set of trajectories by running policy π in the environment
      policy, value = agent(state)
      action = policy.sample()
      log_prob_action, entropy = policy.log_prob(action), policy.entropy()
      next_state, reward, terminal = env.step(action)
      episode_return += reward
      trajectories.append(dict(states=state, actions=action, rewards=torch.tensor([reward], dtype=torch.float32), terminals=torch.tensor([terminal], dtype=torch.float32), log_prob_actions=log_prob_action, old_log_prob_actions=log_prob_action.detach(), values=value, entropies=entropy))
      state = next_state

      if terminal:
        # Store metrics and reset environment
        metrics['train_steps'].append(step)
        metrics['train_returns'].append([episode_return])
        pbar.set_description('Step: %i | Return: %f' % (step, episode_return))
        state, episode_return = env.reset(), 0

        if len(trajectories) >= args.batch_size:
          policy_trajectories = flatten_list_dicts(trajectories)  # Flatten policy trajectories (into a single batch for efficiency; valid for feedforward networks)
          trajectories = []  # Clear the set of trajectories

          if args.imitation in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'PUGAIL', 'RED']:
            # Train discriminator and predict rewards
            if args.imitation in ['AIRL', 'FAIRL', 'GAIL', 'PUGAIL']:
              # Use a replay buffer of previous trajectories to prevent overfitting to current policy
              policy_trajectory_replay_buffer.append(policy_trajectories)
              policy_trajectory_replays = flatten_list_dicts(policy_trajectory_replay_buffer)
              for _ in tqdm(range(args.imitation_epochs), leave=False):
                adversarial_imitation_update(args.imitation, agent, discriminator, expert_trajectories, TransitionDataset(policy_trajectory_replays), discriminator_optimiser, args.imitation_batch_size, args.absorbing, args.r1_reg_coeff, args.pos_class_prior, args.nonnegative_margin)

            # Predict rewards
            states, actions, next_states, terminals = policy_trajectories['states'], policy_trajectories['actions'], torch.cat([policy_trajectories['states'][1:], next_state]), policy_trajectories['terminals']
            if args.absorbing: states, actions, next_states = indicate_absorbing(states, actions, policy_trajectories['terminals'], next_states)

            with torch.no_grad():
              if args.imitation == 'AIRL':
                policy_trajectories['rewards'] = discriminator.predict_reward(states, actions, next_states, policy_trajectories['log_prob_actions'].exp(), terminals)
              elif args.imitation == 'DRIL':
                # Note that by default DRIL also includes behavioural cloning online
                policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
              elif args.imitation in ['FAIRL', 'GAIL', 'PUGAIL']:
                policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
              elif args.imitation == 'GMMIL':
                expert_states, expert_actions = expert_trajectories['states'], expert_trajectories['actions']
                if args.absorbing: expert_states, expert_actions = indicate_absorbing(expert_states, expert_actions, expert_trajectories['terminals'])
                policy_trajectories['rewards'] = discriminator.predict_reward(states, actions, expert_states, expert_actions)
              elif args.imitation == 'RED':
                policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)

          # Compute rewards-to-go R and generalised advantage estimates ψ based on the current value function V
          compute_advantages(policy_trajectories, agent(state)[1], args.discount, args.trace_decay)
          # Normalise advantages
          policy_trajectories['advantages'] = (policy_trajectories['advantages'] - policy_trajectories['advantages'].mean()) / (policy_trajectories['advantages'].std() + 1e-8)

          # Perform PPO updates
          for epoch in tqdm(range(args.ppo_epochs), leave=False):
            ppo_update(agent, policy_trajectories, agent_optimiser, args.ppo_clip, epoch, args.value_loss_coeff, args.entropy_loss_coeff)

    # Evaluate agent and plot metrics
    if step % args.evaluation_interval == 0:
      metrics['test_steps'].append(step)
      metrics['test_returns'].append(evaluate_agent(agent, args.evaluation_episodes, seed=args.seed))
      lineplot(metrics['test_steps'], metrics['test_returns'], 'test_returns')
      if args.imitation != 'BC': lineplot(metrics['train_steps'], metrics['train_returns'], 'train_returns')


  if args.save_trajectories:
    # Store trajectories from agent after training
    _, trajectories = evaluate_agent(agent, args.evaluation_episodes, return_trajectories=True, seed=args.seed)
    torch.save(trajectories, os.path.join('results', 'trajectories.pth'))

  # Save agent and metrics
  torch.save(agent.state_dict(), os.path.join('results', 'agent.pth'))
  if args.imitation in ['AIRL', 'DRIL', 'FAIRL', 'GAIL', 'PUGAIL', 'RED']: torch.save(discriminator.state_dict(), os.path.join('results', 'discriminator.pth'))
  torch.save(metrics, os.path.join('results', 'metrics.pth'))
  env.close()

if __name__ == '__main__':
  main()