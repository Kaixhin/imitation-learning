import torch

from environments import D4RLEnv


# Evaluate agent with deterministic policy π
def evaluate_agent(actor, num_episodes, env_type, env_name, absorbing, seed, return_trajectories=False, render=False):
  env = env_type(env_name, absorbing)
  env.seed(seed)

  returns, trajectories = [], []
  if render: env.render()  # PyBullet requires creating render window before first env reset, and then updates without requiring first call

  with torch.inference_mode():
    for _ in range(num_episodes):
      states, actions, rewards = [], [], []
      state, terminal = env.reset(), False
      while not terminal:
          action = actor.get_greedy_action(state)  # Take greedy action
          state, reward, terminal = env.step(action)

          if return_trajectories:
            states.append(state)
            actions.append(action)
          rewards.append(reward)
      returns.append(sum(rewards))

      if return_trajectories:
        # Collect trajectory data (including terminal signal, which may be needed for offline learning)
        terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
        trajectories.append({'states': torch.cat(states), 'actions': torch.cat(actions), 'rewards': torch.tensor(rewards, dtype=torch.float32), 'terminals': terminals})

  env.close()
  return (returns, trajectories) if return_trajectories else returns
