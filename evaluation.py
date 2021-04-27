import torch

from environments import PendulumEnv, D4RLEnv


# Evaluate agent with deterministic policy π
def evaluate_agent(agent, episodes, return_trajectories=False, Env=PendulumEnv, env_name="", seed=1, render=False):
  env = Env(env_name)
  env.seed(seed)

  returns, trajectories = [], []
  if render:
    env.render()
    env.reset()
  for _ in range(episodes):
    states, actions, rewards = [], [], []
    state, terminal = env.reset(), False
    while not terminal:
      with torch.no_grad():
        action = agent.greedy_action(state)  # take action greedily
        state, reward, terminal = env.step(action)
        if return_trajectories:
          states.append(state)
          actions.append(action)
        rewards.append(reward)
    returns.append(sum(rewards))
    if return_trajectories:
      # Collect trajectory data (including terminal signal, which may be needed for offline learning)
      terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
      trajectories.append(dict(states=torch.cat(states), actions=torch.cat(actions), rewards=torch.tensor(rewards, dtype=torch.float32), terminals=terminals))
  env.close()
  return (returns, trajectories) if return_trajectories else returns
