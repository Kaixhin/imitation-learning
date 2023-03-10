import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']  # TODO: Relative import from environments?


sns.set(style='white')


parser = argparse.ArgumentParser(description='Plot hyperparameter sweep results')
parser.add_argument('--path', type=str, default='outputs/BC_all_sweeper/03-09_19-13-21', help='Output path')
args = parser.parse_args()


# Load all data
experiments = {'scores': []}
entries = [entry for entry in os.scandir(args.path) if entry.is_dir()]  # Get all subdirectories
for entry in sorted(entries, key=lambda e: int(e.name)):  # Natural sorting (so dataframe row can be linked to subdirectory)
  # Parse searched hyperparameters
  with open(os.path.join(entry.path, '.hydra', 'overrides.yaml'), 'r') as stream: hyperparameters = yaml.safe_load(stream)
  hyperparameters = [hyperparameter for hyperparameter in hyperparameters if 'algorithm=' not in hyperparameter]  # Remove algorithm name
  for hyperparameter in hyperparameters:
    name, value = hyperparameter.split('=')
    if name not in experiments: experiments[name] = []
    experiments[name].append(value)
  # Collect scores
  env_scores = []
  for env in ENVS:
    env_scores.append(torch.load(os.path.join(entry.path, env, 'metrics.pth'))['test_returns_normalized'])
  experiments['scores'].append(np.stack(env_scores).mean())  # Transform scores to env x eval_step x num_evals and take mean
df = pd.DataFrame(experiments).sort_values('scores', ascending=False)


# Show parallel coordinates plot
fig, ax = plt.subplots()
cmap = sns.color_palette('rocket', as_cmap=True)
pd.plotting.parallel_coordinates(df, 'scores', color=[cmap(score) for score in df['scores']], ax=ax)
ax.get_legend().remove()
plt.show()
