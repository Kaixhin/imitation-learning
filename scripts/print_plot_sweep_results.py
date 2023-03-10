import argparse
import os

import numpy as np
import plotly.express as px
import pandas as pd
import torch
import yaml

ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']  # TODO: Relative import from environments?



parser = argparse.ArgumentParser(description='Plot hyperparameter sweep results')
parser.add_argument('--path', type=str, default='', help='Output path')
args = parser.parse_args()
assert os.path.exists(args.path), f'Output folder {args.path} does not exist'


# Load all data
experiments = {'score': []}
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
  experiments['score'].append(np.stack(env_scores).mean())  # Transform scores to env x eval_step x num_evals and take mean
df = pd.DataFrame(experiments).sort_values('score', ascending=False)


# Print dataframe and show parallel coordinates plot
print(df)
fig = px.parallel_coordinates(df, color='score')
fig.show()
