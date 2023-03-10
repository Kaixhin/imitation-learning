import argparse
import os

import numpy as np
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import torch
import yaml


ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']  # TODO: Relative import from environments?


parser = argparse.ArgumentParser(description='Plot hyperparameter sweep results')
parser.add_argument('--path', type=str, default='', help='Output path')
args = parser.parse_args()
assert os.path.exists(args.path), f'Output folder {args.path} does not exist'


# Load all data
experiments = {'min_avg_test_return': [], 'test_steps': [], 'test_returns': []}
entries = [entry for entry in os.scandir(args.path) if entry.is_dir()]  # Get all subdirectories
for entry in sorted(entries, key=lambda e: int(e.name)):  # Natural sorting (so dataframe row can be linked to subdirectory)
  # Parse searched hyperparameters
  with open(os.path.join(entry.path, '.hydra', 'overrides.yaml'), 'r') as stream: hyperparameters = yaml.safe_load(stream)
  hyperparameters = [hyperparameter for hyperparameter in hyperparameters if 'algorithm=' not in hyperparameter]  # Remove algorithm name
  for hyperparameter in hyperparameters:
    name, value = hyperparameter.split('=')
    if name not in experiments: experiments[name] = []
    experiments[name].append(value)
  # Collect returns
  env_returns = []
  for env in ENVS:
    metrics = torch.load(os.path.join(entry.path, env, 'metrics.pth'))
    env_returns.append(metrics['test_returns_normalized'])
  experiments['test_steps'].append(metrics['test_steps'])
  experiments['test_returns'].append(np.stack(env_returns))  # Store returns as env x eval_step x num_evals
  experiments['min_avg_test_return'].append(experiments['test_returns'][-1].mean(axis=(1, 2)).min())  # Take mean for each env, then take min over envs


# Create dataframe
df = pd.DataFrame(experiments)
# Remove fixed hyperparameters
nunique = df[[c for c in df.columns if c not in ['test_steps', 'test_returns', 'min_avg_test_return']]].nunique()
df = df.drop(nunique[nunique == 1].index, axis=1)
# Infer best datatypes for data
df = df.apply(pd.to_numeric, errors='ignore')  # Converts to numeric data if possible, otherwise keeps original
df = df.convert_dtypes()  # Converts categorical data to string (np.arrays left as object)
# Print sorted dataframe
print(df.sort_values('min_avg_test_return', ascending=False))


# Plot returns from all experiments
fig, cmap = make_subplots(rows=5, cols=6, shared_xaxes=True, shared_yaxes=True, subplot_titles=list(map(lambda i: f'#{i[0]}: Min Ret. {i[1]:.3f}', enumerate(df['min_avg_test_return'])))), px.colors.qualitative.Plotly  # 30 experiments
for idx, row in df.iterrows():
  for e, env in enumerate(ENVS):
    fig.add_trace(go.Scatter(x=row['test_steps'], y=row['test_returns'][e].mean(axis=1), mode='markers' if row['test_returns'].shape[1] == 1 else 'lines', line={'color': cmap[e]}, name=env, showlegend=idx == 0), row=(idx // 6) + 1, col=(idx % 6) + 1)
fig.update_annotations(font_size=13)  # Reduce title font size
fig.update_yaxes(range=(-0.1, 1.5))  # Set shared plot ranges
fig.show()

# Plot parallel coordinates for hyperparameter search
fig = go.Figure()
dimensions = []
for c in df.columns:
  if c not in ['test_steps', 'test_returns', 'min_avg_test_return']:
    if df[c].dtype == 'string':  # Convert categorical options into ints manually to control plotting properly
      codes, uniques = pd.factorize(df[c])
      dimension = {'values': codes, 'label': c, 'tickvals': codes, 'ticktext': df[c]}
    else:
      dimension = {'values': df[c], 'label': c}
    dimensions.append(dimension)
fig.add_trace(go.Parcoords(line={'color': df['min_avg_test_return'], 'showscale': True}, dimensions=dimensions))
fig.show()
