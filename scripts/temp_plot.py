import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']  # TODO: Relative import from environments?


sns.set(style='white')


parser = argparse.ArgumentParser(description='Plot hyperparameter sweep results')
parser.add_argument('--path', type=str, default='outputs/BC_all_sweeper/03-09_19-13-21', help='Output path')
args = parser.parse_args()


# Load all data
experiments = []
for entry in os.scandir(args.path):
  if entry.is_dir():
    # Collect scores
    env_scores = []
    for env in ENVS:
      env_scores.append(torch.load(os.path.join(entry.path, env, 'metrics.pth'))['test_returns_normalized'])
    # Parse hyperparameters
    with open(os.path.join(entry.path, '.hydra', 'overrides.yaml'), 'r') as stream:
      hyperparameters = yaml.safe_load(stream)
    hyperparameters = [hyperparameter for hyperparameter in hyperparameters if 'algorithm=' not in hyperparameter]  # Remove algorithm name
    experiments.append({'hyperparameters': hyperparameters, 'scores': np.stack(env_scores)})  # Store hyperparameters and scores as env x eval_step x num_evals
