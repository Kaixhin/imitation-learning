import argparse
import os
import sys

import numpy as np
from rliable import library as rly, metrics, plot_utils
import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Allow importing from root dir
from environments import ENVS
BASELINE_ALGORITHM = 'BC'
ALGORITHMS = ['AdRIL', 'GAIL', 'RED']
TRAJECTORIES = ['5', '10', '25']
SEEDS = 10

parser = argparse.ArgumentParser(description='Plot seed sweep results')
parser.add_argument('--reps', type=int, default=50000, help='Number of bootstrap samples')
args = parser.parse_args()


# Load all data into normalised score matrices (runs x envs x evals)
experiments = {}
for trajectories in TRAJECTORIES:
  experiments[trajectories] = {}
  for algorithm in [BASELINE_ALGORITHM] + ALGORITHMS:
     experiments[trajectories][algorithm] = np.zeros((30 * SEEDS, len(ENVS), 100))

for algorithm in [BASELINE_ALGORITHM] + ALGORITHMS:
  for e, env in enumerate(ENVS):
    for trajectories in TRAJECTORIES:
      for seed in range(SEEDS):
        experiments[trajectories][algorithm][30 * seed:30 * (seed + 1), e, :] = np.asarray(torch.load(os.path.join('outputs', f'{algorithm}_{env}_sweeper', f'traj_{trajectories}', str(seed), 'metrics.pth'))['test_returns_normalized']).T


# Calculate bootstrap metrics and plot
aggregate_func = lambda x: np.array([metrics.aggregate_median(x[:, :, -1]), metrics.aggregate_iqm(x[:, :, -1]), metrics.aggregate_mean(x[:, :, -1]), metrics.aggregate_optimality_gap(x[:, :, -1])])
for trajectories in TRAJECTORIES:
  aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(experiments[trajectories], aggregate_func, reps=args.reps)
  fig, axes = plot_utils.plot_interval_estimates(aggregate_scores, aggregate_score_cis, metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'], algorithms=[BASELINE_ALGORITHM] + ALGORITHMS, xlabel='Human Normalized Score')
  fig.savefig(os.path.join('scripts', f'aggregate_scores_traj_{trajectories}.png'), bbox_inches='tight')
