import argparse
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from rliable import library as rly, metrics, plot_utils
import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Allow importing from root dir
from environments import ENVS
ALGORITHMS = ['BC', 'AdRIL', 'GAIL', 'GMMIL', 'RED']
TRAJECTORIES = ['5', '10', '25']
SEEDS = 10

parser = argparse.ArgumentParser(description='Plot seed sweep results')
parser.add_argument('--reps', type=int, default=50000, help='Number of bootstrap samples')
args = parser.parse_args()
cfg = OmegaConf.load(os.path.join('conf', 'train_config.yaml'))  # Load default config


# Load all data into normalised score matrices (runs x envs x evals)
returns_dict = {}
for trajectories in TRAJECTORIES:
  returns_dict[trajectories] = {}
  for algorithm in ALGORITHMS:
    returns_dict[trajectories][algorithm] = np.zeros((cfg.evaluation.episodes * SEEDS, len(ENVS), cfg.steps // cfg.evaluation.interval))  # TODO: Get IQM over episodes in each seed?

for algorithm in ALGORITHMS:
  for e, env in enumerate(ENVS):
    for trajectories in TRAJECTORIES:
      for seed in range(SEEDS):
        returns_dict[trajectories][algorithm][cfg.evaluation.episodes * seed:cfg.evaluation.episodes * (seed + 1), e, :] = np.asarray(torch.load(os.path.join('outputs', f'{algorithm}_{env}_sweeper', f'traj_{trajectories}', str(seed), 'metrics.pth'))['test_returns_normalized']).T


# Calculate bootstrap metrics, print and plot
print('IQM ± 1 C.I.')
iqm = lambda x: np.array([metrics.aggregate_iqm(x[..., t]) for t in range(x.shape[-1])])
for trajectories in TRAJECTORIES:
  iqm_scores, iqm_cis = rly.get_interval_estimates(returns_dict[trajectories], iqm, reps=args.reps)

  print(f'\nTrajectories: {trajectories}')
  for algorithm in ALGORITHMS: print(f'{algorithm}: {iqm_scores[algorithm][-1]:.3f} ± {(iqm_cis[algorithm][1, -1] - iqm_cis[algorithm][0, -1]) // 2:.3f}')
  
  axes = plot_utils.plot_sample_efficiency_curve(range(1, cfg.steps + 1, cfg.evaluation.interval), iqm_scores, iqm_cis, algorithms=ALGORITHMS, xlabel=r'Training Steps', ylabel='IQM Normalised Score')
  plt.savefig(os.path.join('scripts', f'sample_efficiency_traj_{trajectories}.png'), bbox_inches='tight')
