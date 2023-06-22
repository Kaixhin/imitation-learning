import argparse
import math
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from rliable import library as rly, metrics
import scipy
import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Allow importing from root dir
from environments import ENVS
ALGORITHMS = ['BC', 'AdRIL', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED']
COLOURS ={'BC': '#808080', 'AdRIL': '#FF9900', 'DRIL': '#B0660D', 'GAIL': '#26B04A', 'GMMIL': '#971AFF', 'PWIL': '#0639E6', 'RED': '#FD1111'}
TRAJECTORIES = ['5', '10', '25']

parser = argparse.ArgumentParser(description='Plot seed sweep results')
parser.add_argument('--reps', type=int, default=50000, help='Number of bootstrap samples')
parser.add_argument('--algorithms', type=lambda algorithms: [algorithm for algorithm in algorithms.split(',')], default=ALGORITHMS, help='Algorithms to evaluate')
parser.add_argument('--envs', type=lambda envs: [env for env in envs.split(',')], default=ENVS, help='Envs to evaluate')
parser.add_argument('--seeds', type=int, default=10, help='Max number of seeds to evaluate')
args = parser.parse_args()
cfg = OmegaConf.load(os.path.join('conf', 'train_config.yaml'))  # Load default config
plt.style.use('seaborn-v0_8-paper')  # Use Seaborn style
plt.rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})


# Create normalised score matrices (runs x envs x evals)
returns_dict = {}
for trajectories in TRAJECTORIES:
  returns_dict[trajectories] = {}
  for algorithm in args.algorithms:
    returns_dict[trajectories][algorithm] = np.zeros((args.seeds, len(args.envs), cfg.steps // cfg.evaluation.interval))
# Load data into matrices
for algorithm in args.algorithms:
  for e, env in enumerate(args.envs):
    for trajectories in TRAJECTORIES:
      for seed in range(args.seeds):
        returns = np.asarray(torch.load(os.path.join('outputs', f'{algorithm}_{env}_sweeper', f'traj_{trajectories}', str(seed), 'metrics.pth'))['test_returns_normalized']).T
        returns_dict[trajectories][algorithm][seed:seed + 1, e, :] = scipy.stats.trim_mean(returns, proportiontocut=0.25, axis=0)  # Use IQM over evaluation episodes for each seed


# Calculate bootstrap metrics, print and plot
print('IQM ± 1 C.I.')
iqm = lambda x: np.array([metrics.aggregate_iqm(x[..., t]) for t in range(x.shape[-1])])
fig, axes = plt.subplots(1, len(TRAJECTORIES), figsize=(5 * len(TRAJECTORIES) - 4, 4), sharey=True)
artist_handles = []
for t, trajectories in enumerate(TRAJECTORIES):
  iqm_scores, iqm_cis = rly.get_interval_estimates(returns_dict[trajectories], iqm, reps=args.reps)

  print(f'\nTrajectories: {trajectories}')
  axes[t].set_title(f'{trajectories} Trajectories')
  axes[t].set_xlim(0, cfg.steps + 2)
  x = range(1, cfg.steps + 1, cfg.evaluation.interval)
  for algorithm in args.algorithms:
    print(f'{algorithm}: {iqm_scores[algorithm][-1]:.3f} ± {(iqm_cis[algorithm][1, -1] - iqm_cis[algorithm][0, -1]) // 2:.3f}')  # TODO: Check CIs are printed properly
    plot, = axes[t].plot(x, iqm_scores[algorithm], color=COLOURS[algorithm], label=algorithm)
    artist_handles.append(plot)
    axes[t].fill_between(x, iqm_cis[algorithm][0], iqm_cis[algorithm][1], alpha=0.3, facecolor=COLOURS[algorithm])

#axes[0].set_ylabel('IQM Normalised Score')
#axes[floor(len(TRAJECTORIES) // 2)].set_xlabel('Training Steps')
legend = plt.figlegend(handles=artist_handles[:len(args.algorithms)], loc='center', bbox_to_anchor=(0.5, -0.05), ncols=len(args.algorithms))
plt.savefig(os.path.join('scripts', f'sample_efficiency_traj.png'), bbox_extra_artists=(legend, ), bbox_inches='tight')
