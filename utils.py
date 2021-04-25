import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import warnings

sns.set(style='white')


# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts):
  return {k: torch.cat([d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}


# Makes a lineplot with scalar x and statistics of vector y
def lineplot(x, y, filename, xaxis='Steps', yaxis='Returns'):
  y = np.array(y)
  y_mean, y_std = y.mean(axis=1), y.std(axis=1)
  sns.lineplot(x=x, y=y_mean, color='coral')
  plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='coral', alpha=0.3)
  plt.xlim(left=0, right=x[-1])
  plt.ylim(bottom=0, top=500)  # Return limits for CartPole-v1
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  plt.savefig(os.path.join('./results', filename + '.png'))
  plt.close()



