import os
import random

import hydra
from omegaconf import DictConfig
from torch import multiprocessing as mp

from main import run


def map_func(cfg: DictConfig):
  return run(cfg, file_prefix=f'{cfg.env}/')


@hydra.main(version_base=None, config_path='conf', config_name='all_config')
def all(cfg: DictConfig):
  envs = ['ant', 'halfcheetah', 'hopper', 'walker2d']
  env_cfgs = []
  seed = random.randint(0, 99)  # Set a random random seed for all envs
  for env in envs:
    # Create a new config to run
    env_cfg = cfg.copy()
    env_cfg.env, env_cfg.seed = env, seed  # Overwrite the env and seed
    env_cfgs.append(env_cfg)
    os.makedirs(env)  # Create a new (empty) folder for each env
  with mp.Pool(processes=len(envs)) as pool:
    avg_scores = pool.map(map_func, env_cfgs)
  return min(avg_scores)


if __name__ == '__main__':
  all()
