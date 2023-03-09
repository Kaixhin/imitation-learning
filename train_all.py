import os
import random

import hydra
from omegaconf import DictConfig
from torch import multiprocessing as mp

from environments import ENVS
from train import train


def map_func(cfg: DictConfig):
  return train(cfg, file_prefix=f'{cfg.env}/')


@hydra.main(version_base=None, config_path='conf', config_name='train_all_config')
def main(cfg: DictConfig):
  env_cfgs = []
  seed = random.randint(0, 99)  # Set a random random seed for all envs
  for env in ENVS:
    # Create a new config to run
    env_cfg = cfg.copy()
    env_cfg.env, env_cfg.seed = env, seed  # Overwrite the env and seed
    env_cfgs.append(env_cfg)
    os.makedirs(env)  # Create a new (empty) folder for each env
  with mp.Pool(processes=len(ENVS)) as pool:
    avg_scores = pool.map(map_func, env_cfgs)
  return min(avg_scores)


if __name__ == '__main__':
  mp.set_start_method('forkserver')  # Prevent deadlock occurring with Ax sweeper
  main()
