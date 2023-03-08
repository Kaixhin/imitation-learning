import copy
import os
from pathlib import Path
import random

import hydra
from hydra.utils import get_original_cwd
import torch.multiprocessing as mp 
import numpy as np
from omegaconf import DictConfig
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import run 
from environments import get_all_env_baseline

#torch.set_num_threads(1) This work the best, but probably slower
if __name__ == '__main__': mp.set_start_method('forkserver') #This circumvent deadlock


def pool_wrapper(cfg):
  env_type = cfg.env_type
  result = run(cfg, file_prefix=env_type+'/')
  return [env_type, result]


@hydra.main(version_base=None, config_path='conf', config_name='all_config')
def all(cfg: DictConfig):
  all_envs = dict(ant='ant-expert-v2', halfcheetah='halfcheetah-expert-v2', hopper='hopper-expert-v2', walker2d='walker2d-expert-v2')
  filename = os.path.join(get_original_cwd(), 'normalization_data.npz')
  all_env_cfgs = [] 
  seed = random.randint(0, 99)
  Path(f"./seed{seed}.txt").touch()
  for key, value in all_envs.items():
    tmp_cfg = copy.copy(cfg); tmp_cfg.env_type, tmp_cfg.env_name = key, value
    tmp_cfg.seed = seed
    if not os.path.exists(key): os.mkdir(key)
    all_env_cfgs.append(tmp_cfg)
  with mp.Pool(processes=4) as pool:
    normalized_result = {item[0]: item[1] for item in pool.map(pool_wrapper, all_env_cfgs)}
    pool.close()
  result_list = []
  for key, value in normalized_result.items():
    result_list.append(value)
    print(f"{key} Result(normalized): {value}")
  
  return np.min(result_list)


if __name__ == '__main__':
  all()
