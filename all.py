import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import multiprocessing as mp
from .main import main


@hydra.main(config_path='conf', config_name='config')
def all(cfg: DictConfig):
    all_envs = dict(ant='ant-expert-v2', halfcheetah='halfcheetah-expert-v2',
                    hopper='hopper-expret-v2', walker2d='walker2d-expert-v2')
    main(cfg, file_prefix='')
