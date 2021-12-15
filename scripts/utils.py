from datetime import datetime
from time import strftime
import hydra
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import yaml
import math
import types
import gym, d4rl
OUTPUT_FOLDER='./output/'
ALGORITHMS = ['SAC', 'BC', 'GAIL','GMMIL', 'RED', 'DRIL', 'SQIL']
ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']
HYDRA_CONF = ['config', 'overrides', 'hydra']
DEFAULT_DATEFORMAT="%m-%d_%H-%M-%S"

def str_float_format(value):
  if type(value) is float:
    value = "{:.0e}".format(value)
  return value

def remove_same_value_list_dict(list_of_dict: list):
  unique_dict = dict()
  for dic in list_of_dict:
    for key, value in dic.items():
      if key in unique_dict.keys():
        unique_dict[key].add(value)
      else:
        unique_dict[key]=set([value])
  for key, value in unique_dict:
    if len(unique_dict[key]) <= 1:
      for dic in list_of_dict:
        dic.pop(key, None)
  return list_of_dict

def read_hydra_yaml(file_name: str):
  with open(file_name, 'r') as fstream:
    data = yaml.safe_load(fstream)
  if type(data) == dict: return data
  dict_data = dict()
  for d in data:
    key, value = d.split('=')
    try:
      value = float(value)
    except ValueError as e:
      pass
    dict_data[key] = value
  return dict_data

def read_hydra_configs(folder_name: str):
  hydra_conf = dict()
  for conf_name in HYDRA_CONF:
    hydra_conf[conf_name] = read_hydra_yaml(os.path.join(folder_name, conf_name+'.yaml'))
  return hydra_conf

def filter_datefolder(folder_name, date_from=None, date_to=None, date_format=DEFAULT_DATEFORMAT):
  """Input should be the folder containing the datetime folders. Given input folder_name, returns a list of all the subfolders with date_format between given date_from/date_to. 
     If date_from is None, take all folders up to date_to. If date_to is None, take all folders up from date_from. if both are None, just give back all subfolders"""
  assert os.path.isdir(folder_name)
  dirname, subdirname, _ = [dir for dirs in os.walk(folder_name) if  not dirs[2]][0] # Assumes folder with datetime subfolder contain no file.
  if date_from is None and date_to is None:
    return [os.path.join(dirname, sdn) for sdn in subdirname]
  else:
    datetime_folder = [datetime.strptime(sdn, date_format) for sdn in subdirname]
    datetime_from = datetime.strptime(date_from, date_format) if date_from else 0
    datetime_to = datetime.strptime(date_to, date_format) if date_to else 0
    check_datetime = lambda x: (x > datetime_from or date_from is None) and (x < datetime_to or date_to is None)
    filtered_subdirname = [sdn for sdn in subdirname if check_datetime(datetime.strptime(sdn, date_format))]
    return [os.path.join(dirname, sdn) for sdn in filtered_subdirname]


def load_data(env, alg, date_from=None, date_to=None, folder_prefix='all_', output_folder='./outputs/', get_hydra_conf=False):
    if 'all' in folder_prefix: # all folders contain the different envs as subfolder
      data_folder_name = folder_prefix  + alg 
      full_folder_path_name = os.path.join(output_folder, data_folder_name)
      data_subfolders = filter_datefolder(full_folder_path_name, date_from=date_from, date_to=date_to)
      metrics_folders = [subfolder[0] for subfolder in os.walk(data_subfolders) if 'metrics.pth' in subfolder[2] and env in subfolder[0]]
      data = [torch.load(os.path.join(mf, 'metrics.pth')) for mf in metrics_folders]
      if get_hydra_conf:
        hydra_paths = ['/'.join(mf.split('/')[:-1]) for mf in metrics_folders]
        hydra_paths = [os.path.join(hp, '.hydra') for hp in hydra_paths]

    elif 'par' in folder_prefix: #Seed sweeper data generated with  run_parallel_seed_experiments.sh. Does not have datetime subfolder structure
      data_folder_name = folder_prefix  + alg + '_' + env
      full_folder_path_name = os.path.join(output_folder, data_folder_name)
      metrics_folders = [subfolder[0] for subfolder in os.walk(full_folder_path_name) if 'metrics.pth' in subfolder[2]]
      data = [torch.load(os.path.join(mf, 'metrics.pth')) for mf in metrics_folders]
      if get_hydra_conf: hydra_paths = [os.path.join(hp, '.hydra') for hp in metrics_folders]
    else:
      data_folder_name = folder_prefix + env + '_' + alg
      full_folder_path_name = os.path.join(output_folder, data_folder_name)
      data_subfolders = filter_datefolder(full_folder_path_name, date_from=date_from, date_to=date_to)
      metrics_folders = [subfolder[0] for subfolder in os.walk(data_subfolders) if 'metrics.pth' in subfolder[2] ]
      data = [torch.load(os.path.join(mf, 'metrics.pth')) for mf in metrics_folders]
      if get_hydra_conf: hydra_paths = [os.path.join(hp, '.hydra') for hp in metrics_folders]
    if get_hydra_conf:
      hydra_confs = [read_hydra_configs(hydra_path) for hydra_path in hydra_paths]
      return data, hydra_confs
    else:
      return data

def load_all_data(date_from=None, date_to=None, folder_prefix=''):
    data = dict()
    if date_from: print(f'Checking seed data from... {date_from}')
    for env in ENVS:
        metrics = dict()
        for alg in ALGORITHMS:
            try:
                metrics[alg] = load_data(env, alg, date_from=date_from, date_to=date_to, folder_prefix=folder_prefix)
            except Exception as e:
                print("Error: Could not load data from environment: " + env + ' and algorithm: ' + alg)
        data[env] = metrics
    return data

def process_test_data(data):
    x = data[0]['test_steps']
    means, stds = [], []
    n_seed = len(data)
    for metric in data:
        y = np.array(metric['test_returns'])
        y_mean, y_std = y.mean(axis=1), y.std(axis=1)
        means.append(y_mean)
        stds.append(y_std)
    means = np.array(means)
    stds = np.array(stds)
    mean_of_means = means.mean(axis=0)
    std_of_means = means.std(axis=0)
    std_err = std_of_means / np.sqrt(n_seed)
    return np.array(x), mean_of_means, std_err, std_of_means

def scan_folder_trajectories(folder):
  subfolders = [subdir[0] for subdir in os.walk(folder) if 'optimization_results.yaml' in subdir[2]]
  trajectory_nums = []
  for subfolder in subfolders:
    hydra_dict = read_hydra_yaml(os.path.join(subfolder, 'optimization_results.yaml'))
    for key, value in hydra_dict['ax'].items():
      if 'trajectories' in key:
        trajectory_nums.append(int(value))
  return trajectory_nums 


def get_trajectory_subfolder(alg, folder, num_trajectory, include_datetime=False, date_format=DEFAULT_DATEFORMAT):
  subfolders = [subdir[0] for subdir in os.walk(folder) if 'optimization_results.yaml' in subdir[2]] 
  for subfolder in subfolders:
    hydra_dict = read_hydra_yaml(os.path.join(subfolder, 'optimization_results.yaml'))
    for key, value in hydra_dict['ax'].items():
      if 'trajectories' in key:
        if int(value) == num_trajectory:
          if include_datetime:
            subfolder_name = os.path.basename(os.path.normpath(subfolder))
            subfolder_datetime = datetime.strptime(subfolder_name, date_format)
            date_from, date_to = subfolder_datetime - datetime.timedelta(seconds=1), subfolder_datetime + datetime.timedelta(seconds=1) 
            return subfolder, date_from.strftime(date_format), date_to.strftime(date_format)
          else:
            return subfolder
      
def _get_env_baseline(env: gym.Env):
  random_agent_mean, expert_mean = env.ref_min_score, env.ref_max_score
  return expert_mean, random_agent_mean

def get_all_env_baseline(envs: dict):
  data =  dict()
  for env_name in envs.keys():
    env = gym.make(envs[env_name])  # Skip using D4RL class because action_space.sample() does not exist
    print(f"For env: {env_name} with data: {envs[env_name]}")
    expert_mean, random_agent_mean = _get_env_baseline(env)
    data[env_name] = dict(expert_mean=expert_mean, random_agent_mean=random_agent_mean )
  return data