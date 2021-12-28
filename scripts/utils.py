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
ALGORITHMS = ['SAC', 'BC', 'GAIL','GMMIL', 'RED', 'DRIL', 'SQIL', 'AdRIL']
ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']
ENVS_DATA = dict(ant='ant-expert-v2', halfcheetah='halfcheetah-expert-v2', hopper='hopper-expert-v2', walker2d='walker2d-expert-v2')
HYDRA_CONF = ['config', 'overrides', 'hydra']
DEFAULT_DATEFORMAT="%m-%d_%H-%M-%S"

EXCLUDED_KEYS=['hyperparam_opt', 'imitation.trajectories', 'imitation.subsample']
def str_float_format(value):
  if type(value) is float:
    e_format = "{:.0e}".format(value)
    g_format = "{:.4g}".format(value)
    shortest_format = g_format if len(g_format) < len(e_format) else e_format
  return shortest_format

def read_hydra_yaml(file_name: str, exclude_keys=None):
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
    if exclude_keys:
       if key not in exclude_keys: dict_data[key] = value
    else: dict_data[key] = value
  return dict_data

def read_hydra_configs(folder_name: str, exclude_key=False):
  hydra_conf = dict()
  for conf_name in HYDRA_CONF:
    exclude_keys = EXCLUDED_KEYS if exclude_key else None
    hydra_conf[conf_name] = read_hydra_yaml(os.path.join(folder_name, conf_name+'.yaml'), exclude_keys=exclude_keys)
  return hydra_conf

def is_right_datetime(str_input, date_format=DEFAULT_DATEFORMAT):
  try:
    datetime.strptime(str_input, date_format)
    return True
  except ValueError as e:
    pass
  return False
  

def filter_datefolder(folder_name, date_from=None, date_to=None, date_format=DEFAULT_DATEFORMAT):
  """Input should be the folder containing the datetime folders. Given input folder_name, returns a list of all the subfolders with date_format between given date_from/date_to. 
     If date_from is None, take all folders up to date_to. If date_to is None, take all folders up from date_from. if both are None, just give back all subfolders"""
  assert os.path.isdir(folder_name)
  dirname, subdirname = [(path, dirs) for path, dirs, files in os.walk(folder_name) if all([is_right_datetime(d, date_format) for d in dirs])][0] # Assumes folder with datetime subfolder only contain datetime named subfolders.
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
  """Scan folder with data"""
  subfolders = [subdir[0] for subdir in os.walk(folder) if 'optimization_results.yaml' in subdir[2]]
  trajectory_nums = []
  for subfolder in subfolders:
    hydra_dict = read_hydra_yaml(os.path.join(subfolder, 'optimization_results.yaml'))
    for key, value in hydra_dict['ax'].items():
      if 'trajectories' in key:
        trajectory_nums.append(int(value))
  return trajectory_nums 


def get_trajectory_subfolder(alg, folder, num_trajectory, include_datetime=False, date_format=DEFAULT_DATEFORMAT):
  """Returns the subfolder in input:folder with 'optimization_results.yaml' containing given(input:num_trajectory) trajectories value. Takes the first it find."""
  subfolders = [subdir[0] for subdir in os.walk(folder) if 'optimization_results.yaml' in subdir[2]] 
  for subfolder in subfolders:
    if num_trajectory < 0:
      return subfolder # if no trajectories specified return first found
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
    expert_mean, random_agent_mean = _get_env_baseline(env)
    data[env_name] = dict(expert_mean=expert_mean, random_agent_mean=random_agent_mean)
  return data

def find_optimal_result(sweep_folder):
  """Because Ax sweeper gives us the optimal parameters after sweep, but doesnt tell us which folder (num sweep) it is!"""
  optimal_result_filename = os.path.join(sweep_folder, 'optimization_results.yaml')
  optimal_param_dict = read_hydra_yaml(optimal_result_filename)['ax']
  result_folders = [path for path, dirs, files in os.walk(sweep_folder) if 'overrides.yaml' in files]
  for dot_hydra_folder in result_folders:
    overrides_dict = read_hydra_configs(dot_hydra_folder)['overrides']
    match=True
    for key, value in optimal_param_dict.items():
      if key in overrides_dict.keys():
        if overrides_dict[key] != value:
          match=False
          break
      else: match=False
    if match:
      print(f"Folder: {dot_hydra_folder} contains the optimal run")
      optimal_param_str = ','.join([f"{key}={value}" for key, value in optimal_param_dict.items()])
      print(optimal_param_str)
      return dot_hydra_folder
  print("Error: No matching folder found")
  return False

def trim_metrics(folder='./outputs/somefolder/', trim_value=100):
  metrics = [os.path.join(path, 'metrics.pth') for path, dirs, files in os.walk(folder) if 'metrics.pth' in files]
  for metric in metrics:
    data = torch.load(metric)
    trimmed_data = dict()
    for key in data.keys():
      if not 'test_' in key: #Don't trim test, as this is needed for final eval.
        trimmed_data[key] = data[key][::trim_value]
      else:
        trimmed_data[key] = data[key]
    torch.save(trimmed_data, metric)
