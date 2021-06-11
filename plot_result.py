from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os

from environments import D4RL_ENV_NAMES
"""A simple plotting script. Change the global variable depending on setting"""
sns.set(style='white')

algorithms = ['AIRL', 'BC', 'DRIL', 'FAIRL', 'GAIL', 'GMMIL', 'RED']
envs = ['ant', 'halfcheetah', 'hopper', 'walker2d']
colors = ['b', 'g', 'k', 'c', 'm', 'y', 'r']

ENV_NAMES = dict()
for env, d4rlname in zip(envs, D4RL_ENV_NAMES):
    ENV_NAMES[env] = d4rlname

output_folder = './outputs/' # Folder with all the seed sweeper results
seed_prefix = 'seed_sweeper_' #prefix of all seed sweeper folders

def crease_plots(subplots=True):
    if subplots:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax = [ax1, ax2, ax3, ax4]
    plot_dict =  dict()
    plot_dict['fig'] = fig
    for e, a in zip(envs, ax):
        plot_dict[e] = a
    return plot_dict


def load_data(env, alg):
    seed_folder_name = seed_prefix + env + '_' + alg
    seed_folder = os.path.join(output_folder, seed_folder_name)
    assert os.path.isdir(seed_folder)
    seeds = [x[1] for x in os.walk(seed_folder)][0]
    seeds = [os.path.join(seed_folder, x) for x in seeds]
    data = []
    for s in seeds:
        data.append(torch.load(os.path.join(s, 'metrics.pth')))
    return data


def load_all_data():
    data = dict()
    for env in envs:
        metrics = dict()
        for alg in algorithms:
            try:
                metrics[alg] = load_data(env, alg)
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
    return x, mean_of_means, std_err

def plot_environment_result(data, ax, env):
    ax.set_title(ENV_NAMES[env])
    for alg, col in zip(algorithms, colors):
        try:
            metric = data[alg]
            x, mean, std_err = process_test_data(metric)
            ax.plot(x, mean, col, label=alg)
            ax.fill_between(x, mean - std_err, mean + std_err, color=col, alpha=0.3)
        except Exception as e:
            print('\t no ' + alg +' data for env:' + env)
    ax.legend()




def create_all_plots(subplots=True):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
    ax = [ax1, ax2, ax3, ax4]
    data = load_all_data()
    for env, axis in zip(envs, ax):
        env_data = data[env]
        if env_data: #Empty if data couldn't be loaded
            plot_environment_result(env_data, axis, env)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Steps")
    plt.ylabel("Mean reward")
    plt.show()

if __name__ == '__main__':
    create_all_plots()