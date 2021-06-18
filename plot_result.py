#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
import yaml
import math

from environments import D4RL_ENV_NAMES
"""A simple plotting script. Change the global variable depending on setting"""
sns.set(style='white')

algorithms = ['BC', 'GAIL', 'AIRL', 'FAIRL', 'GMMIL', 'RED', 'DRIL']
envs = ['ant', 'halfcheetah', 'hopper', 'walker2d']
colors = ['tab:olive', 'tab:blue', 'tab:purple', 'tab:cyan', 'tab:orange', 'tab:red', 'tab:brown']
output_folder = './outputs/' # Folder with all the seed sweeper results
seed_prefix = 'seed_sweeper_' #prefix of all seed sweeper folders


# Baseline results
BASELINE = dict()  # [mean, std]
BASELINE['ant'] = [570.80, 104.82]; BASELINE['halfcheetah'] = [787.35, 104.31]
BASELINE['hopper'] = [1078.36, 325.52]; BASELINE['walker2d'] = [1106.68, 417.79]
ENV_NAMES = dict()
for env, d4rlname in zip(envs, D4RL_ENV_NAMES):
    ENV_NAMES[env] = d4rlname


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
    return np.array(x), mean_of_means, std_err, std_of_means

def plot_env_baseline(ax, env):
    x = np.linspace(0.0, 2.0 , num=100)
    mean, std = BASELINE[env]
    std_err = std / np.sqrt(5) # Hardcoded I know
    mean = np.repeat(mean, 100)
    std_err = np.repeat(std_err, 100)
    ax.plot(x, mean, 'k', label='Dataset')
    ax.fill_between(x, mean - std_err, mean + std_err, color='k', alpha=0.1)


def plot_environment_result(data, ax, env):
    ax.set_title(env)
    pre_print = "For env: " + env
    print(pre_print)
    plot_env_baseline(ax, env)
    for alg, col in zip(algorithms, colors):
        try:
            metric = data[alg]
            x, mean, std_err, std = process_test_data(metric)
            if alg == "BC":
                x = np.multiply(x, np.linspace(0.0, 100.0, num=100))
                mean = np.repeat(mean, 100)
                std_err = np.repeat(std_err, 100)
                std = np.repeat(std, 100)

            x_pow = math.floor(math.log(x[-1], 10))
            x = x / 10**x_pow
            ax.plot(x, mean, col, label=alg)
            ax.fill_between(x, mean - std_err, mean + std_err, color=col, alpha=0.3)
            result = ' ' * len(pre_print) + alg + ", Result: " + "{:.2f}".format(mean[-1]) + " +/- " + "{:.2f}".format(std[-1])
            print(result)
        except Exception as e:
            print('\t no ' + alg +' data for env:' + env)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=40)




def create_all_plots(x, y, save_fig=False):
    fig, ax = plt.subplots(x, y, sharex=True)
    ax = ax.reshape(-1)
    #fig.tight_layout()
    fig.set_size_inches((11, 8.5), forward=False) # A4 paper size apparently. INCHES, UGH
    data = load_all_data()
    for env, axis in zip(envs, ax):
        env_data = data[env]
        if env_data: #Empty if data couldn't be loaded
            plot_environment_result(env_data, axis, env)
    # Plot the Baseline results
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(algorithms)+1)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r"Steps (x $10^6$)")
    plt.ylabel("Mean reward")
    if save_fig:
        fig.savefig('./figures/result_fig.png', dpi=500, bbox_inches='tight')
    plt.show()

# Below are hyper param plotting code


def read_hyperparam(alg, env):
    param_root = './conf/algorithm'
    param_file = os.path.join(param_root, alg+'/'+env+'.yaml')
    with open(param_file, 'r') as f:
        param_dict = yaml.safe_load(f)
    return param_dict


def relevant_param(alg):
    param_root = './conf/hyperparam_opt'
    param_file = os.path.join(param_root, alg+'.yaml')
    with open(param_file, 'r') as f:
        raw_dict = yaml.safe_load(f)
    relevant_dict = raw_dict['hydra']['sweeper']['ax_config']['params']
    output_dict = dict()
    for key in relevant_dict.keys():
        opt_range = relevant_dict[key]['values']
        output_dict[key] = opt_range
    return output_dict

def get_relevant_param():
    alg_dict = dict()
    for alg in algorithms:
        alg_dict[alg] = relevant_param(alg)
    return alg_dict


def plot_hyperparam_alg(ax, alg):
    relevant_dict = relevant_param(alg)
    ylabel = list(relevant_dict.keys())
    xdata = []
    r1 = np.arange(len(ylabel))
    barWidth = 0.20
    color = ['r', 'g', 'b', 'm']
    for i, env in enumerate(envs):
        data = read_hyperparam(alg, env)
        xdata = []
        for key in ylabel:
            hyperparam_range = relevant_dict[key]
            hyperparam_value = data[key]
            xdata.append( hyperparam_range.index(hyperparam_value) + 1)
        r = [x + i*barWidth for x in r1]
        ax.bar(r, xdata, color=color[i], width=barWidth, edgecolor='white', label=env)
    ax.set_xticks([r + barWidth for r in range(len(ylabel))])
    ax.set_xticklabels(ylabel)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.set_ylabel(alg)
    ax.set_yticklabels([])

def plot_hyperparam_env(ax, env):
    ylabel = list(relevant_param('AIRL').keys())
    r1 = np.arange(len(ylabel))
    barWidth = 0.125
    for i, alg in enumerate(algorithms):
        relevant_dict = relevant_param(alg)
        data = read_hyperparam(alg, env)
        xdata = []
        for key in ylabel:
            if key in relevant_dict.keys():
                hyperparam_range = relevant_dict[key]
                hyperparam_value = data[key]
                xdata.append( hyperparam_range.index(hyperparam_value) + 1)
            else:
                xdata.append(0)
        r = [x + i*barWidth for x in r1]
        ax.bar(r, xdata, color=colors[i], width=barWidth, edgecolor='white', label=alg)
    ax.set_xticks([r + barWidth for r in range(len(ylabel))])
    ax.set_ylabel(env)
    ax.set_yticklabels([])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


def create_env_hyperparam_plot():
    fig, (ax1, ax2) = plt.subplots(2, 2)
    ax = [ax1[0], ax1[1], ax2[0], ax2[1]]
    fig.tight_layout()
    ylabel = list(relevant_param('AIRL').keys())
    for i, env in enumerate(envs):
        env_ax = ax[i]
        plot_hyperparam_env(env_ax, env)
        ax[i].set_xticklabels(ylabel)
    ax[-1].legend()
    plt.show()


def create_alg_hyperparam_plot():
    #fig, ax = plt.subplots(7)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 2)
    ax = [*ax1, *ax2, *ax3, *ax4]
    fig.tight_layout()
    for i, alg in enumerate(algorithms):
        alg_ax = ax[i]
        plot_hyperparam_alg(alg_ax, alg)
        if alg == "BC":
            alg_ax.legend()
    ax[-1].axis('off')
    plt.show()


def plot_hyperparam(ax, alg, param):
    relevant_dict = relevant_param(alg)
    hyperparam_range = relevant_dict[param]
    r1 = np.arange(len(envs))
    xdata = []
    for i, env in enumerate(envs):
        data = read_hyperparam(alg, env)
        hyperparam_value = data[param]
        xdata.append(hyperparam_range.index(hyperparam_value) + 1)
    barlist = ax.bar(r1, xdata, edgecolor='white')
    for i, c in enumerate(['r', 'g', 'b', 'm']):
        barlist[i].set_color(c)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=30)
    ax.set_xticklabels([])
    ax.set_yticks(range(len(hyperparam_range)+1))
    if 'learning_rate' in param:
        hyperparam_range = ['3e-5', '3e-4']
    ax.set_yticklabels(['', *hyperparam_range])
    return barlist

PARAM_TITLE = dict()
PARAM_TITLE['imitation_epochs'] = "Imitation epochs"
PARAM_TITLE['il_learning_rate'] = "Imitation Learning rate"
PARAM_TITLE['imitation_replay_size'] = "Imitation replay size"
PARAM_TITLE['r1_reg_coeff'] = r'$R_1$ gradient penalty'
PARAM_TITLE['batch_size'] = 'Rollout buffer size'
PARAM_TITLE['agent_learning_rate'] = "Agent learning rate"
PARAM_TITLE['ppo_epochs'] = "PPO iterations"
PARAM_TITLE['entropy_loss_coeff'] = r"Entropy loss coeff $c_2$"

def create_hyperparam_plot(x, y, save_fig=False):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 8) #,figsize=(7.5, 15))
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    fig.tight_layout(w_pad=0.0, h_pad=0.1)
    #fig.set_size_inches(11.69, 8.27) # A4 paper size apparently. INCHES, UGH
    l = list(relevant_param('AIRL').keys())
    # Reordered algorithms in less hyperparam order
    #algorithms = ['BC', 'GMMIL', 'RED', 'DRIL', 'AIRL', 'FAIRL', 'GAIL'][::-1]
    ylabel = [l[5], l[4], l[6], l[7], l[0], l[1], l[2], l[3]]
    for i, alg in enumerate(algorithms):
        relevant_dict = relevant_param(alg)
        alg_ax = ax[i]
        for j, param in enumerate(ylabel):
            if param in relevant_dict.keys():
                set_legend = False
                barlist = plot_hyperparam(alg_ax[j], alg, param)
                if not j and not i:
                    fig.legend(barlist, envs, loc=(0.9, 0.3))
            if param not in relevant_dict.keys():
                alg_ax[j].axis('off')
                #alg_ax[j].get_xaxis().set_ticks([])
                #alg_ax[j].get_yaxis().set_ticks([])
            if not i:
                alg_ax[j].set_title(PARAM_TITLE[param])
        if alg == "GMMIL":
            alg_ax[-1].axis('on')
            plot_hyperparam(alg_ax[-1], alg, 'self_similarity')
            alg_ax[-1].set_title('self_similarity')

        alg_ax[0].set_ylabel(alg)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DCNNAE_for_selfdetection')
    parser.add_argument('--n-col', type=int, default=2)
    parser.add_argument('--n-row', type=int, default=2)
    parser.add_argument('--plot-hyperparam', action='store_true', default=False)
    parser.add_argument('--save-fig', action='store_true', default=False)
    args = parser.parse_args()
    if args.plot_hyperparam:
        create_hyperparam_plot(args.n_row, args.n_col, args.save_fig)
    else:
        create_all_plots(args.n_row, args.n_col, args.save_fig)
