#!/usr/bin/env python
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import yaml
import math

from .utils import find_optimal_result, process_test_data, load_data, load_all_data, scan_folder_trajectories, get_trajectory_subfolder, read_hydra_configs, str_float_format, ALGORITHMS, ENVS, ENVS_DATA, get_all_env_baseline 
 
"""A simple plotting script. Change the global variable depending on setting"""

colors = ['green', 'tab:blue', 'tab:purple', 'tab:cyan', 'tab:orange', 'tab:red', 'tab:brown']
fontsize=14
ENV_NAMES = dict()
ENV_NAMES['ant'] = 'Ant'; ENV_NAMES['halfcheetah'] = 'HalfCheetah';
ENV_NAMES['hopper'] = 'Hopper'; ENV_NAMES['walker2d'] = "Walker2D"

folder_dateformat ="%m-%d_%H-%M-%S"

def plot_environment_result(data, ax, env, normalization_data):
    ax.set_title(ENV_NAMES[env], fontsize=fontsize)
    pre_print = "For env: " + env
    print(pre_print)
    for alg, col in zip(ALGORITHMS, colors):
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
            if normalization_data:
              max_reward, min_reward = normalization_data[env]['expert_mean'], normalization_data[env]['random_agent_mean']
              mean = (mean - min_reward) / (max_reward - min_reward)
              low_fill, top_fill = (mean - std_err - min_reward) / (max_reward - min_reward), (mean + std_err - min_reward) / (max_reward - min_reward)
            ax.plot(x, mean, col, label=alg)
            ax.fill_between(x, low_fill, top_fill, alpha=0.3)
            result = ' ' * len(pre_print) + alg + ", Result: " + "{:.2f}".format(mean[-1]) + " +/- " + "{:.2f}".format(std[-1])
            print(result)
        except KeyError as e:
            print('\t no ' + alg +' data for env:' + env)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=40)
    ax.margins(x=0.0, tight=True)



def create_all_plots(x, y, save_fig=False, date_from=None, date_to=None):
    fig, ax = plt.subplots(x, y, sharex=True)
    ax = ax.reshape(-1)
    #fig.tight_layout()
    #fig.set_size_inches((11, 8.5), forward=False) # A4 paper size apparently. INCHES, UGH
    fig.set_size_inches((14, 6), forward=False) # A4 paper size apparently. INCHES, UGH
    data = load_all_data(date_from, date_to)
    for env, axis in zip(ENVS, ax):
        env_data = data[env]
        if env_data: #Empty if data couldn't be loaded
            plot_environment_result(env_data, axis, env)
    # Plot the Baseline results
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(ALGORITHMS)+1, fontsize=fontsize)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Steps (x $10^6$)", fontsize=fontsize)
    plt.ylabel("Mean reward", fontsize=fontsize, labelpad=12)
    if save_fig:
        fig.savefig('./figures/result_fig.png', dpi=500, bbox_inches='tight')
    plt.subplot_tool() #Uncomment when you want to play with the margins between subplots
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
    for alg in ALGORITHMS:
        alg_dict[alg] = relevant_param(alg)
    return alg_dict


def plot_hyperparam_alg(ax, alg):
    relevant_dict = relevant_param(alg)
    ylabel = list(relevant_dict.keys())
    xdata = []
    r1 = np.arange(len(ylabel))
    barWidth = 0.20
    color = ['r', 'g', 'b', 'm']
    for i, env in enumerate(ENVS):
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
    for i, alg in enumerate(ALGORITHMS):
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
    for i, env in enumerate(ENVS):
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
    for i, alg in enumerate(ALGORITHMS):
        alg_ax = ax[i]
        plot_hyperparam_alg(alg_ax, alg)
        if alg == "BC":
            alg_ax.legend()
    ax[-1].axis('off')
    plt.show()


def plot_hyperparam(ax, alg, param):
    relevant_dict = relevant_param(alg)
    hyperparam_range = relevant_dict[param]
    r1 = np.arange(len(ENVS))
    xdata = []
    for i, env in enumerate(ENVS):
        data = read_hyperparam(alg, env)
        hyperparam_value = data[param]
        xdata.append(hyperparam_range.index(hyperparam_value) + 1)
    barlist = ax.bar(r1, xdata, edgecolor='white')
    for i, c in enumerate(['r', 'g', 'b', 'm']):
        barlist[i].set_color(c)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=30)
    ax.yaxis.set_tick_params(pad=0)
    ax.set_xticklabels([])
    ax.set_yticks(range(len(hyperparam_range)+1))
    if 'learning_rate' in param:
        hyperparam_range = ['3e-5', '3e-4']
    ax.set_yticklabels(['', *hyperparam_range] )
    ax.margins(x=0.0, y=0.0, tight=True)
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
PARAM_TITLE['self_similarity'] = "Self-similarity"

def create_hyperparam_plot(x, y, save_fig=False):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 8)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
    #fig.tight_layout()
    l = list(relevant_param('AIRL').keys())
    # Reordered algorithms in less hyperparam order
    #algorithms = ['BC', 'GMMIL', 'RED', 'DRIL', 'AIRL', 'FAIRL', 'GAIL'][::-1]
    ylabel = [l[5], l[4], l[6], l[7], l[0], l[1], l[2], l[3]]
    for i, alg in enumerate(ALGORITHMS):
        relevant_dict = relevant_param(alg)
        alg_ax = ax[i]
        for j, param in enumerate(ylabel):
            if param in relevant_dict.keys():
                set_legend = False
                barlist = plot_hyperparam(alg_ax[j], alg, param)
                if not j and not i:
                    ENV = [ENV_NAMES[env] for env in ENVS]
                    fig.legend(barlist, ENV, loc='lower center', ncol=len(ENVS))
            if param not in relevant_dict.keys():
                alg_ax[j].axis('off')
                #alg_ax[j].get_xaxis().set_ticks([])
                #alg_ax[j].get_yaxis().set_ticks([])
            if not i:
                alg_ax[j].set_title(PARAM_TITLE[param])
        if alg == "GMMIL":
            alg_ax[-1].axis('on')
            plot_hyperparam(alg_ax[-1], alg, 'self_similarity')
            alg_ax[-1].set_title(PARAM_TITLE['self_similarity'])

        alg_ax[0].set_ylabel(alg)
    if save_fig:
        fig.set_size_inches((8.5, 11), forward=False)
        fig.savefig('./figures/hyperparam.png', dpi=500, bbox_inches='tight')
    #plt.subplot_tool() #Uncomment when you want to play with the margins between subplots
    plt.show()

def plot_trajectory_opt_data(alg, trajectory, output_folder='./outputs/', folder_prefix='all_sweep_', date_from=None, date_to=None, normalization_data=None):
    data_folder = os.path.join(output_folder, folder_prefix+alg)
    subfolder = get_trajectory_subfolder(alg, data_folder, trajectory)
    optimal_run_num = os.path.basename(os.path.dirname(find_optimal_result(subfolder)))
    sweep_folders = [path for path, dirs, files in os.walk(subfolder) if 'all.log' in files]
    fig, axes = plt.subplots(len(sweep_folders)//5, 5)
    axes = axes.reshape(-1)
    once, key_order, plot_data, fig_title = True, [], [], ""
    for sweep_folder in sweep_folders:
      subplot_data, env_means = dict(), []
      for env, col in zip(ENVS, colors):
        sweep_env_folder = os.path.join(sweep_folder, env)
        data = torch.load(os.path.join(sweep_env_folder, 'metrics.pth'))
        x, mean, std_err, _ = process_test_data([data])
        if 'BC' in alg:
          x = np.linspace(0.0, 10.0 , num=100)
          mean, std_err = mean.repeat(100), std_err.repeat(100)
        if normalization_data:
          max_reward, min_reward = normalization_data[env]['expert_mean'], normalization_data[env]['random_agent_mean']
          mean = (mean - min_reward) / (max_reward - min_reward)
          low_fill, top_fill = (mean - std_err - min_reward) / (max_reward - min_reward), (mean + std_err - min_reward) / (max_reward - min_reward)
        else:
          low_fill, top_fill = mean - std_err, mean + std_err 
        import pdb; pdb.set_trace()
        env_means.append(np.sum(mean[-5:]/5))
        subplot_data[env] = dict(x=x, mean=mean, low_fill=low_fill, top_fill=top_fill)
      subplot_data['score'] = np.median(env_means)
      hydra_conf_folder = os.path.join(sweep_folder, '.hydra')
      hydra_conf = read_hydra_configs(hydra_conf_folder, exclude_key=True)
      if once:
        once = False
        key_order = [key for key in hydra_conf['overrides'].keys()]
        figure_text = ', '.join(key_order)
        fig_title = f"Trajectory: {trajectory} [ {figure_text} ]"
        #fig.text(0.0, 0.7, figure_text, fontsize=fontsize)
      txt =', '.join([str_float_format(hydra_conf['overrides'][key]) for key in key_order])
      sweep_num = os.path.basename(os.path.normpath(sweep_folder))
      optimal=False
      if sweep_num == optimal_run_num:
        optimal=True
      subplot_data['title'] = f"{sweep_num}: [{txt} ]"
      subplot_data['optimal']= optimal
      plot_data.append(subplot_data)
    plot_data = sorted(plot_data, key=lambda x: x['score'], reverse=True)
    once = True 
    for data, ax in zip(plot_data, axes):
      for env, col in zip(ENVS, colors):
        x, mean, low_fill, top_fill = data[env]['x'], data[env]['mean'], data[env]['low_fill'], data[env]['top_fill'], 
        ax.plot(x, mean, col, label=env)
        ax.fill_between(x, low_fill, top_fill, alpha=0.3)
      if once:
        fig.legend()
        once = False
        fig.suptitle(fig_title)
      if data['optimal']:
        ax.set_facecolor('xkcd:light light green') # We all like xkcd
      ax.set_title(data['title'], fontsize='medium')
      ax.get_xaxis().set_ticks([])
      ax.set_ylim([-0.3, 1.3])




def plot_all_trajectory_opt_data(alg, output_folder='./outputs/', folder_prefix='all_sweep_', date_from=None, date_to=None, save_fig=False):
  if alg=='all':
    for algo in ALGORITHMS:
      plot_all_trajectory_opt_data(algo, output_folder=output_folder, folder_prefix=folder_prefix, date_from=date_from, date_to=date_to, save_fig=save_fig)
  data_folder = os.path.join(output_folder, folder_prefix+alg)
  trajectory_nums = scan_folder_trajectories(data_folder)
  all_envs = ENVS_DATA
  normalization_data = get_all_env_baseline(all_envs)
  trajectory_nums = sorted(trajectory_nums, reverse=True)
  if not trajectory_nums:
    print("Didnt found any data with trajectories; defaulting to traj -1 (first found)")
    trajectory_nums = [-1]
  print(f"Found data with trajectories: {trajectory_nums}")
  for tr in trajectory_nums:
    plot_trajectory_opt_data(alg, tr, output_folder=output_folder, folder_prefix=folder_prefix, date_from=None, date_to=None, normalization_data=normalization_data)
    fig = plt.gcf()
    fig.text(0.06, 0.5, "Normalized Reward", ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, "Steps", ha='center', va='center')
    if save_fig:
      fig.show()
      fig.set_size_inches((16,9), forward=False)
      fig_filename = os.path.join(data_folder, f"{alg}_traj{tr}.png")
      fig.savefig(fig_filename, dpi=500 )
    else:
      fig.show()
      fig.set_size_inches((16,9), forward=False)
      plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--n-col', type=int, default=2)
    parser.add_argument('--n-row', type=int, default=2)
    parser.add_argument('--plot', choices=['hyperparam', 'result', 'trajectories'])
    parser.add_argument('--alg', choices=ALGORITHMS+['all'])
    parser.add_argument('--save-fig', action='store_true', default=False)
    parser.add_argument('--date-from', type=str, default=None)
    parser.add_argument('--par-file', action='store_true', default=False)
    parser.add_argument('--data-folder', type=str, default='./outputs/')
    args = parser.parse_args()
    if args.plot == 'trajectories':
      print(f"reading data from: {args.data_folder}")
      plot_all_trajectory_opt_data(args.alg, output_folder=args.data_folder, folder_prefix='all_sweep_', save_fig=args.save_fig)
    elif args.plot == 'hyperparam':
      create_hyperparam_plot(args.n_row, args.n_col, args.save_fig)
    elif args.plot == 'result':
      create_all_plots(args.n_row, args.n_col, args.save_fig, date_from=args.date_from, par_sweep=args.par_file)
