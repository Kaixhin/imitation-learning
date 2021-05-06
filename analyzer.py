import argparse
import os

import seaborn


parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--save', action='store_true', default=False, help='Store the results of analytics rather than plottng it')
parser.add_argument('--save-folder', type=str, default='./analytics_result/' )
args = parser.parse_args()


def compare_sweeper_result(env_name, algo_name):
    raise NotImplemented


def plot_training_loss(env_name, algo_name):
    raise NotImplemented


def plot_evaluation_loss(env_name, algo_name):
    raise NotImplemented


def plot_IRL_loss(env_name, algo_name):
    raise NotImplemented


if __name__ == '__main__':
    raise NotImplemented()
