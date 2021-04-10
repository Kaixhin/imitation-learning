import argparse
import seaborn
from utils import MetricSaver
parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--save', action='store_true', default=False, help='Store the results of analytics rather than plottng it')
parser.add_argument('--save-folder', type=str, default='./analytics_result/' )
import os
args = parser.parse_args()


class Analyzer():
    def __init__(self, filename=None, save_result=False):
        self.save_result = save_result
        self.data_batch = dict()
        self.data = None
        self.data_batch_env = None
        if filename:
            self.data = MetricSaver(load_file=filename)

    def load_all_env_results(self, env='hopper', hydra_output_folder='./outputs/'):
        self.data_batch_env = env
        dirs = [name.startswith(env) for name in os.listdir(hydra_output_folder)]
        if not dirs:
            raise ValueError('No directories that starts with ' + env + 'in ' + hydra_output_folder)

    def plot_all_env_results(self, result_folder=None):
        raise NotImplemented()


if __name__ == '__main__':
    raise NotImplemented()