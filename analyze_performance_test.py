#!/usr/bin/env python
import sys, os
import glob
import numpy as np

def process_memory_data():
    res = ""
    data_files = glob.glob('./*hopper_time.txt')
    print(data_files)
    for name in data_files:
        alg_name = name.split('_')[0].split('/')[-1]
        with open(name, 'r') as file:
            data = file.read()
            data_list = data.split('\n')
            float_data = []
            for d in data_list:
                try:
                    d_float = float(d)
                    float_data.append(d_float)
                except ValueError as e:
                    continue
            npdata = np.array(float_data)
            mean, std = npdata.mean(), npdata.std()
            txt = "For " + alg_name + " result: " + "{:.2f}".format(mean) + " +/- " + "{:.2f}".format(std)
            print(txt)
            res = res + txt + '\n'
    with open('memory_performance_result.txt', 'w') as file:
        file.write(res)

def process_time_data(result_folder='./outputs', folder_prefix='seed_sweeper_hopper_'):
    glob_folder_name = os.path.join(result_folder, folder_prefix+"*")
    multirun_num = ['0', '1', '2','3', '4']
    result_text = ""
    for folder in glob.glob(glob_folder_name):
        algo_name = folder.split('_')[-1]
        pretrain = []
        train = []
        for n in multirun_num:
            data_folder = os.path.join(folder, n)
            pretrain_data = os.path.join(data_folder, "pre_training_time.txt")
            train_data = os.path.join(data_folder, "training_time.txt")
            if os.path.isfile(pretrain_data):
                with open(pretrain_data, 'r') as file:
                    data = float(file.read())
                    pretrain.append(data)
            if os.path.isfile(train_data):
                with open(train_data, 'r') as file:
                    data = float(file.read())
                    train.append(data)
        pt_mean, pt_std, t_mean, t_std = "-", "", "-", ""
        if pretrain:
            np_pt = np.array(pretrain)
            pt_mean, pt_std = np_pt.mean(), np_pt.std()
        if train:
            np_t = np.array(train)
            t_mean, t_std  = np_t.mean(), np_t.std()
        txt = "For " + algo_name + " Pretraining: " + "{:.2f}".format(pt_mean) + " +/- " + "{:.2f}".format(pt_std) + ".  Training: " + "{:.2f}".format(t_mean) + " +/- " + "{:.2f}".format(t_std)
        print(txt)
        result_text += txt + '\n'
    with open('timing_performance_result.txt', 'w') as file:
        file.write(result_text)


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) > 1:
        print(argv)
        if "mem" in argv[1]:
            print('hello')
            process_memory_data()
        elif "time" in argv[1]:
            process_time_data()
    else:
        process_memory_data()
        process_time_data()
