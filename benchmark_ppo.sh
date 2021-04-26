#!/bin/bash

envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
output_log_file="all_experiments.log"
for e in ${envs[@]}; do
  python3 main.py algorithm="PPO" environment=$e  | tee -a "$output_log_file"
done
