#!/bin/bash

envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
output_log_file="all_experiments.log"
for e in ${envs[@]}; do
  ./run_all_algorithms.sh $e  | tee -a "$output_log_file"
done
