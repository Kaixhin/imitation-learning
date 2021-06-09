#!/bin/bash

envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "PUGAIL" "RED")
[[ " ${algos[@]} " =~ " $1 " ]] && echo "ok" || { echo "invalid input, must be one of ${algos[*]} "; exit 1; }
log_file="full_algorithm.log"
for env in ${envs[@]}; do
    echo "running python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic"
    python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic
done
