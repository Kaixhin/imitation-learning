#!/bin/bash

envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "PUGAIL" "RED")
[[ " ${algos[@]} " =~ " $1 " ]] && echo "algorithm=$1" || { echo "invalid input, first input must be one of ${algos[*]} "; exit 1; }

if [[ $# -eq 2 ]]; then
[[ " ${envs[@]} " =~ " $2 " ]] && echo "environment=$2" || { echo "invalid input, second input must be empty or one of ${envs[*]} "; exit 1; }

    echo "running python3 main.py -m algorithm=$1/$2 hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic"
    python3 main.py -m algorithm=$1/$2 hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic
else
  for env in ${envs[@]}; do
      echo "running python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic"
      python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic
  done
fi