#!/bin/bash

env='hopper'
algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "RED")

for algo in ${algos[@]}; do
    echo "running python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic"
    if [[ $# -eq 1 ]]; then
      echo "running python3 main.py algorithm=$algo/$env check_memory_usage=true hyperparam_opt=empty hydra/sweeper=basic"
      python3 main.py algorithm=$algo/$env check_memory_usage=true hyperparam_opt=empty hydra/sweeper=basic
    else
      echo "running python3 main.py algorithm=$algo/$env check_time_usage=true hyperparam_opt=empty hydra/sweeper=basic"
      python3 main.py algorithm=$algo/$env check_time_usage=true hyperparam_opt=empty hydra/sweeper=basic
    fi
done
