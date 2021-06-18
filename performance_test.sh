#!/bin/bash

env='hopper'
algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "RED")

for algo in ${algos[@]}; do
    if [[ $# -eq 1 ]]; then
      #echo "running python3 main.py algorithm=$algo/$env check_memory_usage=true hyperparam_opt=empty hydra/sweeper=basic"
      #python3 main.py algorithm=$algo/$env check_memory_usage=true hyperparam_opt=empty hydra/sweeper=basic
      echo "running mprof run main.py algorithm=$algo/$env  hyperparam_opt=empty hydra/sweeper=basic "
      mprof run main.py algorithm=$algo/$env  hyperparam_opt=empty hydra/sweeper=basic
    else
      echo "running python3 main.py algorithm=$algo/$env steps=1000000 check_time_usage=true hyperparam_opt=empty hydra/sweeper=basic"
      python3 main.py algorithm=$algo/$env steps=1000000 check_time_usage=true hyperparam_opt=empty hydra/sweeper=basic
    fi
done
