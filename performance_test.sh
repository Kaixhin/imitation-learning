#!/bin/bash

env='hopper'
algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "RED")
count=5

for algo in ${algos[@]}; do
    if [[ $# -eq 1 ]]; then
    for i in $(seq $count); do
        rm ./mprofile_*.dat # clean mprof  
        echo "running mprof run main.py algorithm=$algo/$env  hyperparam_opt=empty hydra/sweeper=basic "
        mprof run main.py algorithm=$algo/$env  hyperparam_opt=empty hydra/sweeper=basic
        awk 'BEGIN{a=0}{if ($2 > 0+a) a=$2} END{print a}' mprofile_*.dat >> "./${algo}_${env}_time.txt"
    done
    else
      echo "running python3 main.py algorithm=$algo/$env steps=1000000 check_time_usage=true hyperparam_opt=empty hydra/sweeper=basic"
      python3 main.py -m algorithm=$algo/$env steps=1000000 check_time_usage=true seed=1,2,3,4,5 hyperparam_opt=empty hydra/sweeper=basic
    fi
done
