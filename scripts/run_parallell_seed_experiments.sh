#!/bin/bash

date
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
SEEDS_PER_PROCESS=2
TOTAL_SEEDS=10
LOOP=($(seq 0 $SEEDS_PER_PROCESS $((TOTAL_SEEDS - SEEDS_PER_PROCESS))))
envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
algos=("BC"  "DRIL" "GAIL" "GMMIL"  "RED", "SAC" "SQIL")
[[ " ${algos[@]} " =~ " $1 " ]] && echo "algorithm=$1" || { echo "invalid input, first input must be one of ${algos[*]} "; exit 1; }

for env in ${envs[@]}; do
  #echo "running python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=1,2,3,4,5 hydra/sweeper=basic"
  for seed_num in ${LOOP[@]}; do
  seed_end=$((seed_num + SEEDS_PER_PROCESS - 1))
  seeds="$(seq -s ',' $seed_num 1 $seed_end)"
  #python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=$seeds hydra/sweeper=basic hydra.sweep.dir=./outputs/par_seed_sweeper_$1_$env/$seed_num > log/$1_${env}_$seed_num.txt &
  echo "python3 main.py -m algorithm=$1/$env hyperparam_opt=empty seed=$seeds hydra/sweeper=basic hydra.sweep.dir=./outputs/par_seed_sweeper_$1_$env/$seed_num > log/$1_${env}_$seed_num.txt &"
  done
done
wait 
