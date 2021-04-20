#!/bin/bash

algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "PUGAIL" "RED")
log_file="full_algorithm.log"
for alg in ${algos[@]}; do
  if [[ $# -eq 0 ]] ; then
    echo "running python3 main.py -m algorithm=$alg" | tee -a "$log_file"
    python3 main.py algorithm=$alg | tee -a "$log_file"
  else
    echo "running python3 main.py -m algorithm=$alg environment=$1"
    python3 main.py -m algorithm=$alg environment=$1
  fi
  done
