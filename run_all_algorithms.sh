#!/bin/bash

algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "PUGAIL" "RED")

for alg in ${algos[@]}; do
  if [[ $# -eq 0 ]] ; then
    echo "running python3 main.py -m algorithm=$alg"
    python3 main.py algorithm=$alg
  else
    echo "running python3 main.py -m algorithm=$alg environment=$1"
    python3 main.py -m algorithm=$alg environment=$1
  fi
  done
