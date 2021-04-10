#!/bin/bash

algos=("BC" "AIRL" "DRIL" "FAIRL" "GAIL" "GMMIL" "PUGAIL" "RED")

for alg in ${algos[@]}; do
  echo "running ./python3 main.py algorithm=$alg"
  python3 main.py algorithm=$alg
  done
