#!/bin/bash

envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
[[ " ${envs[@]} " =~ " $1 " ]] && echo "environment=$1" || { echo "invalid input, second input must be empty or one of ${envs[*]} "; exit 1; }
./scripts/run_seed_experiments.sh SAC $1 $2 "steps=3000000"
