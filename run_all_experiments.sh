#!/bin/bash

envs=('ant' 'halfcheetah' 'hopper' 'walker2d')
for e in ${envs[@]}; do
  ./run_all_algorithms.sh $e
done
