#!/bin/bash

source ~/.bashrc
source activate pb_diff

## evaluate sampling-based planners

export env='randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0'


nm=100; ## how many testing envs
nsp=20; ## how many problems per env
np=50 ## divided the problems to how many parts
nc=32; ## how many cpu cores
pl="bit"; ## which planner, [rrt, bit, sipp]
timeout=5; ## timeout limit for sampling-based planner

{
python3 scripts/trad/run_trad_planner.py \
--env_name $env \
--num_samples_permaze $nsp \
--num_mazes $nm \
--load_mazeEnv False \
--num_cores $nc \
--num_parts $np \
--planner $pl \
--timeout $timeout



exit 0
}

