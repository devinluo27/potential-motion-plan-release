#!/bin/bash

source ~/.bashrc
source activate pb_diff

## Generate evaluation problems for static maze2D

export env="randSmaze2d-testOnly-v0"
export env='randSmaze2d-C43-testOnly-v0'


## this should match the value when register the env
# nsp=200; nm=300; np=100
nsp=20; # the number of problems of one evaluation env
nm=6; # how many different evaluation envs
np=3 # how many parts for multi-processing

{
python3 scripts/gen_static_m2d_val_probs_mp.py \
--env_name $env \
--num_samples_permaze $nsp \
--num_mazes $nm \
--load_mazeEnv False \
--num_cores 32 \
--num_parts $np

exit 0
}

