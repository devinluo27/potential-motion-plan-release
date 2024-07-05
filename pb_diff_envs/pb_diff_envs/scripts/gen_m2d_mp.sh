#!/bin/bash

source ~/.bashrc
source activate pb_diff

## Generating Training dataset for static and dynmaic Maze2D Env

export env='randSmaze2d-testOnly-v0'
# export env='randSmaze2d-C43-testOnly-v0'

## this should match the value when register the env
nsp=25000;np=50
nsp=1000;np=3

{
python3 scripts/gen_m2d_mp.py \
--env_name $env \
--num_samples_permaze $nsp \
--num_mazes -1 \
--load_mazeEnv False \
--num_cores 32 \
--num_parts $np \


exit 0
}

