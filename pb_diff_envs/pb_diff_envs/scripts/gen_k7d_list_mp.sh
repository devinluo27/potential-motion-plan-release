#!/bin/bash

source ~/.bashrc
source activate pb_diff  

## Generate training data for Kuka/Dual Kuka
export env='dualkuka14d-testOnly-v9'
export env='kuka7d-testOnly-v8'

## this should match the value when register the env
nsp=500; # the length of state trajectory of one env
np=5 # how many parts for multi-processing

{
python3 scripts/gen_k7d_list_mp.py \
--env_name $env \
--num_samples_permaze $nsp \
--num_mazes -1 \
--load_mazeEnv False \
--num_cores 32 \
--num_parts $np

exit 0
}

