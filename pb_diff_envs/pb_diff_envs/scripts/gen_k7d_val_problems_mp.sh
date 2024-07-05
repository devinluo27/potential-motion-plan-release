#!/bin/bash

source ~/.bashrc
source activate pb_diff

## Generate evaluation problems for Kuka/Dual Kuka

# export env="kuka7d-ng2ks25k-rand-nv4-se0505-vr0202-hExt20-v0"
# nsp=40; nm=100; np=40

export env="kuka7d-testOnly-v8"
# export env="dualkuka14d-testOnly-v9"

## this should match the value when register the env
nsp=6; # the number of problems of one evaluation env
nm=10; # how many different evaluation envs
np=5 # how many parts for multi-processing

{
python3 scripts/gen_k7d_val_problems_mp.py \
--env_name $env \
--num_samples_permaze $nsp \
--num_mazes $nm \
--load_mazeEnv False \
--num_cores 32 \
--num_parts $np \
--no_check_bit False \


exit 0
}

