#!/bin/bash

source ~/.bashrc
source activate pb_diff

## Generate evaluation problems for dynamic maze2d env
export env='DynrandSmaze2d-testOnly-v2'

## this should match the value when register the env
nsp=200;nm=6;np=6


{
python3 \
scripts/gen_dyn_m2d_val_probs_mp.py \
--env_name $env \
--num_samples_permaze $nsp \
--num_mazes $nm \
--load_mazeEnv False \
--num_cores 32 \
--num_parts $np \
--is_eval True

exit 0
}

