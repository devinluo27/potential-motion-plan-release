#!/bin/bash
source ~/.bashrc
source activate pb_diff
alias python=python3

env_list=( 
    # 'ComprandSmaze2d-luotest-v3' \
    # 'ComprandSmaze2d-luotest-nw61-0505-v0' \
    'ComprandSmaze2d-ms55nw6nw1-hExt0507-v0' \
    'ComprandSmaze2d-ms55nw6nw2-hExt0507-v0' \
    'ComprandSmaze2d-ms55nw6nw3-hExt0507-v0' \
    'ComprandSmaze2d-ms55nw6nw1-hExt0505-gsdiff-v0' \
    'ComprandSmaze2d-ms55nw6nw2-hExt0505-gsdiff-v0' \
    'ComprandSmaze2d-ms55nw6nw3-hExt0505-gsdiff-v0' \
    'ComprandSmaze2d-ms55nw6nw4-hExt0505-gsdiff-v0' \
    'ComprandSmaze2d-ms55nw6nw5-hExt0505-gsdiff-v0' \
    'ComprandSmaze2d-ms55nw6nw6-hExt0505-gsdiff-v0' \
)

## this should match the value when register the env
nsp=25000;np=50 # nm=6
# nsp=20000;np=200 # nm=6
# nsp=1000;np=3
# nsp=500;np=6

{
## loop 
for env in "${env_list[@]}"; do
    python3 scripts/gen_m2d_mp.py \
    --env_name $env \
    --num_samples_permaze $nsp \
    --num_mazes -1 \
    --load_mazeEnv False \
    --num_cores 32 \
    --num_parts $np \

done
exit 0
}

