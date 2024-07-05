#!/bin/bash

source ~/.bashrc
source activate pb_diff


config_list=( \
 "config/comp/Comp_rSmaze_nw6nw1_hExt0505gsd_engy_add.py" \ 
 "config/rm2d/Comp_rSmaze_nw6nw3_hExt0507_engy_add.py" \
)

{
# Use a for loop to iterate over multiple configs
for config in "${config_list[@]}"; do
    echo "Current: $config"
    CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=1 \
    python3 scripts/comp/comp_plan_rm2d.py \
    --config $config \
    --plan_n_maze $1 \
    --cond_w 2.0 2.0

done
exit 0
}

