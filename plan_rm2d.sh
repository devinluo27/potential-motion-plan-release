#!/bin/bash

source ~/.bashrc
source activate pb_diff


## Launch Evaluation for Base Maze2D Env

## Static
export config="config/rm2d/rSmaze_nw6_hExt05_exp.py"
export config="config/rm2d/rSmaze_nw3_hExt07_exp.py"
# export config="config/rm2d/rSmazeC43_concave_nw7_exp.py"

## Dynamic Env
# export config="config/rm2d/DynrSmaze_nw1_hExt10_exp.py"

export g=(0 1 2 3 4 5 6 7)

{
CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=1 \
python3 -B scripts/plan_rm2d.py \
--config $config \
--plan_n_maze $1 \

echo "Done"
exit 0
}
