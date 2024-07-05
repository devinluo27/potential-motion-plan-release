#!/bin/bash

source ~/.bashrc
source activate pb_diff


## Launch Evaluation for Base Kuka/Dual Kuka Env
export config="config/kuka7d/kuka_exp.py"
# export config="config/dualk14/dualk14_exp.py"

export g=(0 1 2 3 4 5 6 7)

echo $CUDA_HOME

{
CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=1 \
python3 scripts/plan_kuka.py \
--config $config \
--plan_n_maze $1 \

echo "Done"
exit 0
}
