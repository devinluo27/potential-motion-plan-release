#!/bin/bash

source ~/.bashrc
source activate pb_diff

export config="config/comp/Comp_rSmaze_nw6nw6_hExt0505gsd_engy_add.py"
export config="config/comp/Comp_rSmaze_nw6nw5_hExt0505gsd_engy_add.py"


export g=(0 1 2 3 4 5 6 7)

echo $CUDA_HOME
{ 
CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=1 \
python3 scripts/comp/comp_plan_rm2d.py \
--config $config \
--plan_n_maze $1 \


echo "Done"
exit 0
}
