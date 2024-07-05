#!/bin/bash

source ~/.bashrc
source activate pb_diff 


### ------ TestOnly version (loading toy dataset) --------
# export config="config/rm2d/rSmaze_engy_testOnly.py"
# export config="config/rm2d/rSmazeC43_engy_testOnly.py"
# export config="config/rm2d/DynrSmaze_engy_testOnly.py"
# export config="config/kuka7d/kuka_engy_testOnly.py"
# export config="config/dualk14/dualk14_engy_testOnly.py"
### -------------------------------------------------------

## Kuka
export config="config/kuka7d/kuka_exp.py"
export config="config/dualk14/dualk14_exp.py"

## Maze2D
export config="config/rm2d/rSmaze_nw6_hExt05_exp.py"
export config="config/rm2d/rSmaze_nw3_hExt07_exp.py"

export config="config/rm2d/rSmazeC43_concave_nw7_exp.py"
export config="config/rm2d/DynrSmaze_nw1_hExt10_exp.py"

export g=(0 1 2 3 4 5 6 7)

{
CUDA_VISIBLE_DEVICES=${g[0]} OMP_NUM_THREADS=1 \
python3 scripts/train.py --config $config

echo "Done"
exit 0
}