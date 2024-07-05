#!/bin/bash

## TODO: Please check your cuda version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

TORCH_V=torch-2.0.1
CUDA_V=cu118


## The installation of the additional torch packages below might take ~10 minutes
## From https://github.com/pyg-team/pytorch_geometric/issues/1876#issuecomment-736317888
## NOTE: These are only used in the SIPP planner in pb_diff_envs.
## If you don't use it or cannot install them, you can simply skip these packages 
## and comment out the corresponding 'import' in the python code.

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_V}+${CUDA_V}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_V}+${CUDA_V}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_V}+${CUDA_V}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_V}+${CUDA_V}.html
pip install torch_geometric==2.3.1



## ----------------------------------------------------
## ------ We moved these steps to the README.md  ------
## install general packages
# pip install -r requirements.txt


## install general packages
# pip install -e pb_diff_envs