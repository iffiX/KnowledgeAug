#!/bin/bash

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda init
conda create --y --name t5 python=3.7
conda activate t5
#conda install -c anaconda cudatoolkit==11.3.1 cudnn==8.2.1

pip install t5[gcp]==0.9.3

export ROOT_DIR="/home/muhan/data/workspace/kb_encoder"
export MODEL_DIR="${ROOT_DIR}/train_t5_original/checkpoint"

mkdir -p "${MODEL_DIR}"