#!/bin/bash
reset
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export ROOT_DIR="/home/muhan/data/workspace/kb_encoder"
export DATA_DIR="${ROOT_DIR}/data/preprocess"
export MODEL_DOWNLOAD_DIR="${ROOT_DIR}/data/google"
export MODEL_DIR="${ROOT_DIR}/train_t5_original/checkpoint"
export TASK="arc"
export PRETRAINED_DIR="${ROOT_DIR}/data/google"
export PRETRAINED_STEPS=1363200
export FINETUNE_STEPS=2000
export PYTHONPATH=`pwd`

t5_mesh_transformer  \
  --module_import="task" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin"\
  --gin_param="utils.run.mesh_shape = 'model:2,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1']" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 16}" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 512)" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --gin_param="utils.run.save_checkpoints_steps=500" \
  --gin_param="utils.run.keep_checkpoint_max=5"
