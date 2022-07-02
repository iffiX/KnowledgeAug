#!/bin/bash
reset
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

export PROJECT="kb-encoder2"
export ZONE="us-central1-a"
export TPU="hanhanmumuqq"
export TPU_SIZE="v3-8"
export TPU_MODEL_PARALLELISM="8"
export TASK="arc"
export DATA_DIR="gs://kb-encoder/preprocess-2"
export MODEL_DIR="gs://kb-encoder/${TASK}/"
export PRETRAINED_DIR="gs://unifiedqa/models_v2/11B"
export PRETRAINED_STEPS=1363200
export FINETUNE_STEPS=17000
export PYTHONPATH=`pwd`

eval "$(conda shell.bash hook)"

nohup t5_mesh_transformer  \
  --module_import="task" \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin"\
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 32}" \
  --gin_param="tpu_mesh_shape.model_parallelism = ${TPU_MODEL_PARALLELISM}" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 4096)" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --gin_param="utils.run.save_checkpoints_steps=1000" \
  --gin_param="get_variable_dtype.activation_dtype='bfloat16'" &

tail -f nohup.out
