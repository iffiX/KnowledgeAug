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
if [ $# -eq 0 ]
  then
    echo "Eval step not set, using default 1363200"
    export EVAL_STEPS=1363200
else
  export EVAL_STEPS=$1
fi

export PYTHONPATH=`pwd`

eval "$(conda shell.bash hook)"

t5_mesh_transformer  \
  --module_import="task" \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin"\
  --gin_file="eval.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="tpu_mesh_shape.model_parallelism = ${TPU_MODEL_PARALLELISM}" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 32}" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 16384)" \
  --gin_param="utils.run.eval_checkpoint_step=${EVAL_STEPS}" \
  --gin_param="utils.run.dataset_split = 'test'"


