#!/bin/bash
export ROOT_DIR="/home/muhan/data/workspace/kb_encoder"
export MODEL_DOWNLOAD_DIR="${ROOT_DIR}/data/google"
mkdir -p "${MODEL_DOWNLOAD_DIR}"
wget -O "${MODEL_DOWNLOAD_DIR}/operative_config.gin" https://storage.googleapis.com/unifiedqa/models_v2/11B/operative_config.gin
wget -O "${MODEL_DOWNLOAD_DIR}/model.ckpt-1363200.data-00000-of-00002" https://storage.googleapis.com/unifiedqa/models_v2/11B/model.ckpt-1363200.data-00000-of-00002
wget -O "${MODEL_DOWNLOAD_DIR}/model.ckpt-1363200.data-00001-of-00002" https://storage.googleapis.com/unifiedqa/models_v2/11B/model.ckpt-1363200.data-00001-of-00002
wget -O "${MODEL_DOWNLOAD_DIR}/model.ckpt-1363200.meta" https://storage.googleapis.com/unifiedqa/models_v2/11B/model.ckpt-1363200.meta
wget -O "${MODEL_DOWNLOAD_DIR}/model.ckpt-1363200.index" https://storage.googleapis.com/unifiedqa/models_v2/11B/model.ckpt-1363200.index
