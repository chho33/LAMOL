#!/bin/bash -i

source ./env

python train.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"
