#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    --exclude=SH-IDC1-10-140-0-232 \
    ${SRUN_ARGS} \
    python -u main.py \
    --model convnext_large --drop_path 0.1 \
    --batch_size 128 --lr 4e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 90 \
    --data_set CEPH22k --nb_classes 21841 --disable_eval true \
    --data_path dataset:s3://imagenet22k \
    --output_dir work_dirs
