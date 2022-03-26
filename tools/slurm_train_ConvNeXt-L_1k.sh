#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    --exclude=SH-IDC1-10-140-0-232 \
    ${SRUN_ARGS} \
    python -u main.py \
    --model convnext_large --drop_path 0.5 \
    --batch_size 128 --lr 4e-3 --update_freq 2 \
    --model_ema true --model_ema_eval true \
    --data_set IMNET1k --nb_classes 1000 \
    --data_path /mnt/lustre/share/images \
    --output_dir work_dirs/convnext_large_1k
