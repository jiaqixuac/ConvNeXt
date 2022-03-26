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
    ${SRUN_ARGS} \
    python -u main.py \
    --model convnext_large --drop_path 0.3 --input_size 384 \
    --batch_size 32 --lr 5e-5 --update_freq 1 \
    --warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
    --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
    --data_set IMNET1k --nb_classes 1000 \
    --finetune pretrained/convnext_large_22k_224.pth \
    --data_path /mnt/lustre/share/images \
    --output_dir work_dirs/convnext_large_ft-1k_384_official
