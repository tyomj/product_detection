#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
#CHECKPOINT=$2
GPUS=$2
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_ensemble.py $CONFIG \
    /home/artem-nb/Projects/SKU110k/work_dirs/cascade_rcnn_x101_32x4d_fpn_anchor_1x_fold2_2s/latest.pth \
    /home/artem-nb/Projects/SKU110k/work_dirs/cascade_rcnn_x101_32x4d_fpn_anchor_1x_fold3_2s/latest.pth \
    --launcher pytorch ${@:4} \
    --format_only --options "jsonfile_prefix=./2_folds_23"
