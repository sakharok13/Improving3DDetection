#!/bin/bash

TASK_DESC=$1
PORT=$((8000 + RANDOM %57535))

export PYTHONPATH=$PYTHONPATH:/home/junbo/ssd/repository/SemiDet3D

# ONCE

# sup
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/once_models/sup_models/second.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT} \
#--pretrained_model="/ssd/junbo/repository/SemiDet3D/pretrained/waymo/CENTER_point2048_fps_batch4_ep50_D10_20211010-230029/latest.pth"

# semi
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 semi_train.py --launcher pytorch --cfg_file cfgs/once_models/semi_learning_models/second/ioumatch3d_second_small.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT}


# sync code
# rsync -av -e 'ssh -p 9527' --exclude=data --exclude=pcdet/ops --exclude=output --exclude=build --exclude=pcdet.egg-info /mnt/data/junbo/repository/SemiDet3D junbo@10.128.104.49:/home/junbo/ssd/repository/

# Waymo

# sup
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/waymo_models/sup_models/second.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT}

# semi
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 semi_train.py --launcher pytorch --cfg_file cfgs/waymo_models/semi_learning_models/second/ensembled_teacher_contrast.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT}
