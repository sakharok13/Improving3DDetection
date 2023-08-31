#!/bin/bash

TASK_DESC=$1
PORT=$((8000 + RANDOM %57535))

export PYTHONPATH=$PYTHONPATH:/home/junbo/ssd/repository/SemiDet3D

# ONCE 

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8  test.py --launcher pytorch --cfg_file cfgs/kitti_models/pointrcnn.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT} \
# --ckpt_dir="/home/ssd2/OpenPCDet-SSL/output/kitti_models/pointrcnn/baseline/ckpt" --eval_all --start_epoch=70 --save_to_file

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  test.py --launcher pytorch --cfg_file cfgs/once_models/sup_models/second.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT} \s
 # --ckpt="/ssd/junbo/repository/ONCE_Benchmark/output/once_models/semi_learning_models/second/ensembled_teacher_small/baseline_4GPU_stage2-cyc05_ep50/ssl_ckpt/student/checkpoint_epoch_46.pth"
 
# Waymo
#CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file cfgs/waymo_models/sup_models/second.yaml --extra_tag ${TASK_DESC}  --result_file /home/junbo/ssd/repository/SemiDet3D/output/waymo_models/sup_models/second/baseline_D5_tta_val/eval/epoch_30/val/default/result.pkl --save_to_file

 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8  test.py --launcher pytorch --cfg_file cfgs/waymo_models/sup_models/second.yaml --extra_tag ${TASK_DESC} --tcp_port ${PORT} --workers=2 \
 --ckpt "/home/junbo/repository/ProficientTeachers/output/waymo_models/sup_models/second/baseline_D20_run1/ckpt/checkpoint_epoch_30.pth"   \
# --save_to_file