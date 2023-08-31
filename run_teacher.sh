SHELL=/bin/bash

BASE=$(shell pwd)

ACTIVATE := $(BASE)/venv/bin/activate 
python := PYTHONPATH=$(BASE) \
        $(BASE)/venv/bin/python3
        
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=True

WAYMO_SECOND_CKPT=${BASE}/output/waymo_models/sup_models/second_d20/default/ckpt/checkpoint_epoch_30.pth

CFG_FLD=${BASE}/tools/cfgs



SUP_FILE_WAYMO=${CFG_FLD}/waymo_models/semi_learning_models/pv_rcnn/waymo_proficient_teachers.yaml

sup_waymo_second:
    cd ./tools && \
    $(python) train.py --cfg_file ${SUP_FILE_WAYMO}



BASE=~/3d_detection
PYTHONPATH=$BASE/venv/bin/python3
python=PYTHONPATH

WAYMO_SECOND_CKPT=${BASE}/output/waymo_models/sup_models/second_d20/default/ckpt/checkpoint_epoch_30.pth

CFG_FLD=${BASE}/tools/cfgs

SUP_FILE_WAYMO=${CFG_FLD}/waymo_models/semi_learning_models/pv_rcnn/waymo_proficient_teachers.yaml