SHELL=/bin/bash

BASE_OTHER=/home/jovyan/3d_detection
ACTIVATE := $(BASE_OTHER)/venv/bin/activate 
python := PYTHONPATH=$(BASE_OTHER) \
 		  $(BASE_OTHER)/venv/bin/python3

BASE=$(shell pwd)

OTHER_WAYMO_PCDET=/home/jovyan/3d_detection/OpenPCDet/data/waymo/
CUR_WAYMO_PDCET=${BASE}/data/waymo
# Steps reproduce experiment on Waymo dataset

export TF_FORCE_GPU_ALLOW_GROWTH=True
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=''

#1
link_waymo_pcdet:
	ln -s ~/data/waymo_open_dataset_v_1_3_2/lidar_raw_data ${CUR_WAYMO_PDCET}/raw_data

#2.1
link_waymo_npys:
	ln -s ${OTHER_WAYMO_PCDET}/waymo_processed_data_v0_5_0 ${CUR_WAYMO_PDCET}/waymo_processed_data

#2.2
link_waymo_infos:
	ln -s ${OTHER_WAYMO_PCDET}/waymo_processed_data_v0_5_0_infos_train.pkl ${CUR_WAYMO_PDCET}/waymo_processed_data_infos_train.pkl
	ln -s ${OTHER_WAYMO_PCDET}/waymo_processed_data_v0_5_0_infos_val.pkl ${CUR_WAYMO_PDCET}/waymo_processed_data_infos_val.pkl

#3
waymo_assets:
	$(python) -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    	--cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

#4
pretrain_slurm:
	GPUS_PER_NODE=1 bash tools/scripts/slurm_pretrain.sh p1 \
	'pretrain_sampled_10' 1 --cfg_file ./tools/cfgs/waymo_models/pv_rcnn.yaml \
	--split pcdet_waymo_dbinfos_train_sampled_10 --extra_tag ./logs \
	--ckpt_save_interval 1 \
	--repeat 2 \
	--dbinfos ./data/waymo/pcdet_waymo_dbinfos_train_sampled_10.pkl

pretrain:
	cd ./tools && \
	$(python) pretrain.py --cfg_file cfgs/waymo_models/pv_rcnn.yaml \
	--split train_part --extra_tag ./logs \
	--ckpt_save_interval 1 \
	--repeat 2 \
	--eval_batch_size 2 \
	--dbinfos pcdet_waymo_dbinfos_train_sampled_10.pkl
