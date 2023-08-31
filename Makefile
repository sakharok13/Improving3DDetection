SHELL=/bin/bash

BASE=$(shell pwd)
ACTIVATE := $(BASE)/venv/bin/activate 
python := PYTHONPATH=$(BASE) \
 		  $(BASE)/venv/bin/python3

export PYTHONPATH=/home/jovyan/3d_detection/venv/bin/python3

#-- GPU parametrisation --
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=True

check_torch_gpu:
	cd src && \
	$(python) -c "from common.utils import check_gpu; check_gpu()"

#-- S3 utils --
aws --endpoint-url=https://n-ws-ao3th-pd11.s3pd11.sbercloud.ru s3 cp --recursive --acl authenticated-read s3://b-ws-ao3th-pd11-5qk/once_dataset/data/ ~/3d_detection/data/once_last

define upload_fld
	aws ${ENDPOINT} s3 cp --recursive --acl authenticated-read ${1}/ s3://b-ws-ao3th-pd11-5qk/${1}
endef

define download_fld
	aws ${ENDPOINT} s3 cp --recursive s3://b-ws-ao3th-pd11-5qk/${1}/ ${2}/
endef

#-- data preparation --

link_once_pcdet:
	ln -s ~/data/once_dataset/data ${BASE}/data/once/data

link_waymo_pcdet:
	ln -s ~/data/waymo_open_dataset_v_1_3_2/lidar_raw_data ${BASE}/data/waymo/raw_data

link_waymo_pcdet_processed:
	ln -s ~/data/waymo_open_dataset_v_1_3_2/waymo_processed_data_v0_5_0 ${BASE}/data/waymo/waymo_processed_data_v0_5_0

once_assets:
	$(python) -m pcdet.datasets.once.once_dataset --func create_once_infos \
		--cfg_file ${BASE}/tools/cfgs/dataset_configs/once_dataset.yaml

waymo_assets:
	$(python) -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    	--cfg_file ${BASE}/tools/cfgs/dataset_configs/waymo_dataset.yaml

waymo_semi_assets:
	$(python) -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    	--cfg_file ${BASE}/tools/cfgs/dataset_configs/waymo_semi_dataset.yaml

POINT_CLOUD_DATA=${BASE}/data/once/data/000034/lidar_roof/1616176019799.bin

demo:
	cd ./OpenPCDet/tools && \
	$(python) demo.py --cfg_file ./cfgs/once_models/pointpillar.yaml \
    --ckpt ${BASE}/epoch_160.pth \
    --data_path ${POINT_CLOUD_DATA}

jpt:
	source ${ACTIVATE} && \
	jupyter lab --ip=localhost --port=9888 ./notebooks/

####################
# Proficiet teachers
####################

# TODO find documentation to pretrain a model

CFG_FLD=${BASE}/tools/cfgs

CONFIG_FILE_WAYMO=${CFG_FLD}/waymo_models/sup_models/centerpoint.yaml

# CONFIG_FILE_ONCE=${CFG_FLD}/once_models/sup_models/centerpoint.yaml
CONFIG_FILE_ONCE=${CFG_FLD}/once_models/sup_models/second.yaml

train_waymo_sup:
	cd ./tools && \
	$(python) train.py --cfg_file ${CONFIG_FILE_WAYMO}

train_once_sup:
	cd ./tools && \
	$(python) train.py --cfg_file ${CONFIG_FILE_ONCE}
    

train_once_semi:
	cd ./tools && \
	$(python) semi_train.py --cfg_file ${CONFIG_FILE_ONCE}

# tensorboard --logdir 

BATCH_SIZE=16
CKPT=${BASE}/output/${BASE}/tools/cfgs/once_models/sup_models/second/default/ckpt
test_once_sup:
	cd ./tools && \
		python test.py --cfg_file ${CONFIG_FILE_ONCE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
	