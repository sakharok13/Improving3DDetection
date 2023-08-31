SHELL=/bin/bash

BASE=$(shell pwd)

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
ENDPOINT = --endpoint-url=https://n-ws-ao3th-pd11.s3pd11.sbercloud.ru

ls:
	aws ${ENDPOINT} s3 ls --recursive s3://b-ws-ao3th-pd11-5qk

define download
	gsutil -m cp -r "gs://waymo_open_dataset_v_1_3_2/archived_files/${1}" ${2}/
endef

define upload
	aws ${ENDPOINT} s3 cp --recursive --acl authenticated-read ${1}/ s3://b-ws-ao3th-pd11-5qk/waymo_open_dataset_v_1_3_2/${2}
endef

define dump_filelist
	find ${1}/data/*.tfrecord  -printf "%f\n" > ${1}.txt
	# ls -1 ${1}/data/*.tfrecord > ${1}.txt
endef

define download_train
	$(call download,training/training_00${1}.tar,.)
endef

define untar
	cd ${1} && mkdir -p data && \
	for file in `find . -name '*.tar'`; do \
        tar -xvf "$${file}" -C ./data ; done
endef

# TRAIN

part21:
	$(call download_train,21)

part22:
	$(call download_train,22)

part23:
	$(call download_train,23)

part24:
	$(call download_train,24)

part25:
	$(call download_train,25)

part26:
	$(call download_train,26)

part27:
	$(call download_train,27)

part28:
	$(call download_train,28)

part29:
	$(call download_train,29)

part30:
	$(call download_train,30)

part31:
	$(call download_train,31)
###

untar_train3:
	$(call untar,training-part14-20)

untar_train4:
	$(call untar,training-part21-31)

#####

upload_train3:
	$(call upload,training-part14-20/data,lidar_raw_data)

upload_train4:
	$(call upload,training-part21-31/data,lidar_raw_data)

# VALIDATION

download_validation:
	$(call download,validation)

download_testing:
	$(call download,testing)

upload_validation:
	$(call upload,validation/data,lidar_raw_data)

################

list_val:
	$(call dump_filelist,validation)

upload_dumplist:
	$(call upload,dumplist,)

################

showtars:
	for file in `find ./validation -name '*.tar'`; do \
        echo "file is $${file}" ; done