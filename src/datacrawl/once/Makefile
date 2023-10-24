SHELL=/bin/bash

BASE=$(shell pwd)
ACTIVATE := $(BASE)/venv/bin/activate 
python := PYTHONPATH=$(BASE):${PYTHONPATH} \
 		  $(BASE)/venv/bin/python3


define upload
	aws ${ENDPOINT} s3 sync --size-only --acl authenticated-read ${1}/ s3://b-ws-ao3th-pd11-5qk/once_dataset/${1}
endef

define download
	source ${ACTIVATE} && \
	gdown ${1}
endef

define download_folder
	source ${ACTIVATE} && \
	gdown --folder ${1}
endef

train_annot:
	$(call download,1Qa6JYbp7SEZtq7wI0Sw7bvhLc)

train_lidar:
	$(call download,1fy_R1TOKOyvMtCgwR5YwpZtiih0c_oD5)

train_camera:
	$(call download_folder,1E85-kPxCatAUGx-EvnNJJpnJ7pfxqmPH)
###############
val_annot:
	$(call download,1slhW1vAR7Ps1TL-QLL4SHpMV6aN6vUse)

val_lidar:
	$(call download,1BQvbh7pdCayjoXKWPD5-_veYtBgvL_8N)

val_camera:
	$(call download_folder,1zYgxnU5NBoAWz9TvMgfapKkZ2YhTtyl7)
###############
test_annot:
	$(call download,19R2MRZxpe-4VWJLuCQbcVTZkS4-LV1N3)

test_lidar:
	$(call download,1-6UwiBMlGjgMtQTo3NIVFC0Il_KU_gWv)

test_camera:
	$(call download_folder,1hd8L36qNuh_7hI0yb_xjt9yYqBVXlMyx)
###############
# -tf for listing
showtars:
	for file in `find . -name '*.tar'`; do \
        echo "file is $${file}" ; done

untarall:
	for file in `find . -name '*.tar'`; do \
        tar -xvf "$${file}" -C ./data_root ; done
###############
upload_imgs:
	cd /Users/nurgaliev/Downloads/once/imgs && \
	$(call upload,data)

upload_annot:
	cd /Users/nurgaliev/Downloads/once/annot && \
	$(call upload,data)

upload_lidar:
	cd /Users/nurgaliev/Downloads/once/lidar && \
	$(call upload,data)
