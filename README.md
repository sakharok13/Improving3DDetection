wget https://www.python.org/ftp/python/3.9.17/Python-3.9.17.tar.xz
tar xfv Python-3.9.17.tar.xz
cd Python-3.9.17/
./configure --prefix=$PWD/Python-3.9.17/Python
make
make install 
## ADD PYTHON3.9 to ~/.bashrc (export PATH=$HOME...python3.9.x/bin....) and refresh with "source ~/.bashrc"

## Install
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements_cu111.txt
# pip install -r requirements_cu116.txt # alternative
# install frameworks OpenPCDet & dcn operator
python setup.py develop
cd pcdet/ops/dcn
python setup.py develop

# vis utils
pip install --trusted-host www.open3d.org -f http://www.open3d.org/docs/latest/getting_started.html open3d
```
#### Data versioning
```bash
pip install dvc[s3]
```


## Data

### Datasets
- Once. Sber S3 [Link](https://console.sbercloud.ru/projects/5ce29824-6053-4491-8a84-4e4cf97b986c/spa/mlspace/file-manager/b-ws-ao3th-pd11-5qk/once_dataset?customerId=95a82001-43a2-4e0e-8f4e-1be50845182a&sw=a012-industrial)
- Waymo. Sber S3 [Link](https://console.sbercloud.ru/projects/5ce29824-6053-4491-8a84-4e4cf97b986c/spa/mlspace/file-manager/b-ws-ao3th-pd11-5qk/waymo_open_dataset_v_2_0_0?customerId=95a82001-43a2-4e0e-8f4e-1be50845182a&sw=a012-industrial)


### Set up datasets
```bash
make link_once_pcdet
make link_waymo_pcdet
make link_waymo_pcdet_processed
dvc pull
```

## Benchmark

Please refer to this [page](related_workds/proficient_teachers/README.md) for detailed legacy benchmark results.


### Detection Models
We provide 1 fusion-based and 5 point cloud based 3D detectors. The training configurations are at `tools/cfgs/once_models/sup_models/*.yaml`

For PointPainting, you have to first produce segmentation results yourself. We used [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1) (pytorch version 1.1) trained on CityScapes to generate segmentation masks. 

### Semi-supervised Learning
We provide 5 semi-supervised methods based on the SECOND detector. The training configurations are at `tools/cfgs/once_models/semi_learning_models/*.yaml`
