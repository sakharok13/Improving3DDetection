# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation

Currently we provide the dataloader of KITTI, NuScenes, Waymo, Lyft and Pandaset. If you want to use a custom dataset, Please refer to our [custom dataset template](CUSTOM_DATASET_TUTORIAL.md).

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
OpenPCDet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, for single/multi-frame)
 
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
# only for single-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

# for single-frame or multi-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
# Ignore 'CUDA_ERROR_NO_DEVICE' error as this process does not require GPU.
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 


### ONCE Dataset
* Please download train/val/test of the official [ONCE Dataset](https://once-for-auto-driving.github.io/download.html#downloads) and 
organize the downloaded files as follows:
* Note that the whole dataset is large (2TB) and most scenes are unlabeled, so if you only need ONCE for supervised 3D object detection and model development, you can just download the training/validation/testing split. If you use ONCE for semi-supervised/self-supervised 3D object detection, you can choose to download the respective unlabeled splits (unlabeled small split: 100k unlabeled scenes; unlabeled medium split: 500k unlabeled scenes; unlabeled large split: 1M unlabeled scenes).
```
ONCE_Benchmark
├── data
│   ├── once
│   │   │── ImageSets
|   |   |   ├──train.txt
|   |   |   ├──val.txt
|   |   |   ├──test.txt
|   |   |   ├──raw_small.txt (100k unlabeled)
|   |   |   ├──raw_medium.txt (500k unlabeled)
|   |   |   ├──raw_large.txt (1M unlabeled)
│   │   │── data
│   │   │   ├──000000
|   |   |   |   |──000000.json (infos)
|   |   |   |   |──lidar_roof (point clouds)
|   |   |   |   |   |──frame_timestamp_1.bin
|   |   |   |   |  ...
|   |   |   |   |──cam0[1-9] (images)
|   |   |   |   |   |──frame_timestamp_1.jpg
|   |   |   |   |  ...
|   |   |   |  ...
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.once.once_dataset --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml
```

    

## Pretrained Models
If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the pretrained [DeepLabV3 model](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and place within the `checkpoints` directory. Please make sure the [kornia](https://github.com/kornia/kornia) is installed since it is needed for `CaDDN`.
```
OpenPCDet
├── checkpoints
│   ├── deeplabv3_resnet101_coco-586e9e4e.pth
├── data
├── pcdet
├── tools
```

## Training & Testing


### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
