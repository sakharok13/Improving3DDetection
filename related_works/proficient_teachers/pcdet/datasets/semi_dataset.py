from collections import defaultdict
from pathlib import Path
import copy
import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .augmentor.ssl_data_augmentor import SSLDataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class SemiDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = Path(root_path) if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features
        )

        if self.dataset_cfg.get('USE_SHARED_AUGMENTOR', False):
            self.share_augmentor = SSLDataAugmentor(
                self.root_path, self.dataset_cfg.SHARED_AUGMENTOR, self.class_names, logger=self.logger
            ) if self.training else None
        else:
            self.share_augmentor = None

        self.teacher_augmentor = SSLDataAugmentor(
            self.root_path, self.dataset_cfg.TEACHER_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None

        self.student_augmentor = SSLDataAugmentor(
            self.root_path, self.dataset_cfg.STUDENT_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.contrastive = self.dataset_cfg.CONTRASTIVE if 'CONTRASTIVE' in self.dataset_cfg else False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    def prepare_data_ssl(self, data_dict, output_dicts):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if 'gt_boxes' in data_dict:
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            data_dict={
                **data_dict,
                'gt_boxes_mask': gt_boxes_mask
            }

        if self.share_augmentor is not None:
            data_dict = self.share_augmentor.forward(data_dict)

        if 'teacher' in output_dicts:
            teacher_data_dict = self.teacher_augmentor.forward(copy.deepcopy(data_dict))
        else:
            teacher_data_dict = None

        if 'student' in output_dicts:
            student_data_dict = self.student_augmentor.forward(copy.deepcopy(data_dict))
            student_data_dict['augmentation_list'] = [student_data_dict['augmentation_list']]
            student_data_dict['augmentation_params'] = [student_data_dict['augmentation_params']]
        else:
            student_data_dict = None

        for data_dict in [teacher_data_dict, student_data_dict]:
            if data_dict is None:
                continue

            if 'gt_boxes' in data_dict:
                if len(data_dict['gt_boxes']) == 0:
                    new_index = np.random.randint(self.__len__())
                    return self.__getitem__(new_index)

                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
                data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
                data_dict['gt_names'] = data_dict['gt_names'][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes'] = gt_boxes

            # if not isinstance(data_dict['points'], list):
            #     data_dict = self.point_feature_encoder.forward(data_dict)
            # else:
            data_dict['use_lead_xyz'] = True

            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )
            data_dict.pop('gt_names', None)

        return teacher_data_dict, student_data_dict


    def prepare_data_ssl_unlabel(self, data_dict, output_dicts):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if 'gt_boxes' in data_dict:
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            data_dict={
                **data_dict,
                'gt_boxes_mask': gt_boxes_mask
            }

        if self.share_augmentor is not None:
            data_dict = self.share_augmentor.forward(data_dict)

        if 'teacher' in output_dicts:
            teacher_data_dict = self.teacher_augmentor.forward(copy.deepcopy(data_dict))
        else:
            teacher_data_dict = None

        if 'student' in output_dicts:
            if self.contrastive:
                student_data_dict = {'frame_id':data_dict['frame_id']}
                student_data_dict_view1, student_data_dict_view2 = data_dict.copy(), data_dict.copy()
                student_data_dict_view1 = self.student_augmentor.forward(copy.deepcopy(student_data_dict_view1))
                student_data_dict_view2 = self.student_augmentor.forward(copy.deepcopy(student_data_dict_view2))
                student_data_dict['points'] = [student_data_dict_view1['points'], student_data_dict_view2['points']]
                student_data_dict['gt_names'] = [student_data_dict_view1['gt_names'], student_data_dict_view2['gt_names']]
                student_data_dict['gt_boxes'] = [student_data_dict_view1['gt_boxes'], student_data_dict_view2['gt_boxes']]
                student_data_dict['augmentation_list'] = [student_data_dict_view1['augmentation_list'], student_data_dict_view2['augmentation_list']]
                student_data_dict['augmentation_params'] = [student_data_dict_view1['augmentation_params'], student_data_dict_view2['augmentation_params']]
            else:
                student_data_dict = self.student_augmentor.forward(copy.deepcopy(data_dict))
                student_data_dict['augmentation_list'] = [student_data_dict['augmentation_list']]
                student_data_dict['augmentation_params'] = [student_data_dict['augmentation_params']]
        else:
            student_data_dict = None

        for data_dict in [teacher_data_dict, student_data_dict]:
            if data_dict is None:
                continue

            if 'gt_boxes' in data_dict:
                if isinstance(data_dict['gt_boxes'], list):
                    if len(data_dict['gt_boxes'][0]) == 0:
                        new_index = np.random.randint(self.__len__())
                        return self.__getitem__(new_index)

                    for i in range(2):
                        gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names'][i]],
                                              dtype=np.int32)
                        gt_boxes = np.concatenate((data_dict['gt_boxes'][i], gt_classes.reshape(-1, 1).astype(np.float32)),
                                                  axis=1)
                        data_dict['gt_boxes'][i] = gt_boxes
                else:
                    if len(data_dict['gt_boxes']) == 0:
                        new_index = np.random.randint(self.__len__())
                        return self.__getitem__(new_index)

                    selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
                    data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
                    data_dict['gt_names'] = data_dict['gt_names'][selected]
                    gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                    gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                    if 'gt_scores' in data_dict:
                        data_dict['gt_scores'] = data_dict['gt_scores'][selected]
                        gt_scores = data_dict['gt_scores'].reshape(-1, 1).astype(np.float32)
                        gt_boxes = np.concatenate((gt_boxes, gt_scores), axis=1)
                        data_dict.pop('gt_scores', None)
                    data_dict['gt_boxes'] = gt_boxes

            # if not isinstance(data_dict['points'], list):
            #     data_dict = self.point_feature_encoder.forward(data_dict)
            # else:
            data_dict['use_lead_xyz'] = True

            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )
            data_dict.pop('gt_names', None)

        return teacher_data_dict, student_data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):

        def collate_single_batch(batch_list):
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                if isinstance(cur_sample, dict):
                    for key, val in cur_sample.items():
                        data_dict[key].append(val)
                else:
                    raise Exception('batch samples must be dict')

            batch_size = len(batch_list)
            ret = {}
            for key, val in data_dict.items():
                try:
                    if key in ['voxels', 'voxel_num_points']:
                        if isinstance(val[0], list):
                            for i, ele in enumerate(val):
                                val[i] = np.concatenate(ele, axis=0)
                        ret[key] = np.concatenate(val, axis=0)
                    elif key in ['points', 'voxel_coords']:
                        if isinstance(val[0], list):
                            coors = []
                            for i in range(batch_size):  # add the batch index in the front of tensor
                                for j, coor in enumerate(val[i]):
                                    coor_pad = np.pad(
                                        coor, ((0, 0), (1, 0)), mode="constant", constant_values=i * len(val[0]) + j
                                        # i:batch, j:frame
                                    )

                                    coors.append(coor_pad)
                            ret[key] = np.concatenate(coors, axis=0)
                        else:
                            coors = []
                            for i, coor in enumerate(val):
                                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                                coors.append(coor_pad)
                            ret[key] = np.concatenate(coors, axis=0)
                    elif key in ['gt_boxes']:
                        if isinstance(val[0], list):
                            val = [x for y in val for x in y]
                            max_gt = max([len(x) for x in val])
                            batch_gt_boxes3d = np.zeros((len(val), max_gt, val[0].shape[-1]), dtype=np.float32)
                            for k in range(len(val)):
                                batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                            ret[key] = batch_gt_boxes3d
                        else:
                            max_gt = max([len(x) for x in val])
                            batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                            for k in range(batch_size):
                                batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                            ret[key] = batch_gt_boxes3d
                    elif key in ['augmentation_list', 'augmentation_params']:
                        if isinstance(val[0], list):
                            val = [x for y in val for x in y]
                        ret[key] = val
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

            ret['batch_size'] = batch_size
            return ret

        if isinstance(batch_list[0], dict):
            return collate_single_batch(batch_list)
        elif isinstance(batch_list[0], tuple):
            if batch_list[0][0] is None:
                teacher_batch = None
            else:
                teacher_batch_list = [sample[0] for sample in batch_list]
                teacher_batch = collate_single_batch(teacher_batch_list)
            if batch_list[0][1] is None:
                student_batch = None
            else:
                student_batch_list = [sample[1] for sample in batch_list]
                student_batch = collate_single_batch(student_batch_list)
            return teacher_batch, student_batch
        else:
            raise Exception('batch samples must be dict or tuple')
