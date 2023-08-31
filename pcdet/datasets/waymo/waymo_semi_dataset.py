import copy
import pickle
import numpy as np
from pathlib import Path
import multiprocessing
import SharedArray
from ..semi_dataset import SemiDatasetTemplate
from ...utils import box_utils, common_utils

def split_waymo_semi_data(info_paths, data_splits, root_path, labeled_ratio, logger):
    waymo_pretrain_infos = []
    waymo_test_infos = []
    waymo_labeled_infos = []
    waymo_unlabeled_infos = []

    def class_balance(waymo_infos):
        class_names = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']
        _cls_infos = {name: [] for name in class_names}
        for info in waymo_infos:
            gt_names = info['annos']["name"]
            for name in set(gt_names):
                if name in class_names:
                    _cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
        _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

        cbgs_once_infos = []

        frac = 1.0 / len(class_names)
        ratios = [frac / v for v in _cls_dist.values()]

        for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):  # todo: replace=False
            cbgs_once_infos += [cls_infos[i] for i in
                                 np.random.randint(low=0, high=len(cls_infos), size=int(len(cls_infos) * ratio))]

        return cbgs_once_infos

    root_path = Path(root_path)

    train_split = data_splits['train']
    for info_path in info_paths[train_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            # infos = class_balance(infos)
            waymo_pretrain_infos.extend(copy.deepcopy(infos))
            waymo_labeled_infos.extend(copy.deepcopy(infos))

    test_split = data_splits['test']
    for info_path in info_paths[test_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            waymo_test_infos.extend(copy.deepcopy(infos))

    raw_split = data_splits['raw']
    for info_path in info_paths[raw_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            waymo_unlabeled_infos.extend(copy.deepcopy(infos))

    logger.info('Total samples for Waymo pre-training dataset: %d' % (len(waymo_pretrain_infos)))
    logger.info('Total samples for Waymo testing dataset: %d' % (len(waymo_test_infos)))
    logger.info('Total samples for Waymo labeled dataset: %d' % (len(waymo_labeled_infos)))
    logger.info('Total samples for Waymo unlabeled dataset: %d' % (len(waymo_unlabeled_infos)))

    return waymo_pretrain_infos, waymo_test_infos, waymo_labeled_infos, waymo_unlabeled_infos

class WaymoSemiDataset(SemiDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG

        if training:
            interval = dataset_cfg.SAMPLED_INTERVAL['train']
        else:
            interval = dataset_cfg.SAMPLED_INTERVAL['val']

        self.infos = infos[::interval]

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def get_image(self, sequence_id, frame_id, cam_name):
        return self.toolkits.load_image(sequence_id, frame_id, cam_name)

    def project_lidar_to_image(self, sequence_id, frame_id):
        return self.toolkits.project_lidar_to_image(sequence_id, frame_id)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN',
                                                      False) and 'num_points_in_gt' in annos:
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

class WaymoPretrainDataset(WaymoSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    # def __getitem__(self, index):
    #     if self._merge_all_iters_to_one_epoch:
    #         index = index % len(self.once_infos)
    #
    #     info = copy.deepcopy(self.once_infos[index])
    #     frame_id = info['frame_id']
    #     seq_id = info['sequence_id']
    #     points = self.get_lidar(seq_id, frame_id)
    #     input_dict = {
    #         'points': points,
    #         'frame_id': frame_id,
    #     }
    #
    #     if 'annos' in info:
    #         annos = info['annos']
    #         input_dict.update({
    #             'gt_names': annos['name'],
    #             'gt_boxes': annos['boxes_3d'],
    #             'num_points_in_gt': annos.get('num_points_in_gt', None)
    #         })
    #
    #     data_dict = self.prepare_data(data_dict=input_dict)
    #     data_dict.pop('num_points_in_gt', None)
    #     return data_dict

class WaymoLabeledDataset(WaymoSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.labeled_data_for = dataset_cfg.LABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN',
                                                      False) and 'num_points_in_gt' in annos:
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.labeled_data_for)
        if teacher_dict is not None: teacher_dict.pop('num_points_in_gt', None)
        if student_dict is not None: student_dict.pop('num_points_in_gt', None)
        return tuple([teacher_dict, student_dict])

class WaymoUnlabeledDataset(WaymoSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.unlabeled_data_for = dataset_cfg.UNLABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_lidar'],
            })
            if 'score' in annos:
                input_dict.update({
                    'gt_scores': annos['score'],
                })

        teacher_dict, student_dict = self.prepare_data_ssl_unlabel(input_dict, output_dicts=self.unlabeled_data_for)
        return tuple([teacher_dict, student_dict])

class WaymoTestDataset(WaymoSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=False, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is False
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
