import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.ensemble_boxes_wbf_3d import weighted_nms

from nuscenes.utils.data_classes import Box

def random_world_flip(box_preds, params, reverse = False):
    if reverse:
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
    else:
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
    return box_preds

def random_world_rotation(box_preds, params, reverse = False):
    if reverse:
        noise_rotation = -params
    else:
        noise_rotation = params

    angle = torch.tensor([noise_rotation]).to(box_preds.device)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(1)
    ones = angle.new_ones(1)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(3, 3).float()
    box_preds[:, :3] = torch.matmul(box_preds[:, :3], rot_matrix)
    box_preds[:, 6] += noise_rotation
    return box_preds

def tta_wbf(preds, conf_type='avg', weights=None, iou_thr=0.45):

    box_list = []
    box_score_list = []
    box_label_list = []
    for i in range(len(preds)):
        pred = preds[i]

        if isinstance(pred['pred_boxes'], torch.Tensor):
            bbox = pred['pred_boxes'].cpu().numpy()
        else:
            bbox = pred['pred_boxes']

        if isinstance(pred['pred_scores'], torch.Tensor):
            box_score = pred['pred_scores'].cpu().numpy()
        else:
            box_score = pred['pred_scores']

        if isinstance(pred['pred_labels'], torch.Tensor):
            box_label = pred['pred_labels'].cpu().numpy()
        else:
            box_label = pred['pred_labels']

        box_list.append(bbox)
        box_score_list.append(box_score)
        box_label_list.append(box_label)

    boxes, scores, labels = weighted_nms(box_list, box_score_list, box_label_list, weights=weights,
                                         conf_type=conf_type, iou_thr=iou_thr, skip_box_thr=0.0)

    output = {
        'pred_boxes': torch.tensor(boxes),
        'pred_scores': torch.tensor(scores),
        'pred_labels': torch.tensor(labels).long(),
    }
    return [output]

def visualization(points, pred_boxes, token, name='', gt=None):
    import matplotlib.pyplot as plt
    from pyquaternion import Quaternion
    import os
    GT_COLOR = (229, 67, 67)  # (229, 67, 67); (67, 67, 67)
    GT_LINEWIDTH = 2
    DT_COLOR = (67, 67, 67)  # (229, 67, 67); (67, 67, 67)
    DT_LINEWIDTH = 1

    save_dir = '/home/junbo/ssd/repository/SemiDet3D/output/visual/{}'.format(name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if gt is not None:
        gt_boxes = gt.cpu().numpy()

    token = token[0]
    points = points[points[:,0]==0, 1:].cpu()
    # pred_boxes = pred_boxes[0]['pred_boxes']
    if torch.is_tensor(pred_boxes):
        pred_boxes=pred_boxes.detach().cpu().numpy()

    points = points.transpose(1, 0).numpy()
    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / 75 / np.sqrt(2))
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    # if torch.is_tensor(preds['box3d_lidar']):
    #     pred_boxes = preds['box3d_lidar'].cpu().detach().numpy()
    # else:
    #     pred_boxes = preds['box3d_lidar']
    if gt is not None:
        for box in gt_boxes:
            box = Box(center=[box[0], box[1], box[2]], size=[box[4], box[3], box[5]], # lwh
                      orientation=Quaternion(axis=(0, 0, 1), radians=box[-1]))
            c = np.array(GT_COLOR) / 255.0
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=GT_LINEWIDTH)
    for box in pred_boxes:
        box = Box(center=[box[0], box[1], box[2]], size=[box[4], box[3], box[5]],
                  orientation=Quaternion(axis=(0, 0, 1), radians=box[-1]))
        c = np.array(DT_COLOR) / 255.0
        box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=DT_LINEWIDTH)
    axes_limit = 75

    # Limit visible range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    ax.axis('off')
    ax.set_aspect('equal')

    fig.savefig(save_dir + '/{}.png'.format(token), format='png')
    plt.close(fig)

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, result_file=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    if result_file is None:
        if cfg.LOCAL_RANK == 0:
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
        start_time = time.time()
        for i, batch_dict in enumerate(dataloader):
            # add temperature for adpative radius learning
            if cfg.OPTIMIZATION.get('USE_TEMPERATURE', False):
                batch_dict.update({'temperature': cfg.OPTIMIZATION.DECAY_TEMPERATURE[-1]})

            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                # print(batch_dict.keys())
                pred_dicts, ret_dict = model(batch_dict)

            # for i in range(len(pred_dicts)):
            #     pred_dicts[i]['pred_boxes'][:,[3,4]] = pred_dicts[i]['pred_boxes'][:,[4,3]]
            #     pred_dicts[i]['pred_boxes'][:, -1] = -pred_dicts[i]['pred_boxes'][:, -1] - np.pi/2

            # visualization(batch_dict['points'], pred_dicts[0]["pred_boxes"], batch_dict['frame_id'], 'debug_tta',
            #               batch_dict['gt_boxes'][0][:, :-1])
            # todo: reverse

            if cfg.DATA_CONFIG.TEST_AUGMENTOR.ENABLE:
                tta_configs = cfg.DATA_CONFIG.TEST_AUGMENTOR.AUG_CONFIG_LIST
                aug_types = [ele['NAME'] for ele in tta_configs]
                jmax, kmax = 1, 1
                if 'test_world_flip' in aug_types:
                    flip_list = tta_configs[aug_types.index('test_world_flip')]['ALONG_AXIS_LIST']
                    jmax = len(flip_list)+1
                if 'test_world_rotation' in aug_types:
                    rot_list = tta_configs[aug_types.index('test_world_rotation')]['WORLD_ROT_ANGLE']
                    kmax = len(rot_list)

                for j in range(jmax):
                    for k in range(kmax):
                        pred_dict = pred_dicts[j*kmax+k]
                        if 'test_world_rotation' in aug_types:
                            rot = rot_list[k]
                            pred_dict['pred_boxes'] = random_world_rotation(pred_dict['pred_boxes'], rot, reverse=True)
                        if 'test_world_flip' in aug_types:
                            if j == 0:
                                continue
                            flip = flip_list[j-1]
                            pred_dict['pred_boxes'] = random_world_flip(pred_dict['pred_boxes'], flip, reverse=True)
                pred_dicts = tta_wbf(pred_dicts, iou_thr=0.45)
                # visualization(batch_dict['points'], pred_dicts[0]["pred_boxes"], batch_dict['frame_id'], 'debug_kitti', batch_dict['gt_boxes'][0][:,:-1])

            disp_dict = {}

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )

            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

        if cfg.LOCAL_RANK == 0:
            progress_bar.close()

        if dist_test:
            rank, world_size = common_utils.get_dist_info()
            det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
            metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

        logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
        sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
        logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        if cfg.LOCAL_RANK != 0:
            return {}

        ret_dict = {}
        if dist_test:
            for key, val in metric[0].items():
                for k in range(1, world_size):
                    metric[0][key] += metric[k][key]
            metric = metric[0]

        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        total_pred_objects = 0
        for anno in det_annos:
            total_pred_objects += anno['name'].__len__()
        logger.info('Average predicted number of objects(%d samples): %.3f'
                    % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)
        if cfg.MODEL.POST_PROCESSING.EVAL_METRIC == 'pseudo_label':
            exit()
    else:
        with open(result_file, 'rb') as f:
            det_annos = pickle.load(f)
        for det_anno in det_annos:
            if torch.is_tensor(det_anno['score']):
                det_anno['score'] = det_anno['score'].cpu().numpy()
            if 'boxes_3d' in det_anno:
                if torch.is_tensor(det_anno['boxes_3d']):
                    det_anno['boxes_3d'] = det_anno['boxes_3d'].cpu().numpy()
            elif 'boxes_lidar' in det_anno:
                if torch.is_tensor(det_anno['boxes_lidar']):
                    det_anno['boxes_lidar'] = det_anno['boxes_lidar'].cpu().numpy()
        ret_dict = {}
    det_annos = sorted(det_annos, key=lambda x: x['frame_id'])
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_one_epoch_ssl(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=True, result_dir=None, result_file=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)



    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    # start_time = time.time()
    import cv2
    for i, batch_dict in enumerate(dataloader):
        if not i%100 == 0:
            continue
        load_data_to_gpu(batch_dict)
        batch_dict['ssl_mode'] = True
        with torch.no_grad():
            heatmap, points_img = model(batch_dict)

        token = batch_dict['frame_id'][0]
        cv2.imwrite(str(final_output_dir) + '/{}_heatmap.jpg'.format(token), heatmap)
        cv2.imwrite(str(final_output_dir) + '/{}_pcl.jpg'.format(token), points_img[0])

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('****************Evaluation done.*****************')

if __name__ == '__main__':
    pass
