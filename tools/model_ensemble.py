import pickle
import time
import datetime
import argparse
import numpy as np
import torch
import tqdm
import os
from pathlib import Path

from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils.ensemble_boxes_wbf_3d import weighted_nms
import glob

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument("--result_list", required=True, type=str, help="the list of result.pkl, e.g., ../result1.pkl,../result2.pkl")
    parser.add_argument('--for_test', action='store_true', default=False, help='test for the test split')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def wbf(preds, conf_type='avg', weights=None, iou_thr=0.45):
    CLASS_NAMES = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']
    detections = []
    gt_boxes = None
    FileNum = len(preds)
    for i in range(len(preds[0])):
        token = preds[0][i]['frame_id']
        box_list = []
        box_score_list = []
        box_label_list = []
        for iter in range(FileNum):
            pred = preds[iter]

            if (isinstance(pred[i]['boxes_3d'], torch.Tensor)):
                bbox = pred[i]['boxes_3d'].cpu().numpy()
            else:
                bbox = pred[i]['boxes_3d']

            if (isinstance(pred[i]['score'], torch.Tensor)):
                box_score = pred[i]['score'].cpu().numpy()
            else:
                box_score = pred[i]['score']

            box_label = np.stack([CLASS_NAMES.index(ele) for ele in pred[i]['name']])

            box_list.append(bbox)
            box_score_list.append(box_score)
            box_label_list.append(box_label)

        boxes, scores, labels = weighted_nms(box_list, box_score_list, box_label_list, weights=weights,
                                             conf_type=conf_type, iou_thr=iou_thr, skip_box_thr=0.0)

        labels = np.array([CLASS_NAMES[ele] for ele in labels.astype('int64')])
        output = {
            'boxes_3d': boxes,
            'score': scores,
            'name': labels,
            'frame_id': token
        }
        detections.append(output)
    return detections


def eval(cfg, det_annos, dataloader, logger, dist_test=False, save_to_file=False, result_dir=None):

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names


    logger.info('*************** Performance of Ensemble *****************')

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

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

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=None
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('****************Evaluation done.*****************')
    return

if __name__ == '__main__':
    args, cfg = parse_config()

    if args.for_test:
        cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'test'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'model_ensemble' / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    #
    # if ',' in args.result_list:
    #     pred_list = [item.strip() for item in args.result_list.split(',')]
    # else:
    #     pred_list = glob.glob(os.path.join(args.result_list, 'result*.pkl'))
    #
    # for pred_pkl in pred_list:
    #     assert os.path.exists(pred_pkl), "{} not found!".format(pred_pkl)
    #
    # weights = [1] * len(pred_list)
    #
    # preds = []
    # for fileId in range(len(pred_list)):
    #     file_path = pred_list[fileId]
    #     with open(os.path.join(file_path), 'rb') as f:
    #         pred = pickle.load(f)
    #         pred = sorted(pred, key=lambda x: x['frame_id'])
    #         preds.append(pred)
    #
    # detections = wbf(preds, conf_type='avg', weights=weights, iou_thr=0.45)
    #
    # with open(output_dir / 'ensemble_result.pkl', 'wb') as f:
    #     pickle.dump(detections, f)
    # logger.info('Result is save to %s' % output_dir)

    with open("/ssd/junbo/repository/SemiDet3D/output/once_models/sup_models/model_ensemble/trainval_pseudo-test_ensemble-10_val/ensemble_result.pkl", 'rb') as f:
        detections = pickle.load(f)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_test, workers=6, logger=logger, training=False
    )

    eval(cfg, detections, test_loader, logger)
