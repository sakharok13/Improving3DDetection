import numpy as np
import pickle
import os
from tqdm import tqdm


def merge_two_datainfo(datainfo1, datainfo2, save_path):
    def check_annos(info):
        return 'annos' in info

    #once_save_path = "/home/data_disk/workspace/ONCE_Benchmark_junbo/data/once"
    #datainfo1=os.path.join(once_save_path, "once_infos_train.pkl")
    #datainfo2=os.path.join(once_save_path, "once_infos_raw_mini_large_second_resnet.pkl")
    #save_path=os.path.join(once_save_path, "once_infos_gt_train_pseudo_raw_mini_large_second_resnet.pkl")

    with open(datainfo1, "rb") as F1:
        datainfo1 = pickle.load(F1)
        # datainfo1 = list(filter(check_annos, datainfo1))

    with open(datainfo2, "rb") as F2:
        datainfo2 = pickle.load(F2)

    datainfo_merge = datainfo1 + datainfo2

    with open(save_path, 'wb') as F3:
        pickle.dump(datainfo_merge, F3)
    print('ONCE info file is saved to %s' % (save_path))

def make_once_infos_from_pred(raw_once_infos_path, 
                            predict_path, 
                            output_infos_path, 
                            score_thresh = {'Car':0, 'Bus':0, 'Truck':0, 'Pedestrian':0, 'Cyclist':0.3}):
    def check_annos(info):
        return 'annos' in info

    with open(raw_once_infos_path, "rb") as Fval:
        raw_once_infos = pickle.load(Fval)

    with open(predict_path, "rb") as Fpseu_val:
        pseudo_results = pickle.load(Fpseu_val)

    pseudo_results_map = {}
    for idx, val_result in enumerate(pseudo_results):
        pseudo_results_map[val_result["frame_id"]] = val_result

    for idx, val_info in tqdm(enumerate(raw_once_infos)):
        if check_annos(val_info):
            raw_once_infos[idx].pop("annos")

        raw_once_infos[idx]["annos"] = {}

        pseudo_val = pseudo_results_map[val_info["frame_id"]]

        if score_thresh is not None:

            sample_split_class = {'name': [], 'score': [], 'boxes_3d': []}

            for _, calss_name in enumerate(score_thresh.keys()):
                this_class_mask = pseudo_val['name']==calss_name
                mask = pseudo_val["score"][this_class_mask] > score_thresh[calss_name]
                name = pseudo_val["name"][this_class_mask][mask]
                score = pseudo_val["score"][this_class_mask][mask]
                boxes_3d = pseudo_val["boxes_3d"][this_class_mask][mask]
                sample_split_class['name'].append(name)
                sample_split_class['score'].append(score)
                sample_split_class['boxes_3d'].append(boxes_3d)

            name = np.concatenate(sample_split_class['name'])
            score = np.concatenate(sample_split_class['score'])
            boxes_3d = np.concatenate(sample_split_class['boxes_3d'])
        else:
            name = pseudo_val["name"]
            score = pseudo_val["score"]
            boxes_3d = pseudo_val["boxes_3d"]

        raw_once_infos[idx]["annos"]["name"] = name
        raw_once_infos[idx]["annos"]["score"] = score
        raw_once_infos[idx]["annos"]["boxes_3d"] = boxes_3d
        # raw_once_infos[idx]["annos"]["num_points_in_gt"] = num_points_in_gt

    # save pseudo_val
    with open(output_infos_path, 'wb') as f:
        pickle.dump(raw_once_infos, f)
    print('ONCE info %s file is saved to %s' % ("pseudo_val", output_infos_path))

def make_waymo_infos_from_pred(raw_waymo_infos_path,
                            predict_path,
                            output_infos_path,
                            score_thresh = {'Vehicle':0, 'Pedestrian':0, 'Cyclist':0}):
    def check_annos(info):
        return 'annos' in info

    with open(raw_waymo_infos_path, "rb") as Fval:
        raw_once_infos = pickle.load(Fval)

    with open(predict_path, "rb") as Fpseu_val:
        pseudo_results = pickle.load(Fpseu_val)

    pseudo_results_map = {}
    for idx, val_result in enumerate(pseudo_results):
        pseudo_results_map[val_result["frame_id"]] = val_result

    for idx, val_info in tqdm(enumerate(raw_once_infos)):

        if check_annos(val_info):
            raw_once_infos[idx].pop("annos")

        raw_once_infos[idx]["annos"] = {}

        pseudo_val = pseudo_results_map[val_info["frame_id"]]

        if score_thresh is not None:

            sample_split_class = {'name': [], 'score': [], 'boxes_lidar': []}

            for _, calss_name in enumerate(score_thresh.keys()):
                this_class_mask = pseudo_val['name']==calss_name
                mask = pseudo_val["score"][this_class_mask] > score_thresh[calss_name]
                name = pseudo_val["name"][this_class_mask][mask]
                score = pseudo_val["score"][this_class_mask][mask]
                boxes_3d = pseudo_val["boxes_lidar"][this_class_mask][mask]
                sample_split_class['name'].append(name)
                sample_split_class['score'].append(score)
                sample_split_class['boxes_lidar'].append(boxes_3d)

            name = np.concatenate(sample_split_class['name'])
            score = np.concatenate(sample_split_class['score'])
            boxes_3d = np.concatenate(sample_split_class['boxes_lidar'])
        else:
            name = pseudo_val["name"]
            score = pseudo_val["score"]
            boxes_3d = pseudo_val["boxes_lidar"]

        raw_once_infos[idx]["annos"]["name"] = name
        raw_once_infos[idx]["annos"]["score"] = score
        raw_once_infos[idx]["annos"]["boxes_lidar"] = boxes_3d
        # raw_once_infos[idx]["annos"]["num_points_in_gt"] = num_points_in_gt

    # save pseudo_val
    with open(output_infos_path, 'wb') as f:
        pickle.dump(raw_once_infos, f)
    print('Waymo info %s file is saved to %s' % ("pseudo_val", output_infos_path))

if __name__ == "__main__":
    
    # ONCE
    # make_once_infos_from_pred(raw_once_infos_path='/ssd/junbo/repository/SemiDet3D/data/once/once_infos_raw_large.pkl',
    #                         predict_path="/ssd/junbo/repository/SemiDet3D/output/once_models/sup_models/second/baseline_4GPU_tta_raw/raw_large_tta/baseline_pretrain4GPU_stage2-cyc05_ep150_raw_large_tta_epoch_143_box_refine.result.pkl",
    #                         output_infos_path="/ssd/junbo/repository/SemiDet3D/output/once_models/sup_models/second/baseline_4GPU_tta_raw/raw_large_tta/once_infos_second_4GPU_tta_4stage_cyc03_large.pkl",
    #
    #                           )

    # Waymo
    make_waymo_infos_from_pred(raw_waymo_infos_path='/home/junbo/ssd/repository/SemiDet3D/data/waymo/waymo_infos_train_D1.pkl',
                                predict_path='/home/junbo/ssd/repository/SemiDet3D/output/waymo_models/sup_models/second/baseline_D1_tta_train/eval/epoch_30/val/default/result.pkl',
                                output_infos_path="/home/junbo/ssd/repository/SemiDet3D/data/waymo/second_D1_tta_train-D1.pkl",

                                  )
