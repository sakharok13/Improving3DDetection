import torch
from .semi_utils import reverse_transform, load_data_to_gpu, construct_pseudo_label, construct_pseudo_label_soft, reverse_transform_tta
import copy

def pseudo_label(teacher_model, student_model,
                  ld_teacher_batch_dict, ld_student_batch_dict,
                  ud_teacher_batch_dict, ud_student_batch_dict,
                  cfgs, epoch_id, dist
                 ):
    assert ld_teacher_batch_dict is None # Only generate labels for unlabeled data

    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)

    if not dist:
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.post_processing(ud_teacher_batch_dict)
    else:
        _, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.module.onepass.post_processing(ud_teacher_batch_dict)

    ud_teacher_batch_dict['augmentation_list'] = [[] for i in range(len(teacher_boxes))]
    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    gt_boxes = construct_pseudo_label_soft(teacher_boxes)
    ud_student_batch_dict['gt_boxes'] = gt_boxes[0]

    if not dist:
        _, ld_ret_dict, _, _ = student_model(ld_student_batch_dict)
        _, ud_ret_dict, tb_dict, disp_dict = student_model(ud_student_batch_dict)
    else:
        (_, ld_ret_dict, _, _), (_, ud_ret_dict, tb_dict, disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    return loss, tb_dict, disp_dict


def proficient_teachers(teacher_model, student_model,     # offline teacher
                  ld_teacher_batch_dict, ld_student_batch_dict,
                  ud_teacher_batch_dict, ud_student_batch_dict,
                  cfgs, epoch_id, dist
                 ):
    assert ld_teacher_batch_dict is None # Only generate labels for unlabeled data

    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)

    batch_pseudo_boxes = ud_teacher_batch_dict['gt_boxes']
    teacher_boxes = []
    for i in range(len(batch_pseudo_boxes)):  # batch to list
        cur_gt = batch_pseudo_boxes[i]
        cnt = cur_gt.__len__() - 1
        while cnt > 0 and cur_gt[cnt].sum() == 0:
            cnt -= 1
        cur_gt = cur_gt[:cnt + 1]
        if batch_pseudo_boxes.shape[-1] == 9:
            final_boxes = cur_gt[:, :-2]
            final_labels = cur_gt[:, -2]
            final_scores = cur_gt[:, -1]
        else:
            final_boxes = cur_gt[:,:-1]
            final_labels = cur_gt[:, -1]
            final_scores = torch.ones(len(cur_gt)).type_as(cur_gt)
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels,
        }
        teacher_boxes.append(record_dict)

    # prepare teacher label
    ud_teacher_batch_dict.pop('voxels')
    ud_teacher_batch_dict.pop('voxel_coords')
    ud_teacher_batch_dict.pop('voxel_num_points')
    ud_teacher_batch_dict['augmentation_list'] = [[] for i in range(len(teacher_boxes)*2)]
    ud_teacher_batch_dict['augmentation_params'] = [val for val in ud_teacher_batch_dict['augmentation_params'] for i in range(2)]

    if cfgs.STUDENT.CONTRASTIVE:
        teacher_boxes = [copy.deepcopy(val) for val in teacher_boxes for i in range(2)]

    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    gt_boxes, num_gt_list = construct_pseudo_label_soft(teacher_boxes)
    ud_student_batch_dict['gt_boxes'] = gt_boxes

    ld_student_batch_dict['contrastive'] = False
    ud_student_batch_dict['contrastive'] = False #cfgs.STUDENT.CONTRASTIVE
    ud_student_batch_dict['num_gt_list'] = num_gt_list

    if not dist:
        _, ld_ret_dict, _, _ = student_model(ld_student_batch_dict)
        _, ud_ret_dict, tb_dict, disp_dict = student_model(ud_student_batch_dict)
    else:
        (_, ld_ret_dict, _, _), (_, ud_ret_dict, tb_dict, disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    if ud_student_batch_dict['contrastive']:
        loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean() + ud_ret_dict['contrastive_loss']*0.005
        tb_dict.update({'contrastive_loss':ud_ret_dict['contrastive_loss'].item()})
    else:
        loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    return loss, tb_dict, disp_dict

