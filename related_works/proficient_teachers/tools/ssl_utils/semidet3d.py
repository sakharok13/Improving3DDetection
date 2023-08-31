import torch
import torch.nn.functional as F
import numpy as np
from .semi_utils import reverse_transform, load_data_to_gpu, filter_boxes, construct_pseudo_label
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms
from tools.eval_utils.eval_utils import visualization


def get_consistency_loss(teacher_boxes, student_boxes):
    center_losses, size_losses, cls_losses = [], [], []
    batch_normalizer = 0
    if 'pred_cls_preds' in teacher_boxes[0]:  # SECOND
        for teacher_box, student_box in zip(teacher_boxes, student_boxes):
            teacher_cls_preds = teacher_box['pred_cls_preds'].detach_()
            teacher_box_preds = teacher_box['pred_boxes'].detach_()
            student_cls_preds = student_box['pred_cls_preds']
            student_box_preds = student_box['pred_boxes']
            num_teacher_boxes = teacher_box_preds.shape[0]
            num_student_boxes = student_box_preds.shape[0]
            if num_teacher_boxes == 0 or num_student_boxes == 0:
                batch_normalizer += 1
                continue

            teacher_centers, teacher_sizes, teacher_rot = teacher_box_preds[:, :3], teacher_box_preds[:, 3:6], teacher_box_preds[:, [6]]
            student_centers, student_sizes, student_rot = student_box_preds[:, :3], student_box_preds[:, 3:6], student_box_preds[:, [6]]

            with torch.no_grad():
                teacher_class = torch.max(teacher_cls_preds, dim=-1, keepdim=True)[1] # [Nt, 1]
                student_class = torch.max(student_cls_preds, dim=-1, keepdim=True)[1] # [Ns, 1]
                not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]
                MAX_DISTANCE = 1000000
                dist = teacher_centers[:, None, :] - student_centers[None, :, :] # [Nt, Ns, 3]
                dist = (dist ** 2).sum(-1) # [Nt, Ns]
                dist += not_same_class * MAX_DISTANCE # penalty on different classes
                student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
                teacher_dist_of_student, teacher_index_of_student = dist.min(0) # [Ns]
                # different from standard sess, we only consider distance<1m as matching
                MATCHED_DISTANCE = 1
                matched_teacher_mask = (teacher_dist_of_student < MATCHED_DISTANCE).float().unsqueeze(-1) # [Ns, 1]
                matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1) # [Nt, 1]

            matched_teacher_centers = teacher_centers[teacher_index_of_student] # [Ns, :]
            matched_student_centers = student_centers[student_index_of_teacher] # [Nt, :]

            matched_student_sizes = student_sizes[student_index_of_teacher] # [Nt, :]
            matched_student_cls_preds = student_cls_preds[student_index_of_teacher] # [Nt, :]

            center_loss = (((student_centers - matched_teacher_centers) * matched_teacher_mask).abs().sum()
                           + ((teacher_centers - matched_student_centers) * matched_student_mask).abs().sum()) \
                          / (num_teacher_boxes + num_student_boxes)
            size_loss = F.mse_loss(matched_student_sizes, teacher_sizes, reduction='none')
            size_loss = (size_loss * matched_student_mask).sum() / num_teacher_boxes

            # kl_div is not feasible, since we use sigmoid instead of softmax for class prediction
            # cls_loss = F.kl_div(matched_student_cls_preds.log(), teacher_cls_preds, reduction='none')
            cls_loss = F.mse_loss(matched_student_cls_preds, teacher_cls_preds, reduction='none') # use mse loss instead
            cls_loss = (cls_loss * matched_student_mask).sum() / num_teacher_boxes

            center_losses.append(center_loss)
            size_losses.append(size_loss)
            cls_losses.append(cls_loss)
            batch_normalizer += 1

    elif 'pred_scores' in teacher_boxes[0]:  # CenterPoint
        for teacher_box, student_box in zip(teacher_boxes, student_boxes):
            teacher_cls_preds = teacher_box['pred_scores'].detach_().unsqueeze(dim=-1)
            teacher_box_preds = teacher_box['pred_boxes'].detach_()
            student_cls_preds = student_box['pred_scores'].unsqueeze(dim=-1)
            student_box_preds = student_box['pred_boxes']
            num_teacher_boxes = teacher_box_preds.shape[0]
            num_student_boxes = student_box_preds.shape[0]
            if num_teacher_boxes == 0 or num_student_boxes == 0:
                batch_normalizer += 1
                continue

            teacher_centers, teacher_sizes, teacher_rot = teacher_box_preds[:, :3], teacher_box_preds[:, 3:6], teacher_box_preds[:, [6]]
            student_centers, student_sizes, student_rot = student_box_preds[:, :3], student_box_preds[:, 3:6], student_box_preds[:, [6]]

            with torch.no_grad():
                teacher_class = teacher_box['pred_labels'].unsqueeze(dim=-1) #torch.max(teacher_cls_preds, dim=-1, keepdim=True)[1] # [Nt, 1]
                student_class = student_box['pred_labels'].unsqueeze(dim=-1) #torch.max(student_cls_preds, dim=-1, keepdim=True)[1] # [Ns, 1]
                not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]
                MAX_DISTANCE = 1000000
                dist = teacher_centers[:, None, :] - student_centers[None, :, :] # [Nt, Ns, 3]
                dist = (dist ** 2).sum(-1) # [Nt, Ns]
                dist += not_same_class * MAX_DISTANCE # penalty on different classes
                student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
                teacher_dist_of_student, teacher_index_of_student = dist.min(0) # [Ns]
                # different from standard sess, we only consider distance<1m as matching
                MATCHED_DISTANCE = 1
                matched_teacher_mask = (teacher_dist_of_student < MATCHED_DISTANCE).float().unsqueeze(-1) # [Ns, 1]
                matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1) # [Nt, 1]

            matched_teacher_centers = teacher_centers[teacher_index_of_student] # [Ns, :]
            matched_student_centers = student_centers[student_index_of_teacher] # [Nt, :]

            matched_student_sizes = student_sizes[student_index_of_teacher] # [Nt, :]
            matched_student_cls_preds = student_cls_preds[student_index_of_teacher] # [Nt, :]

            center_loss = (((student_centers - matched_teacher_centers) * matched_teacher_mask).abs().sum()
                           + ((teacher_centers - matched_student_centers) * matched_student_mask).abs().sum()) \
                          / (num_teacher_boxes + num_student_boxes)
            size_loss = F.mse_loss(matched_student_sizes, teacher_sizes, reduction='none')
            size_loss = (size_loss * matched_student_mask).sum() / num_teacher_boxes

            # kl_div is not feasible, since we use sigmoid instead of softmax for class prediction
            # cls_loss = F.kl_div(matched_student_cls_preds.log(), teacher_cls_preds, reduction='none')
            cls_loss = F.mse_loss(matched_student_cls_preds, teacher_cls_preds, reduction='none') # use mse loss instead
            # cls_loss = 1-teacher_cls_preds
            cls_loss = (cls_loss * matched_student_mask).sum() / num_teacher_boxes

            center_losses.append(center_loss)
            size_losses.append(size_loss)
            cls_losses.append(cls_loss)
            batch_normalizer += 1

    return sum(center_losses)/batch_normalizer, sum(size_losses)/batch_normalizer, sum(cls_losses)/batch_normalizer

def sigmoid_rampup(current, rampup_start, rampup_end):
    assert rampup_start <= rampup_end
    if current < rampup_start:
        return 0
    elif (current >= rampup_start) and (current < rampup_end):
        rampup_length = max(rampup_end, 0) - max(rampup_start, 0)
        if rampup_length == 0: # no rampup
            return 1
        else:
            phase = 1.0 - (current - max(rampup_start, 0)) / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    elif current >= rampup_end:
        return 1
    else:
        raise Exception('Impossible condition for sigmoid rampup')

def semidet3d_pseudo(teacher_model, student_model,
         ld_teacher_batch_dict, ld_student_batch_dict,
         ud_teacher_batch_dict, ud_student_batch_dict,
         cfgs, epoch_id, dist
        ):

    load_data_to_gpu(ld_teacher_batch_dict)
    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)

    # pseudo label
    if not dist:
        ld_teacher_batch_dict = teacher_model(ld_teacher_batch_dict)
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.post_processing(ud_teacher_batch_dict)
    else:
        ld_teacher_batch_dict, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.module.onepass.post_processing(ud_teacher_batch_dict)

    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    gt_boxes = construct_pseudo_label(teacher_boxes)
    ud_student_batch_dict['gt_boxes'] = gt_boxes

    if not dist:
        ld_student_batch_dict, ld_ret_dict, ld_tb_dict, ld_disp_dict = student_model(ld_student_batch_dict)
        ud_student_batch_dict, ud_ret_dict, ud_tb_dict, ud_disp_dict = student_model(ud_student_batch_dict)
    else:
        (ld_student_batch_dict, ld_ret_dict, ld_tb_dict, ld_disp_dict), (ud_student_batch_dict, ud_ret_dict, ud_tb_dict, ud_disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    for k in ld_tb_dict.keys():
        ld_tb_dict['ld_'+k] = ld_tb_dict[k]
        del  ld_tb_dict[k]
    for k in ud_tb_dict.keys():
        ud_tb_dict['ud_' + k] = ud_tb_dict[k]
        del ud_tb_dict[k]
    ld_tb_dict.update(ud_tb_dict)
    return loss, ld_tb_dict, ld_disp_dict


def semidet3d_pseudo_consis(teacher_model, student_model,
         ld_teacher_batch_dict, ld_student_batch_dict,
         ud_teacher_batch_dict, ud_student_batch_dict,
         cfgs, epoch_id, dist
        ):

    load_data_to_gpu(ld_teacher_batch_dict)
    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)

    # pseudo label
    if not dist:
        ld_teacher_batch_dict = teacher_model(ld_teacher_batch_dict)
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.post_processing(ud_teacher_batch_dict)
    else:
        ld_teacher_batch_dict, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.module.onepass.post_processing(ud_teacher_batch_dict)

    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    gt_boxes = construct_pseudo_label(teacher_boxes)
    ud_student_batch_dict['gt_boxes'] = gt_boxes

    if not dist:
        ld_student_batch_dict, ld_ret_dict, ld_tb_dict, ld_disp_dict = student_model(ld_student_batch_dict)
        ud_student_batch_dict, ud_ret_dict, ud_tb_dict, ud_disp_dict = student_model(ud_student_batch_dict)
    else:
        (ld_student_batch_dict, ld_ret_dict, ld_tb_dict, ld_disp_dict), (ud_student_batch_dict, ud_ret_dict, ud_tb_dict, ud_disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    det_loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    # consistency
    ld_teacher_boxes = filter_boxes(ld_teacher_batch_dict, cfgs)
    ud_teacher_boxes = filter_boxes(ud_teacher_batch_dict, cfgs)
    ld_student_boxes = filter_boxes(ld_student_batch_dict, cfgs)
    ud_student_boxes = filter_boxes(ud_student_batch_dict, cfgs)

    ld_teacher_boxes = reverse_transform(ld_teacher_boxes, ld_teacher_batch_dict, ld_student_batch_dict)
    ud_teacher_boxes = reverse_transform(ud_teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)

    ld_center_loss, ld_size_loss, ld_cls_loss = get_consistency_loss(ld_teacher_boxes, ld_student_boxes)
    ud_center_loss, ud_size_loss, ud_cls_loss = get_consistency_loss(ud_teacher_boxes, ud_student_boxes)

    consistency_loss = (ld_cls_loss + ud_cls_loss) * cfgs.CLASS_WEIGHT
    consistency_weight = cfgs.CONSISTENCY_WEIGHT * sigmoid_rampup(epoch_id, cfgs.TEACHER.EMA_EPOCH[0],
                                                                  cfgs.TEACHER.EMA_EPOCH[1])

    loss = det_loss + consistency_weight * consistency_loss

    for k in ld_tb_dict.keys():
        ld_tb_dict['ld_'+k] = ld_tb_dict[k]
        del  ld_tb_dict[k]
    for k in ud_tb_dict.keys():
        ud_tb_dict['ud_' + k] = ud_tb_dict[k]
        del ud_tb_dict[k]
    ld_tb_dict.update(ud_tb_dict)
    ld_tb_dict.update({'consistency_loss':consistency_loss.item()})
    return loss, ld_tb_dict, ld_disp_dict
