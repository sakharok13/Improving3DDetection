import torch
import torch.nn as nn
from ...utils import common_utils, loss_utils


class BoxContrastHead(nn.Module):
    def __init__(self, model_cfg, num_class=1, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.CONTRASTIVE_LOSS.ENABLE:
            FC_CHANNELS = self.model_cfg.NUM_FILTERS

            self.embed_fc_layer = nn.Sequential(
                nn.Linear(FC_CHANNELS[0], FC_CHANNELS[1], bias=False),
                nn.BatchNorm1d(FC_CHANNELS[1]),
                nn.ReLU(inplace=True),
                nn.Linear(FC_CHANNELS[2], FC_CHANNELS[2], bias=False),
            )
            self.voxel_range = model_cfg.POINT_CLOUD_RANGE
            self.voxel_size = model_cfg.VOXEL_SIZE

    def get_box_center(self, boxes):

        # box [List]
        centers = []
        for box in boxes:
            if self.num_point == 1 or len(box['box3d_lidar']) == 0:
                centers.append(box['box3d_lidar'][:, :3])

            elif self.num_point == 5:
                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1]) / 2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3]) / 2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3]) / 2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2]) / 2, height], dim=-1)

                points = torch.cat([box['box3d_lidar'][:, :3], front_middle, back_middle, left_middle, \
                                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def bilinear_interpolate_torch(self, im, x, y):
        """
        Args:
            im: (H, W, C) [y, x]
            x: (N)
            y: (N)
        Returns:
        """
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        # d_x0, d_x1, d_y0, d_y1 = x - x0.type_as(x), x1.type_as(x) - x, y - y0.type_as(y), y1.type_as(y) - y
        # wa = d_x1 * d_y1
        # wb = d_x1 * d_y0
        # wc = d_x0 * d_y1
        # wd = d_x0 * d_y0

        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(
            torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
        return ans

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride, voxel=False):
        if not voxel:
            x_idxs = (keypoints[:, :, 0] - self.voxel_range[0]) / self.voxel_size[0]
            y_idxs = (keypoints[:, :, 1] - self.voxel_range[1]) / self.voxel_size[1]
            x_idxs = x_idxs / bev_stride
            y_idxs = y_idxs / bev_stride
        else:
            x_idxs = keypoints[:, :, 2].type_as(bev_features)
            y_idxs = keypoints[:, :, 1].type_as(bev_features)

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = self.bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    @staticmethod
    def _create_buffer(N, s):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8, device=s.device)
        pos_ind = (torch.arange(N * 2, device=s.device),  # for each row
                   2 * torch.arange(N, dtype=torch.long, device=s.device).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze())  # select pos samples
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8, device=s.device)
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        if not self.training:
            return batch_dict

        if not batch_dict['contrastive']:
            return batch_dict

        bev_feature = batch_dict['spatial_features']
        centers = batch_dict['gt_boxes'][:,:,:3]

        z_batch = self.interpolate_from_bev_features(centers, bev_feature, batch_dict['batch_size'], batch_dict['encoded_spconv_tensor_stride'])
        mask = batch_dict['num_gt_list']
        z_list = []
        for i in range(batch_dict['batch_size']):
            z_list.append(z_batch[i][:mask[i]])

        count = 0
        z_interval = []
        for i in range(batch_dict['batch_size']//2):
            z_this = z_list[count:2 * (i + 1)]
            z_this = torch.stack(z_this).transpose(0,1).reshape([-1, self.model_cfg.NUM_FILTERS[0]])
            count = 2 * (i + 1)
            z_interval.append(z_this)
        z_interval = torch.cat(z_interval, dim=0)

        z_box = self.embed_fc_layer(z_interval)
        z_box = nn.functional.normalize(z_box, dim=1, p=2)

        # simclr loss
        N = z_box.size(0) // 2
        s = torch.matmul(z_box, z_box.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, s)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        point_pos = s[pos_ind].unsqueeze(1)  # (2N)x1
        point_neg = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        batch_dict['box_embedding'] = (point_pos, point_neg)
        batch_dict['box_memory'] = z_box

        return batch_dict