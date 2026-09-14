import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix, use_warp_feature=True):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            if use_warp_feature:
                neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                t_matrix[i, :, :, :],
                                                (H, W))
            else:
                neighbor_feature = batch_node_features[b]
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out

class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix, use_warp_feature=True):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        xx : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            if use_warp_feature:
                x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            else:
                x = batch_node_features[b]
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out

class SimplePointPillarScatter(nn.Module):
    def __init__(self, feature_dim, grid_size):
        super().__init__()

        self.num_bev_features = feature_dim
        self.nx, self.ny, self.nz = grid_size  # [704, 200, 1] 

        assert self.nz == 1

    def forward(self, pillar_features, coords):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
            pillars = pillars.t() # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
            batch_spatial_features.append(spatial_feature) 

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]

        return batch_spatial_features



class Where2comm(nn.Module):
    def __init__(self, args, dim):
        super(Where2comm, self).__init__()

        self.fully = args['fully']

        if args['fusion'] == 'att':
            self.fuse_modules = AttFusion(dim)
        elif args['fusion'] == 'max':
            self.fuse_modules = MaxFusion()
        
        self.naive_communication = Communication(args['communication'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, psm_single, record_len, normalized_affine_matrix, req_mask=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            normalized_affine_matrix : torch.Tensor
                The normalized affine transformation matrix from each cav to ego, 
                shape: (B, L, L, 2, 3) 

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]
        
        # warp the confidence map
        batch_node_features = self.regroup(x, record_len)
        batch_confidence_maps = self.regroup(psm_single, record_len)
        batch_warp_x = []
        batch_warp_confidence_maps = []
        batch_warp_maks_list = []
        
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0
            confidence_map_L, _, confidence_map_H, confidence_map_W = batch_confidence_maps[b].shape
            warp_mask = torch.ones((confidence_map_L, 1, confidence_map_H, confidence_map_W)).to(x.device)
            warp_mask = warp_affine_simple(warp_mask, t_matrix[i, :, :, :], (H, W))
            confidence_map = warp_affine_simple(batch_confidence_maps[b],
                                                t_matrix[i, :, :, :], (H, W))
            warp_x = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :], (H, W))
            batch_warp_confidence_maps.append(confidence_map)
            batch_warp_x.append(warp_x)
            batch_warp_maks_list.append(warp_mask)
        warp_x = torch.cat(batch_warp_x, dim=0)
        
        # Prune
        communication_masks, \
            communication_rates = self.naive_communication(batch_warp_confidence_maps, B,
                                                            batch_warp_maks_list, req_mask)
        
        # mask the features
        if self.fully:
            communication_masks = torch.tensor(1).to(warp_x.device)
        else:
            if warp_x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(
                    communication_masks, size=(warp_x.shape[-2], warp_x.shape[-1]),
                    mode='bilinear', align_corners=False)
            warp_x = warp_x * communication_masks
        
        x_out = self.fuse_modules(warp_x, record_len,
                                  normalized_affine_matrix, use_warp_feature=False)
        return x_out, communication_rates


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.k_ratio = 0
        self.threshold = 0
        if 'k_ratio' in args:
            self.k_ratio = args['k_ratio']
        if 'threshold' in args:
            self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1,
                                             padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) +
                                                             np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B, batch_warp_maks_list, req_mask=None):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
            batch_warp_maks_list: [(1, H, W), (1, H, W), ...], used to mask padding areas
        """

        _, _, H, W = batch_confidence_maps[0].shape
        if req_mask is not None:
            _, H_points_mask, W_points_mask = req_mask[0].shape
            if H_points_mask != H or W_points_mask != W:
                req_mask = F.interpolate(req_mask, size=(H, W), mode='nearest')

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            # Note: If there is no warp mask, the padding value will be 0.5
            # and it will affect the selection of communication mask!
            ori_communication_maps = ori_communication_maps * batch_warp_maks_list[b]
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.k_ratio:
                K = int(H * W * self.k_ratio)
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            if req_mask is not None:
                communication_mask = req_mask[b] * communication_mask
            
            if L > 1:
                communication_rate = communication_mask[1:].sum() / ((L - 1) * H * W)
            else:
                communication_rate = 0.0
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates
