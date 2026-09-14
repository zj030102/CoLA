import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.delta_fuse_select import DeltaFusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
import torch

class Teacher(nn.Module):
    def __init__(self, args):
        super(Teacher, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = DeltaFusion(args['delta_fusion'])

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

    def forward(self, data_dict):

        right_voxel_features = data_dict['right_processed_lidar']['voxel_features']
        right_voxel_coords = data_dict['right_processed_lidar']['voxel_coords']
        right_voxel_num_points = data_dict['right_processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        right_batch_dict = {'voxel_features': right_voxel_features,
                    'voxel_coords': right_voxel_coords,
                    'voxel_num_points': right_voxel_num_points,
                    'record_len': record_len}
        # n, 4 -> n, c   c是64
        right_batch_dict = self.pillar_vfe(right_batch_dict)
        # n, c -> N, C, H, W
        right_batch_dict = self.scatter(right_batch_dict)
        right_batch_dict = self.backbone(right_batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        right_spatial_features_2d = right_batch_dict['spatial_features_2d']

        # Down-sample feature to reduce memory
        if self.shrink_flag:
            right_spatial_features_2d = self.shrink_conv(right_spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            right_spatial_features_2d = self.naive_compressor(right_spatial_features_2d)

        return right_spatial_features_2d

class PointPillarDeltaFusionAblation(nn.Module):
    def __init__(self, args):
        super(PointPillarDeltaFusionAblation, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = DeltaFusion(args['delta_fusion'])

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()
        
        if 'has_trained' in args:
            self.has_trained = args['has_trained']

            self.teacher = Teacher(args)
            for params in self.teacher.parameters():
                params.requires_grad = False
        else:
            self.has_trained = None

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']


        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c   c是64
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                        spatial_features_2d.shape[2],
                                        spatial_features_2d.shape[3])

        if self.has_trained is not None and self.has_trained is True:
            right_spatial_features_2d = self.teacher(data_dict)
            fused_feature, communication_rates, gram_loss = self.fusion_net(spatial_features_2d,
                                                                    psm_single,
                                                                    record_len,
                                                                    prior_encoding,
                                                                    right_x=right_spatial_features_2d,
                                                                    teacher = self.teacher)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                    psm_single,
                                                                    record_len,
                                                                    prior_encoding)

        # save_heatmap_fig(fused_feature, "clean")

        psm = self.cls_head(fused_feature)
        # np.savetxt('tmp.txt', psm.sigmoid().max(dim=1, keepdim=False)[0].squeeze(0).detach().cpu().numpy())
        rm = self.reg_head(fused_feature)

        if self.has_trained is not None and self.has_trained is True:
            output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates, 'gram_loss': gram_loss} 
        else:
            output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}

        return output_dict


def save_heatmap_fig(fused_feature, map_type):
    # ===================== 正确续写：根据 fused_feature 计算中心余弦热度图 =====================

    import os
    import numpy as np
    from PIL import Image

    # 全局序号（自动顺序保存）
    if not hasattr(torch, 'heatmap_idx'):
        torch.heatmap_idx = 0

    save_dir = f"/home/lkshpc/zj/code/OpenCOOD/opencood/logs/point_pillar_multi_baseline_fully_ablation_select_gram_location_4/heatmap_cos_center_{map_type}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 获取形状 [1, C, H, W]
    B, C, H, W = fused_feature.shape
    feat = fused_feature.squeeze(0)  # [C, H, W]

    # 2. 取出【中心特征点】作为基准
    cy, cx = H // 2, W // 2
    center_feat = feat[:, 1, 1]  # [C]

    # 3. 将特征图展成 [C, H*W]，便于余弦计算
    feat_flat = feat.flatten(1)  # [C, H*W]

    # 4. 计算【所有位置特征 与 中心特征 的余弦相似度】
    dot_product = torch.sum(center_feat.unsqueeze(1) * feat_flat, dim=0)  # [H*W]
    norm_center = torch.norm(center_feat)
    norm_feat = torch.norm(feat_flat, dim=0)
    cos_similarity = dot_product / (norm_center * norm_feat + 1e-8)

    # 5. 还原成 [H, W] 热度图（这就是真实语义特征热度）
    heatmap = cos_similarity.reshape(H, W)

    # 6. 归一化转图像
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = (heatmap + 1.0) / 2.0  # 直接把 [-1,1] 映射到 [0,1]，永久统一！
    # gamma = 1  # 可调：2~4 之间最好
    # heatmap = np.power(heatmap, gamma)
    heatmap = (heatmap * 255).astype(np.uint8)

    # 7. 按顺序保存
    save_path = os.path.join(save_dir, f"{torch.heatmap_idx:03d}.png")
    Image.fromarray(heatmap).save(save_path)
    torch.heatmap_idx += 1

    # ===================== 正确续写：根据 fused_feature 计算中心余弦热度图 =====================


def save_gram_fig(fused_feature, map_type):
    # ===================== 正确续写：根据 fused_feature 计算gram度图 =====================
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F
    # 全局序号（自动顺序保存）
    if not hasattr(torch, 'heatmap_idx'):
        torch.heatmap_idx = 0

    save_dir = f"/home/lkshpc/zj/code/OpenCOOD/opencood/logs/point_pillar_multi_baseline_fully_ablation_select_gram_location_4/gram_{map_type}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 获取形状 [1, C, H, W]
    B, C, H, W = fused_feature.shape
    feat = fused_feature.squeeze(0)  # [C, H, W]

    feat_flat = feat.flatten(1).transpose(0,1) 
    feat_flat = F.normalize(feat_flat, p=2, dim=-1, eps=1e-6).unsqueeze(0) 

    gram = torch.bmm(feat_flat, feat_flat.transpose(1,2)).squeeze(0)  # [HW, HW]

    # 6. 归一化转图像
    gram = gram.detach().cpu().numpy()

    # 7. 按顺序保存
    save_path = os.path.join(save_dir, f"{torch.heatmap_idx:03d}.png")
    plt.imsave(
        save_path,
        gram,
        cmap='jet',       # 彩色配色
        vmin=0, vmax=1    # 固定颜色范围
    )
    torch.heatmap_idx += 1

    # ===================== 正确续写：根据 fused_feature 计算gram图 =====================
