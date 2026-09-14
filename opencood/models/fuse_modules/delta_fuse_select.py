"""
Implementation of Where2comm fusion.
"""
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.models.fuse_modules.delta.pureDelta import RUN_RWKV7_FP32

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0.1, 0.4))
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

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        print(communication_rates)
        communication_masks = torch.cat(communication_masks, dim=0)

        return communication_masks, communication_rates

class PosEncoding2d(nn.Module):
    """
    2D 坐标编码：归一化坐标 -> 256 维
    输入 (h,w) 任意，输出 [1,256,h,w]
    """
    def __init__(self, channels=256):
        super().__init__()
        self.fc = nn.Conv2d(2, channels, 1, 1, 0)   # 2->256

    def forward(self, h, w, device, dtype):
        y = torch.linspace(0, 1, h, dtype=dtype, device=device)
        x = torch.linspace(0, 1, w, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # [h,w]
        coord = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,h,w]
        return self.fc(coord)                              # [1,256,h,w]

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class DeltaFusion(nn.Module):
    def __init__(self, args):
        super(DeltaFusion, self).__init__()

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        self.naive_communication = Communication(args['communication'])

        self.single = DeltaBlock(d_model=args['in_channels'])
        self.norm = nn.LayerNorm(args['in_channels'])

        # self.pos_enc = PosEncoding2d(channels=args['in_channels'])

        self.mul_correction = DeltaBlock(d_model=args['in_channels'])
        self.norm1 = nn.LayerNorm(args['in_channels'])
        self.ffn1 = FeedForward(args['in_channels'], 2*args['in_channels'])

        self.prior_encoder = nn.Sequential(
            nn.Linear(3, args['in_channels'] // 4),         
            nn.ReLU(inplace=True),                           #
            nn.Linear(args['in_channels'] // 4, args['in_channels'])  
        )
        self.beta_encoding = nn.Linear(args['in_channels'] * 2, args['in_channels'])

        self.align = DeltaBlock(d_model=args['in_channels'])

        self.gate = DualGate(C=args['in_channels'])

        self.w = nn.Sequential(
            nn.Linear(args['in_channels']*3, args['in_channels'], bias=False),   
            nn.LayerNorm(args['in_channels']),
            nn.ReLU(),
            nn.Linear(args['in_channels'], args['in_channels']*2, bias=False)
        )

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def get_gram_loss(self, student, teacher):
        l ,seq, _ = student.shape
        student = F.normalize(student, p=2, dim=-1, eps=1e-6)
        teacher = F.normalize(teacher, p=2, dim=-1, eps=1e-6)
        gram_student = torch.bmm(student, student.transpose(1,2))
        gram_teacher = torch.bmm(teacher, teacher.transpose(1,2))

        gram_loss = torch.sum((gram_student - gram_teacher) **2) / (l*seq*seq)
        return gram_loss
        # diff = gram_student - gram_teacher  
        # diff.square_()  
        # gram_loss = torch.sum(diff) / (l*seq*seq)
        
        # return gram_loss

    def forward(self, x, psm_single, record_len, prior_encodings, right_x = None, teacher = None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            prior_encoding B, max_cav, 3(dt dv infra), H, W

        Returns:
            Fused feature.
        """
        N, C, H, W = x.shape
        seq = H * W
        B = record_len.shape[0]

        # Communication (mask the features)
        if self.fully:
            communication_rates = torch.tensor(1).to(x.device)
        else:
            # Prune
            batch_confidence_maps = self.regroup(psm_single, record_len)
            communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
            x = x * communication_masks

        x = snake_flatten(x)   # 蛇形排列可以增加为4个方向，上下左右
        x = self.norm(x)
        x = self.single(x, x, x, x) + x

        x = snake_unflatten(x.permute(0,2,1), H, W) # N,C,H,W

        # Split the features
        # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
        # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
        batch_node_features = self.regroup(x, record_len)
        if right_x is not None:
            right_x = right_x.detach()
            right_x = snake_flatten(right_x)   # 蛇形排列可以增加为4个方向，上下左右
            right_x = teacher.fusion_net.norm(right_x)
            right_x = teacher.fusion_net.single(right_x, right_x, right_x, right_x) + right_x
            right_x = snake_unflatten(right_x.permute(0,2,1), H, W) # N,C,H,W

            right_batch_node_features = self.regroup(right_x, record_len)

            gram_loss_list = []

        # 3. Fusion
        x_fuse = []
        for b in range(B):
            if record_len[b] == 1 or batch_node_features[b].shape[0] <= 1:
                x_fuse.append(batch_node_features[b])
                continue

            # 先验状态编码
            prior_encoding = prior_encodings[b, :record_len[b]]# (L,3,H,W)

            neighbor_feature = batch_node_features[b]   
            L, _, _, _ = neighbor_feature.shape

            # pos = self.pos_enc(H, W, neighbor_feature.device, neighbor_feature.dtype)
            # neighbor_feature = neighbor_feature + pos

            ego = neighbor_feature[0:1]                # (1,C,H,W)
            others = neighbor_feature[1:]              # (L-1,C,H,W)

            ego_prior = prior_encoding[0:1]                # (1,3,H,W)
            others_prior = prior_encoding[1:]              # (L-1,3,H,W)

            ego_flat_unsqueeze = snake_flatten(ego)   # (1, seq, C)
            ego_flat = ego_flat_unsqueeze.squeeze(0)  # (seq, C)
            others_flat = snake_flatten(others)        # (L-1,seq,C)

            ego_prior_flat = snake_flatten(ego_prior).squeeze(0)   # (seq, 3)
            ego_prior_flat = self.prior_encoder(ego_prior_flat)     # (1,seq,C)
            others_prior_flat = snake_flatten(others_prior)        # (L-1,seq,3)
            others_prior_flat = self.prior_encoder(others_prior_flat) # (L-1,seq,C)

            idx_even = torch.arange(0, 2*seq, 2, device=ego_flat.device)  # [0,2,4,...]
            idx_odd  = torch.arange(1, 2*seq, 2, device=ego_flat.device)  # [1,3,5,...]

            out = torch.empty(L-1, 2*seq, C, device=ego_flat.device, dtype=ego_flat.dtype)
            prior_weight = torch.empty(L-1, 2*seq, C, device=ego_flat.device, dtype=ego_flat.dtype)

            for l in range(L-1):
                out[l, idx_even] = ego_flat          # ego 在前（偶数位）
                out[l, idx_odd]  = others_flat[l]    # other 在后（奇数位）

                prior_weight[l, idx_even] = ego_prior_flat
                prior_weight[l, idx_odd]  = others_prior_flat[l]   # (L-1,2*seq,C)
            merged = torch.cat([prior_weight, out], dim=-1)  # (L-1,2*seq, 2*C)
            beta = self.beta_encoding(merged) # (L-1,2*seq, C)
 
            out = self.mul_correction(out, out, out, beta)

            ego_query = ego_flat_unsqueeze.expand(L-1, -1, -1) # (L-1,seq, C)
            non_ego_feats = self.norm1( out[:, idx_odd, :])  # (L-1,seq, C)  
            non_ego_feats = self.ffn1(non_ego_feats)   # (L-1,seq, C)  

            align_feats = self.align(ego_query, non_ego_feats, non_ego_feats, non_ego_feats) # (L-1,seq, C)

            if right_x is not None:
                right_neighbor_feature = right_batch_node_features[b].detach()
                
                right_ego = right_neighbor_feature[0:1]                # (1,C,H,W)
                right_others = right_neighbor_feature[1:]              # (L-1,C,H,W)

                right_ego_flat_unsqueeze = snake_flatten(right_ego)   # (1, seq, C)
                right_ego_flat = right_ego_flat_unsqueeze.squeeze(0)  # (seq, C)
                right_others_flat = snake_flatten(right_others)        # (L-1,seq,C)

                right_out = torch.empty(L-1, 2*seq, C, device=right_ego_flat.device, dtype=right_ego_flat.dtype)
                right_prior_weight = torch.empty(L-1, 2*seq, C, device=right_ego_flat.device, dtype=right_ego_flat.dtype)

                for l in range(L-1):
                    right_out[l, idx_even] = right_ego_flat          # ego 在前（偶数位）
                    right_out[l, idx_odd]  = right_others_flat[l]    # other 在后（奇数位）

                    right_prior_weight[l, idx_even] = ego_prior_flat
                    right_prior_weight[l, idx_odd]  = others_prior_flat[l]   # (L-1,2*seq,C)
                right_merged = torch.cat([right_prior_weight, right_out], dim=-1)  # (L-1,2*seq, 2*C)
                right_beta = teacher.fusion_net.beta_encoding(right_merged) # (L-1,2*seq, C)
    
                right_out = teacher.fusion_net.mul_correction(right_out, right_out, right_out, right_beta)

                right_ego_query = right_ego_flat_unsqueeze.expand(L-1, -1, -1) # (L-1,seq, C)
                right_non_ego_feats = teacher.fusion_net.norm1(right_out[:, idx_odd, :])  # (L-1,seq, C)  
                right_non_ego_feats = teacher.fusion_net.ffn1(right_non_ego_feats)   # (L-1,seq, C)  

                right_align_feats = teacher.fusion_net.align(right_ego_query, right_non_ego_feats, right_non_ego_feats, right_non_ego_feats) # (L-1,seq, C)

                # clean_others = right_batch_node_features[b][1:].detach() # L-1,C,H,W
                # # clean_others = clean_others.view(L-1, C, seq).permute(0, 2, 1) # (L-1,seq, C)
                # clean_others = clean_others.flatten(2).permute(0, 2, 1)
                if align_feats.shape == right_align_feats.shape:
                    right_align_feats = right_align_feats.detach()
                    gram_loss = self.get_gram_loss(align_feats, right_align_feats)
                    gram_loss_list.append(gram_loss)    
                else:
                    print(f'shape error {align_feats.shape} != {right_align_feats.shape}')            

            align_feats = snake_unflatten(align_feats.permute(0,2,1), H, W) # L-1,C,H,W

            noisy_feat = self.gate(align_feats) # 1,C,H,W
            noisy_feat = snake_flatten(noisy_feat)   # (1, seq, C)

            latency = torch.mean(others_prior_flat, dim=0, keepdim=True)  # (1,seq,C)

            cat = torch.cat([latency, ego_flat_unsqueeze, noisy_feat], dim=-1)  # (1,seq,3C)
            weight = self.w(cat) # (1,seq,2C)
            weight = weight.view(1, seq, 2, C)
            weight = F.softmax(weight, dim=2).view(1, seq, -1)

            fuse_feat =  ego_flat_unsqueeze * weight[:, :, 0:C] + noisy_feat * weight[:, :, C: ]   # (1,seq,C) 

            fuse_feat = snake_unflatten(fuse_feat.permute(0,2,1), H, W) # 1,C,H,W

            x_fuse.append(fuse_feat)

        x_fuse = torch.cat(x_fuse, dim=0)

        if right_x is None:
            return x_fuse, communication_rates
        else:
            if len(gram_loss_list) == 0:
                return x_fuse, communication_rates, torch.tensor(0.0, device=x.device, requires_grad=True)
            else:
                return x_fuse, communication_rates, torch.stack(gram_loss_list).mean()


    
class DeltaBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_inner: int = 256,
        num_heads: int = 4,
        conv_size: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.num_heads = num_heads
        self.conv_size = conv_size

        self.head_dim = self.d_inner // num_heads

        assert self.d_inner % num_heads == 0, f"d_model must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(d_model, d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, d_inner, bias=False)

        self.beta_proj = nn.Linear(d_model, d_inner, bias=False) # use vector b (instead of scalar b) for each head
        self.o_norm = nn.LayerNorm(self.head_dim)
        self.o_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        _,t,_ = query.shape
        
        query = self.pad(query)
        key= self.pad(key)
        value = self.pad(value)
        beta = self.pad(beta)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q, k, v= map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), (q, k, v))

        beta = self.beta_proj(beta).sigmoid()

        B,T,H,D = q.shape

        beta = beta.view(B,T,H,D)
        q = F.normalize(q.view(B,T,H,D), dim=-1, p=2.0) # align with delta_rule kernel
        k = F.normalize(k.view(B,T,H,D), dim=-1, p=2.0) # align with delta_rule kernel
        o = RUN_RWKV7_FP32(r=q*(D**(-0.5)), w=q*0-999, k=k, v=beta*v, a=-beta*k, b=k) # align with delta_rule kernel (set w to a very negative number to disable it)

        o = self.o_norm(o.float())
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o[:, -t:, :]
    
    def pad(self, x):
        _, T, _ = x.shape
        pad = (80 - T % 80) % 80        
        if pad == 0:                   
            return x
        else:
            x_pad = F.pad(x, (0, 0, pad, 0)) 
            return x_pad
    

def snake_flatten(x):
    """
    x: tensor of shape (N,C,H,W) or (C,H,W)
    return: 每通道按蛇形排列
    """
    N, C, H, W = x.shape

    col = torch.arange(W, device=x.device)
    row = torch.arange(H, device=x.device)
    mask = row % 2 == 1          
    idx = col.view(1, W).expand(H, W).clone()
    idx[mask] = idx[mask].flip(dims=[1])   
    idx = idx + row.view(H, 1) * W       
    idx = idx.view(-1)                

    x = x.view(N, C, H*W)
    return x[..., idx].permute(0, 2, 1)  # N,L,C



def snake_unflatten(flat, H, W):
    """
    flat: tensor of shape (..., H*W)  # 最后维度是蛇形flatten结果
    H,W : 原始空间高宽
    return: (..., H, W)
    """

    col = torch.arange(W, device=flat.device)
    row = torch.arange(H, device=flat.device)
    mask = row % 2 == 1
    idx = col.view(1, W).expand(H, W).clone()
    idx[mask] = idx[mask].flip(dims=[1])
    idx = (idx + row.view(H, 1) * W).view(-1) 

    inv_idx = torch.empty_like(idx)
    inv_idx[idx] = torch.arange(H * W, device=flat.device)

    return flat[..., inv_idx].view(*flat.shape[:-1], H, W)

    

class DualGate(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.spatial_gate = nn.Conv2d(C, 1, 3, 1, 1)   # 每帧一个 0~1  mask
        self.channel_gate = nn.Conv2d(C, C, 1)         # 每帧一个 C-dim 权重
        self.ln = nn.LayerNorm(C)
    def forward(self, x):                              # x: [L, C, H, W]
        L, C, H, W = x.shape
        w_s = torch.softmax(self.spatial_gate(x), dim=0)   # [L,1,H,W] 沿 L 归一

        c = self.channel_gate(x)                           # [L,C,H,W]
        c = c.permute(0, 2, 3, 1).reshape(L*H*W, C)        # [LHW,C]
        w_c = self.ln(c).sigmoid().reshape(L, H, W, C).permute(0, 3, 1, 2)  # [L,C,H,W]
        w_c = w_c.mean(dim=[2, 3], keepdim=True)           # [L,C,1,1]

        gated = x * w_s * w_c
        return  gated.sum(0).unsqueeze(0) # [1,C,H,W]