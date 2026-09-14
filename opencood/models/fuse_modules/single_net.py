import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = dim ** 0.5

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class SingleAttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(SingleAttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        """
        x:           [B, C, H, W]  每个批次只含 1 辆车
        record_len:  [B]           元素恒为 B（单 vehicle）
        return:      [B, C, H, W]  像素级自注意力后特征
        """
        B, C, H, W = x.shape

        x_seq = x.view(B, C, -1).permute(0, 2, 1)          # [B, H*W, C]
        out_seq = self.att(x_seq, x_seq, x_seq)            # [B, H*W, C]
        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        return out