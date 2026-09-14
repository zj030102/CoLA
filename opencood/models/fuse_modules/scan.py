import torch
import torch.nn as nn
import math

class RearFan120Sampler(nn.Module):
    """
    120° 后扇形可变形采样器
    --------------------------------
    输入：
        offset_pred : (B, n_heads*K*2, H, W)  网络预测的相对偏移（像素单位）
        heading_pred: (B, 1, H, W)            模型预测的朝向角（弧度，车头=0）
        K           : 每个 query 的采样点数量（建议 9 或 16）
    输出：
        sample_coords : (B, n_heads, K, H, W, 2)  归一化坐标 [-1,1]，可直接喂 grid_sample
    """

    def __init__(self, n_heads=8, K=9, fan_deg=120):
        super().__init__()
        self.n_heads = n_heads
        self.K = K
        self.fan_deg = fan_deg
        self.register_buffer("fan_coords", self._build_fan_coords(K, fan_deg))  # (K, 2)

    # ---------- 1. 在“车头=+X”坐标系里生成 120° 后扇形 ----------
    def _build_fan_coords(self, K, fan_deg):
        rad_step = math.radians(fan_deg) / (K - 1)          # 扇形步长
        start_angle = math.radians(180 - fan_deg / 2)       # 扇形左端
        angles = start_angle + rad_step * torch.arange(K, dtype=torch.float32)
        # 同时保留“原点”——把角度=0 插到第 0 位
        angles = torch.cat([torch.zeros(1), angles])[:K]    # 长度仍为 K
        # 单位圆上取点（半径=1）
        coords = torch.stack([torch.cos(angles), torch.sin(angles)], -1)  # (K, 2)
        return coords                                        # 归一化到 1 像素半径

    # ---------- 2. 把扇形转到“车尾方向”，并广播到全图 ----------
    def forward(self, offset_pred, heading_pred):
        B, _, H, W = offset_pred.shape
        device = offset_pred.device
        K = self.K

        # 2.1 拆偏移：相对 (dx, dy) 单位=像素
        dx_dy = offset_pred.view(B, self.n_heads, K, 2, H, W)  # (B, heads, K, 2, H, W)

        # 2.2 计算旋转矩阵：车头→车尾（+π）
        theta = heading_pred.squeeze(1) + math.pi             # (B, H, W)
        cos_t = torch.cos(theta)                              # (B, H, W)
        sin_t = torch.sin(theta)
        # 2×2 旋转矩阵元素
        R11 = cos_t;  R12 = -sin_t
        R21 = sin_t;  R22 = cos_t                               # 所有形状 (B, H, W)

        # 2.3 把扇形坐标转到车尾坐标系，并放大到“像素半径”
        fan_xy = self.fan_coords.to(device)                   # (K, 2)
        fan_x = fan_xy[:, 0].view(1, 1, K, 1, 1)              # (1, 1, K, 1, 1)
        fan_y = fan_xy[:, 1].view(1, 1, K, 1, 1)
        # 旋转 + 半径=1 像素
        rear_x = R11.view(B, 1, 1, H, W) * fan_x + R12.view(B, 1, 1, H, W) * fan_y
        rear_y = R21.view(B, 1, 1, H, W) * fan_x + R22.view(B, 1, 1, H, W) * fan_y
        # 现在 rear_x/y 形状：(B, 1, K, H, W)  单位=像素

        # 2.4 加上网络预测的相对偏移
        final_dx = rear_x + dx_dy[:, :, :, 0]                 # (B, heads, K, H, W)
        final_dy = rear_y + dx_dy[:, :, :, 1]

        # 2.5 转成归一化坐标 [-1,1] 供 grid_sample
        norm_x = final_dx / (W - 1) * 2 - 1
        norm_y = final_dy / (H - 1) * 2 - 1
        sample_coords = torch.stack([norm_x, norm_y], -1)     # (B, heads, K, H, W, 2)

        return sample_coords
    


class DeformAttnWithRearFan(nn.Module):
    def __init__(self, in_ch, n_heads=8, K=9):
        super().__init__()
        self.offset_gen = OffsetGenerator(in_ch, n_heads, K)   # 你已有的 offset 生成器
        self.sampler = RearFan120Sampler(n_heads, K)

    def forward(self, x, heading):
        offset = self.offset_gen(x)                    # (B, n_heads*K*2, H, W)
        coords = self.sampler(offset, heading)         # (B, n_heads, K, H, W, 2)
        #  reshape 给 grid_sample
        B, n_heads, K, H, W, _ = coords.shape
        coords = coords.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, heads, H, W, K, 2)
        coords = coords.view(B * n_heads * H * W, K, 2)         # (N, K, 2)

        # 把 x 也展开成 (N, C, 1, 1) 供 grid_sample
        x = x.view(B * n_heads, C // n_heads, H, W)
        sampled = F.grid_sample(x, coords, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        # 再 reshape 回下游所需形状即可
        return sampled.view(B, n_heads, K, C // n_heads, H, W)