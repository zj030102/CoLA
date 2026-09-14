
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
import torch


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class FlowBaseAttFusion(nn.Module):
    def __init__(self, args):
        super(FlowBaseAttFusion, self).__init__()
        self.att = ScaledDotProductAttention(args['in_channels'])

    def forward(self, x):
        C, W, H = x[0].shape[1:]
        out = []
        for xx in x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            out.append(h)
        return torch.stack(out)

class ReduceInfTC(nn.Module):
    def __init__(self, channel):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(channel//2, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(channel//4, channel//8, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(channel//8, channel//64, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(channel//64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(channel//64, channel//8, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(channel//8, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn2_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(channel//4, channel//4, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.bn2_3 = nn.BatchNorm2d(channel//4, track_running_stats=True)

    def forward(self, x):
        outputsize = x.shape
        #out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))
         
        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))
        #x_1 = F.relu(self.bn2_4(self.deconv2_4(out)))
        return x_1

class FlowGenerator(nn.Module):
    def __init__(self, args):
        super(FlowGenerator, self).__init__()

        self.predictor = ReduceInfTC(args['fusion_args']['in_channels']*4)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def forward(self, regroup_feature_t0, regroup_feature_t1):

        flow_pred_list = []
        for b in range(len(regroup_feature_t0)):
            regroup_feature = torch.cat([regroup_feature_t0[b][1:], regroup_feature_t1[b][1:]], dim=1)
            regroup_feature = self.predictor(regroup_feature)
            flow_pred_list.append(regroup_feature)

        return flow_pred_list
        
