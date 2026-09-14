import torch.nn as nn
import math
import torch
from einops import rearrange, repeat
import torch.nn.functional as F


from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
    

class MambaVision(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=256, 
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv # 暂且无用
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)  

        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True


        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner//2))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        b, c, h, w = hidden_states.shape

        hidden_states = hidden_states.view(b, c, -1).permute(0, 2, 1).contiguous()          # [B, H*W, C]

        _, seqlen, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return out


import torch.optim as optim
# 导入你定义的 MambaVisionMixer 和 SelectiveScanFn 类
# （确保 selective_scan_cuda 已正确编译并可导入，若为CPU环境需替换为CPU版本的选择性扫描实现）


def main():
    # -------------------------- 1. 测试配置 --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")
    
    # 模型核心参数（与你定义的 MambaVisionMixer 初始化参数匹配）
    d_model = 256       # 输入/输出特征维度
    d_state = 256       # 状态空间维度（你设置的自定义值）
    d_conv = 4          # 卷积核大小（当前代码暂未使用，不影响测试）
    expand = 2          # 通道扩展因子
    batch_size = 2      # 批量大小（不宜过大，避免显存不足）
    seq_len = 128       # 序列长度（模拟长图像序列展开后的token数）
    num_steps = 3       # 反向传播测试的训练步数
    
    print(f"\n测试参数:")
    print(f"- d_model: {d_model}, d_state: {d_state}, expand: {expand}")
    print(f"- batch_size: {batch_size}, seq_len: {seq_len}, num_steps: {num_steps}")


    # -------------------------- 2. 初始化模型与输入 --------------------------
    try:
        # 初始化 MambaVisionMixer 模型
        model = MambaVision(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank="auto",  # 按 d_model/16 自动计算（d_model=256 时 dt_rank=16）
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            bias=False
        ).to(device)
        print(f"\n模型初始化成功！模型设备: {next(model.parameters()).device}")
        
        # 生成随机测试输入（形状：(B, L, D) = (batch_size, seq_len, d_model)）
        x = torch.randn(batch_size, d_model, 16, 32,device=device, requires_grad=False)
        print(f"输入张量形状: {x.shape}, 输入设备: {x.device}")

    except Exception as e:
        print(f"\n初始化失败！错误信息: {str(e)}")
        return


    # -------------------------- 3. 前向传播测试 --------------------------
    print(f"\n=== 开始前向传播测试 ===")
    try:
        # 前向计算
        with torch.no_grad():  # 先关闭梯度，仅验证前向输出
            output = model(x)
        
        # 验证输出形状（需与输入形状一致：(B, L, D)）
        assert output.shape == x.shape, \
            f"前向输出形状错误！预期 {x.shape}，实际 {output.shape}"
        
        # 验证输出设备（需与输入设备一致）
        assert output.device == x.device, \
            f"前向输出设备错误！预期 {x.device}，实际 {output.device}"
        
        print(f"前向传播成功！")
        print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
        print(f"输出张量范围: [{output.min():.4f}, {output.max():.4f}]")

    except Exception as e:
        print(f"前向传播失败！错误信息: {str(e)}")
        return


    # -------------------------- 4. 反向传播测试 --------------------------
    print(f"\n=== 开始反向传播测试 ===")
    try:
        # 初始化优化器（模拟训练场景）
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 模拟多步训练（验证梯度是否稳定计算）
        for step in range(num_steps):
            # 重置梯度
            optimizer.zero_grad()
            
            # 前向计算（开启梯度）
            output = model(x)
            
            # 构造虚拟损失（模拟分类/回归任务的损失）
            # 这里用 "输出与随机目标的MSE损失" 作为示例
            target = torch.randn_like(output, device=device)
            loss = F.mse_loss(output, target)
            
            # 反向传播（计算梯度）
            loss.backward()
            
            # 验证关键参数是否有梯度（排除禁用权重衰减的参数）
            grad_check_params = [
                ("in_proj.weight", model.in_proj.weight),
                ("x_proj.weight", model.x_proj.weight),
                ("dt_proj.weight", model.dt_proj.weight),
                ("out_proj.weight", model.out_proj.weight)
            ]
            has_valid_grad = True
            for param_name, param in grad_check_params:
                if param.grad is None:
                    has_valid_grad = False
                    print(f"警告: {param_name} 无梯度！")
                else:
                    # 验证梯度是否为有效数值（非NaN/Inf）
                    assert not torch.isnan(param.grad).any(), f"{param_name} 梯度包含NaN！"
                    assert not torch.isinf(param.grad).any(), f"{param_name} 梯度包含Inf！"
            
            # 参数更新
            optimizer.step()
            
            print(f"第 {step+1}/{num_steps} 步训练:")
            print(f"  损失值: {loss.item():.6f}")
            print(f"  关键参数梯度有效性: {'正常' if has_valid_grad else '异常'}")

        print(f"\n反向传播测试成功！{num_steps} 步训练均正常完成，梯度计算与参数更新无异常。")

    except Exception as e:
        print(f"反向传播失败！错误信息: {str(e)}")
        # 打印详细的错误堆栈（便于定位问题）
        import traceback
        traceback.print_exc()
        return


    # -------------------------- 5. 最终结论 --------------------------
    print(f"\n=== 测试总结 ===")
    print(f"✅ MambaVisionMixer 前向传播正常（输出形状/设备匹配）")
    print(f"✅ MambaVisionMixer 反向传播正常（梯度计算/参数更新无异常）")
    print(f"✅ 模型在 {device} 设备上可正常运行")


if __name__ == "__main__":
    main()