
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os

# 修改超参数重新编译时需要删除原有~/.cache/torch_extensions/py38_cu118/rwkv7_wind_fp32_hs32/

CHUNK_LEN = 80
Channel = 64
here = os.path.dirname(os.path.realpath(__file__))
flags = ['-res-usage', f'-D_C_={Channel}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="rwkv7_wind_fp32_hs32", sources=[f'{here}/rwkv7_fp32_hs32.cu', f'{here}/rwkv7_fp32_hs32.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class rwkv7_wind_fp32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,r,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.float32 for i in [w,r,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,r,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.rwkv7_wind_fp32 = torch.ops.rwkv7_wind_fp32_hs32  # 绑定算子，backward时调用rwkv7_wind_fp32
        torch.ops.rwkv7_wind_fp32_hs32.forward(w,r,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,r,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.float32 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,r,k,v,z,b,s,sa = ctx.saved_tensors
        B,T,H,C = w.shape 
        dw,dr,dk,dv,dz,db = [torch.empty_like(x) for x in [w,r,k,v,z,b]]
        torch.ops.rwkv7_wind_fp32.backward(w,r,k,v,z,b, dy,s,sa, dw,dr,dk,dv,dz,db)
        return dw,dr,dk,dv,dz,db
    
def RUN_RWKV7_FP32(r,w,k,v,a,b):
    B,T,H,C = r.shape
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    r, w, k, v, a, b = [i.contiguous() for i in (r, w, k, v, a, b)]
    return rwkv7_wind_fp32.apply(w,r,k,v,a,b).view(B,T,H,C)

class PureDeltaNet(nn.Module):

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
        
        self.o_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, -1).permute(0, 2, 1)          # [B, H*W, C]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k, v= map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), (q, k, v))

        beta = self.beta_proj(hidden_states).sigmoid()

        B,T,H,D = q.shape

        beta = beta.view(B,T,H,D)
        q = F.normalize(q.view(B,T,H,D), dim=-1, p=2.0) # align with delta_rule kernel
        k = F.normalize(k.view(B,T,H,D), dim=-1, p=2.0) # align with delta_rule kernel
        o = RUN_RWKV7_FP32(r=q*(D**(-0.5)), w=q*0-999, k=k, v=beta*v, a=-beta*k, b=k) # align with delta_rule kernel (set w to a very negative number to disable it)

        o = o.float()
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        o = o.permute(0, 2, 1).view(b, c, h, w)

        return o
    
def inspect_unit(unit: torch.Tensor):
    """
    计算 unit = exp(M - importance_sq) 并打印详细统计量
    返回: unit 张量（与 M 同设备、同 dtype）
    """
    with torch.no_grad():
        # ---- 基本统计 ----
        print("==========  unit = exp(M - importance_sq) 统计  ==========")
        print(f"shape        : {unit.shape}")
        print(f"dtype        : {unit.dtype}")
        print(f"device       : {unit.device}")
        print(f"min          : {unit.min().item():.6e}")
        print(f"max          : {unit.max().item():.6e}")
        print(f"mean         : {unit.mean().item():.6e}")
        print(f"std          : {unit.std().item():.6e}")
        print(f"median       : {unit.median().item():.6e}")

        # ---- 极端值检查 ----
        inf_mask = torch.isinf(unit)
        nan_mask = torch.isnan(unit)
        zero_mask = unit == 0
        pos_mask = unit > 0
        print(f"inf  数量    : {inf_mask.sum().item()}")
        print(f"nan  数量    : {nan_mask.sum().item()}")
        print(f"zero 数量    : {zero_mask.sum().item()}")
        print(f"正数 数量    : {pos_mask.sum().item()}")

        # ---- 数值分布 ----
        # 把 unit 拉平后按数量级分段
        flat = unit.view(-1)
        logflat = torch.log10(flat.clamp(min=1e-300))   # 避免 log(0)
        hist = torch.histc(logflat, bins=20, min=-300, max=10)
        bins = torch.linspace(-300, 10, 21)
        print("对数数量级分布 (log10(unit)):")
        for i in range(20):
            print(f"[{bins[i]:5.0f}, {bins[i+1]:5.0f})  {int(hist[i]):8d}")

        print("===========================================================")
