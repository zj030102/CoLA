
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os

CHUNK_LEN = 80
Channel = 256
here = os.path.dirname(os.path.realpath(__file__))
flags = ['-res-usage', f'-D_C_={Channel}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
# load(name="rwkv7_wind_fp32_hs32", sources=[f'{here}/rwkv7_fp32_hs32.cu', f'{here}/rwkv7_fp32_hs32.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
# 需要时在启用，不然和puredelta冲突
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

class DeltaNet(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        conv_size: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.conv_size = conv_size

        self.head_dim = self.d_model // num_heads

        assert self.d_model % num_heads == 0, f"d_model must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=conv_size,
            groups=self.d_model,
            bias=True,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=conv_size,
            groups=self.d_model,
            bias=True,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=conv_size,
            groups=self.d_model,
            bias=True,
        )
        nn.init.xavier_uniform_(self.q_conv1d.weight, gain=1e-2)
        nn.init.xavier_uniform_(self.k_conv1d.weight, gain=1e-2)
        nn.init.xavier_uniform_(self.v_conv1d.weight, gain=1e-2)
        nn.init.zeros_(self.q_conv1d.bias)
        nn.init.zeros_(self.k_conv1d.bias)
        nn.init.zeros_(self.v_conv1d.bias)

        self.beta_proj = nn.Linear(d_model, d_model, bias=False) # use vector b (instead of scalar b) for each head
        
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(self.d_model, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, -1).permute(0, 2, 1)          # [B, H*W, C]

        hidden_q = self.q_proj(hidden_states)
        hidden_k = self.k_proj(hidden_states)
        hidden_v = self.v_proj(hidden_states)
        hidden_q = hidden_q.transpose(1, 2)  
        hidden_k = hidden_k.transpose(1, 2)  
        hidden_v = hidden_v.transpose(1, 2)  
        conv_q = F.conv1d(input=hidden_q, weight=self.q_conv1d.weight, bias=self.q_conv1d.bias, padding='same', groups=self.d_model)
        conv_k = F.conv1d(input=hidden_k, weight=self.k_conv1d.weight, bias=self.k_conv1d.bias, padding='same', groups=self.d_model)
        conv_v = F.conv1d(input=hidden_v, weight=self.v_conv1d.weight, bias=self.v_conv1d.bias, padding='same', groups=self.d_model)
        q = F.silu(conv_q) * conv_q + hidden_q
        k = F.silu(conv_k) * conv_k + hidden_k
        v = F.silu(conv_v) * conv_v + hidden_v
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  

        q, k, v= map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), (q, k, v))

        beta = self.beta_proj(hidden_states).sigmoid()

        B,T,H,D = q.shape
        beta = beta.view(B,T,H,D)
        q = F.normalize(q.view(B,T,H,D), dim=-1, p=2.0) # align with delta_rule kernel
        k = F.normalize(k.view(B,T,H,D), dim=-1, p=2.0) # align with delta_rule kernel
        o = RUN_RWKV7_FP32(r=q*(D**(-0.5)), w=q*0-999, k=k, v=beta*v, a=-beta*k, b=k) # align with delta_rule kernel (set w to a very negative number to disable it)

        o = self.o_norm(o.float())
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        o = o.permute(0, 2, 1).view(b, c, h, w)

        return o
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x * self.weight
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W, D = 6, 88, 100, 256
    x = torch.randn(B, D, H, W, device=device)


    model = DeltaNet(d_model=D, num_heads=1, conv_size=4).to(device)
    model.eval()  # 测试模式，关闭 dropout 等随机性

    with torch.no_grad():
        out = model(x)

    print('Input shape :', x.shape)
    print('Output shape:', out.shape)
    print('Expected    : torch.Size([6, 8800, 256])')

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f'Peak GPU mem: {mem:.2f} GB')

if __name__ == '__main__':
    main()