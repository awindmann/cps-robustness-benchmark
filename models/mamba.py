# Mamba model.
# 
# Original paper: https://openreview.net/forum?id=tEYskw1VY2
# Based on the original code (https://github.com/state-spaces/mamba/tree/main) as well as the repository "Time-Series-Library" (https://github.com/thuml/Time-Series-Library) under the MIT License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from models.utils.Embed import DataEmbedding
from models.base_module import BaseLitModule


class Mamba(BaseLitModule):
    """
    Simplified Mamba model.
    Lacks the GPU memory optimization.
    Args:
        d_model: int, dimension of the model. If set, uses a data embedding, else set to d_features of the data.
        expand: int, expansion factor for inner layer
        d_conv: int, kernel size for depthwise convolution
        d_state: int, state dimension
        n_mamba_blocks: int, number of mamba blocks
        use_rms_B: bool, whether to use RMS normalization for B
        use_rms_C: bool, whether to use RMS normalization for C
        use_rms_delta: bool, whether to use RMS normalization for delta
        dropout_embed: float, dropout of data embedding, only relevant if d_model is set
        loss: str, loss function
    """
    def __init__(self, d_model=None, expand=2, d_conv=4, d_state=16, n_mamba_blocks=2, use_rms_B=False, use_rms_C=False, use_rms_delta=False,
                 dropout_embed=0.0, loss="MSE", 
                 **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "Mamba"

        self.d_model = d_model if d_model is not None else self.d_features
        self.d_inner = self.d_model * expand
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_conv = d_conv
        self.d_state = d_state
        
        # data embedding
        self.embedding = DataEmbedding(self.d_features, self.d_model, dropout=dropout_embed) if d_model is not None else None
        # residual blocks for encoding
        self.layers = nn.ModuleList(
            [MambaBlock(self.d_model, self.d_inner, self.dt_rank, self.d_conv, self.d_state, use_rms_B, use_rms_C, use_rms_delta)
             for _ in range(n_mamba_blocks)]
        )

        # normalization layer
        self.norm = RMSNorm(self.d_model)

        # out projection if necessary
        self.out_proj = nn.Linear(self.d_model, self.d_features, bias=False) if d_model is not None else None

        # loss function
        self.loss_fn = nn.MSELoss() if loss == "MSE" else None

    def _shared_step(self, x, y):
        x = self.embedding(x, None) if self.embedding is not None else x

        for layer in self.layers:
            x = layer(self.norm(x)) + x  # standard residual block. original code differs here only to fuse add + norm
        
        x = self.norm(x)

        x = self.out_proj(x) if self.out_proj is not None else x

        y_pred = x[:, -self.d_seq_out:, :]
        if not y_pred.is_contiguous():
            y_pred = y_pred.contiguous()

        loss = self.loss_fn(y_pred, y) if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv, d_state, use_rms_B, use_rms_C, use_rms_delta):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.d_state = d_state

        # input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner
        )

        # projections for delta, B, and C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # initialize dt_proj
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # initialize delta bias
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))

        # initialize A and D parameters
        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # rms normalization (optional)
        self.use_rms_B = use_rms_B
        self.use_rms_C = use_rms_C
        self.use_rms_delta = use_rms_delta

    def forward(self, x):
        """
        mamba block with ssm and residual logic
        """
        b, l, d = x.shape

        # input projection and split
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # depthwise convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        # activation
        x = F.silu(x)

        # apply state-space model
        y = self.ssm(x)

        # combine with residual
        y = y * F.silu(res)

        # output projection
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """
        state-space model computation
        """
        d_in, n = self.A_log.shape

        # initialize A and D
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # projections for delta, B, and C
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)

        # optional normalization for delta, B, and C
        if self.use_rms_delta:
            delta = rms_norm_forward(delta, None, None, eps=1e-6)
        delta = F.softplus(self.dt_proj(delta))

        if self.use_rms_B:
            B = rms_norm_forward(B, None, None, eps=1e-6)
        if self.use_rms_C:
            C = rms_norm_forward(C, None, None, eps=1e-6)

        # selective scan operation
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        simplified selective scan implementation, lacks efficiency
        """
        b, l, d_in = u.shape
        n = A.shape[1]

        # precompute deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        # iterative scan
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        # stack outputs and add skip connection
        y = torch.stack(ys, dim=1)
        y = y + u * D

        return y


def rms_norm_forward(x, weight, bias, eps=1e-6):
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        norm = norm * weight
    if bias is not None:
        norm = norm + bias
    return norm


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return rms_norm_forward(x, self.weight, None, eps=self.eps)