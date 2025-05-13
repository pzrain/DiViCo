import torch
import torch.nn as nn

import math
import einops as ein

    
def batch_index_select(x, idx):

    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AvgPool(nn.Module):
    def __init__(
        self,
        query_num: int = 64,
    ):
        super().__init__()
        self.query_num = query_num
        self.build_net()
    
    def build_net(self):
        hw = int(self.query_num ** 0.5)
        # sampler = nn.AdaptiveAvgPool2d((hw, hw))
        sampler = nn.AdaptiveMaxPool2d((hw, hw))
        self.sampler = sampler
        self.hw = hw
    
    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, h_dim = visual_feat.shape # 576
        hw = int(seq_len ** 0.5) # 24
        shaped_visual_feat = ein.rearrange(visual_feat, "b (h w) d -> b d h w", h=hw, w=hw) # torch.Size([64, 1024, 24, 24])
        
        pooled_visual_feat = self.sampler(shaped_visual_feat) # torch.Size([64, 1024, 12, 12])
        
        start_points = (torch.arange(self.hw, dtype=torch.float32) * (hw / self.hw)).long()
        column = start_points.repeat(self.hw, 1)
        row_offset = start_points.view(self.hw, 1) * hw
        position_ids = (column + row_offset).view(self.hw * self.hw)
        
        output_feat = ein.rearrange(pooled_visual_feat, "b d h w -> b (h w) d") # [64, 144, 1024]
        
        return output_feat, position_ids
