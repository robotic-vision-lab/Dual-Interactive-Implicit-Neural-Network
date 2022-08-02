from src.models.components import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.rdn import make_rdn

class IMSISR(nn.Module):
    def __init__(self,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        self.encoder = make_rdn()
        self.tail = ImplicitDecoder(64, 256, 4)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res, scale)
        x = self.add_mean(x)

        return x 

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=128, depth=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.synthesis = nn.ModuleList()
        self.modulate = nn.ModuleList()
        self.last_block = nn.ModuleList()
        self.modulate.append(nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1),
                            nn.ReLU()))
        self.synthesis.append(nn.Sequential(nn.Conv2d(2, hidden_channels, 1),
                            SineAct()))
        for i in range(depth-1):
            self.synthesis.append(nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 1),
                                                SineAct()))
            self.modulate.append(nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 1),
                                                nn.ReLU()))

        self.last_conv = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1)

    def _make_pos_encoding(self, x, scale_factor):
        B, C, H, W = x.shape
        #H_up, W_up = size
        H_up = int(H * scale_factor)
        W_up = int(W * scale_factor)

        h_idx = torch.arange(H, device=x.device)/H * 2 - 1 + 1/H
        w_idx = torch.arange(W, device=x.device)/W * 2 - 1 + 1/W
        h_grid, w_grid = torch.meshgrid(h_idx, w_idx)

        h_idx_up = torch.arange(H_up, device=x.device)/H_up * 2 - 1 + 1/H_up
        w_idx_up = torch.arange(W_up, device=x.device)/W_up * 2 - 1 + 1/W_up
        h_up_grid, w_up_grid = torch.meshgrid(h_idx_up, w_idx_up)

        h_relative_grid = (h_up_grid - F.interpolate(h_grid.unsqueeze(0).unsqueeze(0), size=(H_up, W_up), mode='nearest'))*H
        w_relative_grid = (w_up_grid - F.interpolate(w_grid.unsqueeze(0).unsqueeze(0), size=(H_up, W_up), mode='nearest'))*W
        
        grid = torch.cat((h_relative_grid, w_relative_grid), dim=1).expand(B,-1,-1,-1) #(B, 2, H_up, W_up)
        
        return grid.contiguous().detach()

    def forward(self, x, scale_factor):
        #input x(B,F,H,W) image representations
        p = self._make_pos_encoding(x, scale_factor) #2
        #x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False) #64
        h = F.interpolate(self.modulate[0](x), scale_factor=scale_factor, mode='nearest', align_corners=False) #128
        p = self.synthesis[0](p)
        for i in range(1, len(self.synthesis)):
            h = self.modulate[i](h) #128
            p = h*self.synthesis[i](p) #128
        p = self.last_conv(p)
        return p