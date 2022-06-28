from src.models.components import common
import torch
import torch.nn as nn
import torch.nn.functional as F

class IMSISR(nn.Module):
    def __init__(self, in_channels, n_resblocks, n_feats, kernel_size, res_scale=1., rgb_range=255., conv=common.default_conv):
        super(IMSISR, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3 
        act = nn.ReLU(True)
        
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = ImplicitDecoder(n_feats, n_feats, n_resblocks//4)

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
        d_model = self.hidden_channels
        B, C, H, W = x.shape
        #H_up, W_up = size
        H_up = int(H * scale_factor)
        W_up = int(W * scale_factor)

        #h_idx = torch.arange(-1 + 1/H, 1 + 1/H, 2/H, device=x.device)
        #w_idx = torch.arange(-1 + 1/W, 1 + 1/W, 2/W, device=x.device)
        h_idx = torch.arange(H, device=x.device)/H * 2 - 1 + 1/H
        w_idx = torch.arange(W, device=x.device)/W * 2 - 1 + 1/W
        h_grid, w_grid = torch.meshgrid(h_idx, w_idx)

        #h_idx_up = torch.arange(-1 + 1/H_up, 1 + 0.5/H_up, 2/H_up, device=x.device)
        #w_idx_up = torch.arange(-1 + 1/W_up, 1 + 0.5/W_up, 2/W_up, device=x.device)
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
        h = F.interpolate(self.modulate[0](x), scale_factor=scale_factor, mode='bilinear', align_corners=False) #128
        p = self.synthesis[0](p)
        for i in range(1, len(self.synthesis)):
            h = self.modulate[i](h) #128
            p = h*self.synthesis[i](p) #128
        p = self.last_conv(p)
        return p