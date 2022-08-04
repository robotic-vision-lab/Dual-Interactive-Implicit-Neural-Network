from src.models.components import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.rdn import make_rdn
import pdb
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
        self.decoder = ImplicitDecoder()

    def forward(self, x, size, bsize=None):
        x = self.encoder(x)
        x = self.decoder(x, size, bsize)
        return x 

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[256, 256, 256, 256]):
        super().__init__()
        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()

        last_dim_K = in_channels
        last_dim_Q = 3
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                        nn.ReLU()))
            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_K = hidden_dim
            last_dim_Q = hidden_dim
        self.last_layer = nn.Conv2d(hidden_dims[-1], 3, 3, padding=1)

    def _make_pos_encoding(self, x, size):
        B, C, H, W = x.shape
        H_up, W_up = size
       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx, indexing='ij'), dim=0)

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up, indexing='ij'), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest-exact')) #important! mode='nearest' gives inconsistent results
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W

        return rel_grid.contiguous().detach()

    def step(self, x, syn_inp):
        k = self.K[0](x)
        q = k*self.Q[0](syn_inp)
        #out = q * k
        
        for i in range(1, len(self.K)):
            k = self.K[i](k)
            q = k*self.Q[i](q)
            #out = k * q
        q = self.last_layer(q)
        return q

    def batched_step(self, x, syn_inp, bsize):
        with torch.no_grad():
            n = syn_inp.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.step(x[:, ql: qr, :], syn_inp[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    def reshape_pred(self, pred, size):
        shape = [pred.shape[0], *size, 3]
        pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        return pred

    def forward(self, x, size, bsize=None):
        B, C, H_in, W_in = x.shape
        rel_coord = self._make_pos_encoding(x, size).expand(B, -1, *size) #2
        ratio = x.new_tensor([(H_in*W_in)/(size[0]*size[1])]).view(1, -1, 1, 1).expand(B, -1, *size) #2
        syn_inp = torch.cat([rel_coord, ratio], dim=1) #4
        #x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=size, mode='nearest-exact').view(B, size[0]*size[1], -1)
        x = F.interpolate(x, size=size, mode='nearest-exact')
        if bsize is None:
            pred = self.step(x, syn_inp)
        else:
            pred = self.batched_step(x, syn_inp, bsize)
        #return self.reshape_pred(pred, size)
        return pred