import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.rdn import make_rdn
from src.models.components.mlp import MLP
import pdb


class LIIF(nn.Module):

    def __init__(self, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = make_rdn()

        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(imnet_in_dim, 3, [256, 256, 256, 256])


    def gen_feat(self, inp):
        return self.encoder(inp)
            
    
    def make_coord(self, size, device, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(size):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def make_coord_and_cell(self, inp, size, ranges=None, flatten=True):
        """ Convert the image to coord-RGB pairs.
            img: Tensor, (3, H, W)
        """
        hr_coord = self.make_coord(size, inp.device)
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / size[-2]
        cell[:, 1] *= 2 / size[-1]
        return hr_coord.unsqueeze(0).expand(inp.shape[0], -1, -1).contiguous(), cell.unsqueeze(0).expand(inp.shape[0], -1, -1).contiguous()
        
    def query_rgb(self, feat, coord, cell=None):

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.make_coord(feat.shape[-2:], device=feat.device, flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def batched_predict(self, feat, coord, cell, bsize):
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.query_rgb(feat, coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred

    def reshape_pred(self, pred, size):
        shape = [pred.shape[0], *size, 3]
        pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        return pred

    def forward(self, inp, size, bsize=None):
        coord, cell = self.make_coord_and_cell(inp, size)
        feat = self.gen_feat(inp)
        if bsize is not None:
            pred = self.batched_predict(feat, coord, cell, bsize)
        else:
            pred = self.query_rgb(feat, coord, cell)
        return self.reshape_pred(pred, size)
 
 
 
 
 
'''
def make_grid(size, offset=[0, 0], device='cuda'):
    H, W = size
    if offset == [0, 0]:
        h_idx = - 1 + 1/H + 2*(1/H)*torch.arange(H, device=device) 
        w_idx = - 1 + 1/W + 2*(1/W)*torch.arange(W, device=device)
        return torch.stack(torch.meshgrid(h_idx, w_idx, indexing='ij'), dim=-1) 
    else:
        h_offset = offset[0]/H + offset[0]*1e-6
        w_offset = offset[1]/W + offset[1]*1e-6
        h_idx = h_offset - 1 + 1/H + 2*(1/H)*torch.arange(H, device=device) 
        w_idx = w_offset - 1 + 1/W + 2*(1/W)*torch.arange(W, device=device)
        return torch.stack(torch.meshgrid(h_idx, w_idx, indexing='ij'), dim=-1).clamp_(-1 + 1e-6, 1 - 1e-6).contiguous().detach() 

class LIIF(nn.Module):
    def __init__(self,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 mlp_dims = [256, 256, 256, 256],
                 out_dim=3):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        self.encoder = make_rdn()
        
        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        layers = []
        lastv = imnet_in_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(lastv, dim))
            layers.append(nn.ReLU())
            lastv = dim
        layers.append(nn.Linear(lastv, out_dim))
        self.decoder = nn.Sequential(*layers)

    def step(self, x, split):
        _, _, _, W  = x.shape
        if split > 1:
            out = []
            for i in range(0, W, W//split):
                out.append(self.decoder(x[:,:,:,i:i+W//split].permute(0,2,3,1)).permute(0,3,1,2))
            return torch.cat(out, dim=-1)
        else:
            return self.decoder(x.permute(0,2,3,1)).permute(0,3,1,2)

    def decode(self, feat, size, split):
        #feat = self.encoder(x)

        B, C, H_in, W_in = feat.size()

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(B, C*9, H_in, W_in)
        
        if self.local_ensemble:
            h_offsets = [-1, 1]
            w_offsets = [-1, 1]
        else:
            h_offsets = [0]
            w_offsets = [0]
        
        preds = {}
        areas = {}

        feat_global_grid = make_grid(size=feat.shape[-2:], device=feat.device).permute(2,0,1).unsqueeze(0).expand(B, -1, -1, -1) #(B,2,H_in,W_in)
        out_global_grid = make_grid(size=size, device=feat.device).permute(2,0,1).unsqueeze(0).expand(B, -1, -1, -1) #(B,2,H_out,W_out)
        
        for h_offset in h_offsets:
            for w_offset in w_offsets:
                grid_offset = make_grid(size=size, offset=[h_offset, w_offset], device=feat.device).unsqueeze(0).expand(B, -1, -1, -1) #(B,H_out,W_out,2)
                
                #get nearest feature with respect to offsets
                feat_o = F.grid_sample(feat, grid_offset.flip(-1), mode='nearest', align_corners=False) #(B,C*9,H_out,W_out)
                
                #get relative coords w.r.t grid_offset
                rel_coords = out_global_grid - F.grid_sample(feat_global_grid, grid_offset.flip(-1), mode='nearest', align_corners=False) #(B,2,H_out,W_out)
                rel_coords[:, 0, :, :] *= H_in
                rel_coords[:, 1, :, :] *= W_in
                inp = torch.cat([feat_o, rel_coords], dim=1)

                if self.cell_decode:
                    cell_h = feat.new_tensor(2*H_in/size[0])
                    cell_w = feat.new_tensor(2*W_in/size[1])
                    cell = torch.stack([cell_h, cell_w]).view(1,2,1,1).expand(B, -1, *size) #(B,2,H_out,W_out)
                    inp = torch.cat([inp, cell], dim=1)
                
                preds[(h_offset, w_offset)] = self.step(inp, split)
                
                area = torch.abs(rel_coords[:,[0],:,:] * rel_coords[:,[1],:,:])
                areas[(h_offset, w_offset)] = area + 1e-6
                
        total_area = torch.cat(list(areas.values()), dim=1).sum(dim=1, keepdim=True) #(B,1,H_out,W_out)

        ret = 0
        for h_offset in h_offsets:
            for w_offset in w_offsets:
                ret += preds[(h_offset, w_offset)] * (areas[(-h_offset, -w_offset)] / total_area)

        return ret

    def forward(self, x, size, split=0):
        x = self.encoder(x)
        x = self.decode(x, size, split)
        return x


if __name__ == '__main__':
    net = LIIF()
    out = net(torch.rand(1,16,4,6), [6,10])

''' 