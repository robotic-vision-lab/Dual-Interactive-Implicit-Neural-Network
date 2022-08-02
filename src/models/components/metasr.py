
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.rdn import make_rdn
from src.models.components.mlp import MLP
import pdb


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
        return torch.stack(torch.meshgrid(h_idx, w_idx, indexing='ij'), dim=-1).clamp_(-1 + 1e-6, 1 - 1e-6) 
    
class MetaSR(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = make_rdn()
        imnet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 3,
                'out_dim': self.encoder.out_dim * 9 * 3,
                'hidden_list': [256]
            }
        }
        self.imnet = MLP(3, self.encoder.out_dim * 9 * 3, [256])

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

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
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_coord = self.make_coord(feat.shape[-2:], device=feat.device, flatten=False)
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 3)
        return pred

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
