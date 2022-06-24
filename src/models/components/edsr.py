from src.models.components import common

import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, kernel_size=3, scale=4, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3 
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(1)
        self.add_mean = common.MeanShift(1, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1.
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, _):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 
