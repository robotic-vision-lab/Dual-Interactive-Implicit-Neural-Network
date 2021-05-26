import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__()
        self.fc = nn.Conv2d(in_channels, out_channels, 1)
        self.attn_fc = nn.Conv2d(2 * out_channels, 1, 1)

    def forward(self, x):
        #x(N,C,H,num_neighbors+1)
        z = self.fc(x)
        z2 = torch.cat((z, z[:,:,:,[0]].expand(-1,-1,-1,z.shape[-1])), dim=1) #(N,C*2,H,num_neighbors+1)
        alpha = F.leaky_relu(self.attn_fc(z2)) #(N,1,H,num_neighbors+1)
        alpha = F.softmax(alpha, dim=-1).expand(-1, z.shape[1], -1, -1) #(N,C*2,H,num_neighbors+1)
        out = torch.sum(alpha * z, dim=-1, keepdim=True) #(N,C,H,1)
        return out

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_channels, out_channels))
        self.merge = merge
    
    def forward(self, x):
        #x(N,C,H,num_neighbors+1)
        head_outs = [attn_head(x) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1) #(N,C*K,H,1)
        elif self.merge == 'avg':
            return torch.mean(torch.stack(head_outs), dim=0, keepdim=False)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(hidden_channels, out_channels, 3, padding=1))
    
    def forward(self, x):
        x = self.conv(x)
        return x


class UDown(nn.Module):
    def __init__(self, in_channels):
        super(UDown, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, in_channels, 2, 2),
                                        nn.ReLU())

    def forward(self, x):
        x = self.downsample(x)
        return x

class UUp(nn.Module):
    def __init__(self):
        super(UUp, self).__init__()
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.upsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(UNet, self).__init__()
        self.convblock1 = ConvBlock(in_channels, 64, 64)
        self.down1 = UDown(64)
        self.convblock2 = ConvBlock(64, 128, 128)
        self.down2 = UDown(128)
        self.convblock3 = ConvBlock(128, 256, 256)
        self.down3 = UDown(256)
        self.convblock4 = ConvBlock(256, 512, 1024)
        self.up = UUp()
        self.convblock5 = ConvBlock(512,512,512)
        self.convblock6 = ConvBlock(256,256,256)
        self.convblock7 = ConvBlock(128,128,128)
        
        num_heads = 4
        self.gat1 = MultiHeadGATLayer(128, 256, num_heads, merge='cat')
        self.mlp = nn.Sequential(nn.Conv2d(256*num_heads, 1024, 1),
                                nn.ReLU(),
                                nn.Conv2d(1024, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 3, 1),
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

        d = 0.003
        self.displacement = torch.Tensor([[0, 0], [-d, -d], [d, d], [d, -d], [-d, d]]).cuda()


    def forward(self, x, p):
        d = 1.0/x.shape[2]
        self.displacement = torch.Tensor([[0, 0], [-d, -d], [d, d], [d, -d], [-d, d]]).cuda()
        p = torch.cat([p + d for d in self.displacement], dim=2) #(N,num_points,5,2)
        x1 = self.convblock1(x)                     #64
        x2 = self.convblock2(self.down1(F.relu(x1))) #128
        x3 = self.convblock3(self.down2(F.relu(x2))) #256
        x4 = self.convblock4(self.down3(F.relu(x3))) #1024

        x3 = torch.cat((x3, self.up(x4)), dim=1)    #256+256=512
        x3 = F.relu(self.convblock5(x3))
        x2 = torch.cat((x2, self.up(x3)), dim=1)    #128+128=256
        x2 = F.relu(self.convblock6(x2))
        x1 = torch.cat((x1, self.up(x2)), dim=1)    #64+64=128
        x1 = F.relu(self.convblock7(x1))
        features = F.grid_sample(x1, p, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features = F.relu(self.gat1(features)) #(N, C, num_points, 1)
        out = self.mlp(features)
        return out


class Mark_4(nn.Module):
    def __init__(self, in_channels=3):
        super(Mark_4, self).__init__()
        self.convblock1 = ConvBlock(in_channels, 64, 64)
        self.down1 = UDown(64)
        self.convblock2 = ConvBlock(64, 128, 128)
        self.down2 = UDown(128)
        self.convblock3 = ConvBlock(128, 256, 256)
        self.down3 = UDown(256)
        self.convblock4 = ConvBlock(256, 512, 1024)
        self.up = UUp()
        self.convblock5 = ConvBlock(512,512,512)
        self.convblock6 = ConvBlock(256,256,256)
        self.convblock7 = ConvBlock(128,128,128)
        
        num_heads = 4
        self.gat1 = MultiHeadGATLayer(128, 256, num_heads, merge='avg')
        self.gat2 = MultiHeadGATLayer(256, 256, num_heads, merge='avg')
        self.gat3 = MultiHeadGATLayer(512, 512, num_heads, merge='avg')
        self.mlp = nn.Sequential(nn.Conv2d(256+256+512, 1024, 1),
                                nn.ReLU(),
                                nn.Conv2d(1024, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 3, 1),
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

        d = 0.003
        self.displacement = torch.Tensor([[0, 0], [-d, -d], [d, d], [d, -d], [-d, d]]).cuda()


    def forward(self, x, p):
        x1 = self.convblock1(x)                     #64
        x2 = self.convblock2(self.down1(F.relu(x1))) #128
        x3 = self.convblock3(self.down2(F.relu(x2))) #256
        x4 = self.convblock4(self.down3(F.relu(x3))) #1024

        x3 = torch.cat((x3, self.up(x4)), dim=1)    #256+256=512
        x3 = F.relu(self.convblock5(x3))
        x2 = torch.cat((x2, self.up(x3)), dim=1)    #128+128=256
        x2 = F.relu(self.convblock6(x2))
        x1 = torch.cat((x1, self.up(x2)), dim=1)    #64+64=128
        x1 = F.relu(self.convblock7(x1))
        
        p1 = self.add_neighbors(p, 1.0/x1.shape[1], 1.0/x1.shape[2])
        features_1 = F.grid_sample(x1, p1, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features_1 = F.relu(self.gat1(features_1)) #(N, C, num_points, 1)

        p2 = self.add_neighbors(p, 1.0/x2.shape[1], 1.0/x2.shape[2])
        features_2 = F.grid_sample(x2, p2, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features_2 = F.relu(self.gat2(features_2)) #(N, C, num_points, 1)
        
        p3 = self.add_neighbors(p, 1.0/x3.shape[1], 1.0/x3.shape[2])
        features_3 = F.grid_sample(x3, p3, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features_3 = F.relu(self.gat3(features_3)) #(N, C, num_points, 1)
        
        features = torch.cat((features_1, features_2, features_3), dim=1)
        out = self.mlp(features)
        return out

    def add_neighbors(self, p, d1, d2):
        displacement = torch.Tensor([[0, 0], [-d1, -d2], [d1, d2], [d1, -d2], [-d1, d2]])
        if p.is_cuda: displacement = displacement.cuda()
        return torch.cat([p + d for d in self.displacement], dim=2) #(N,num_points,5,2)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, act=nn.ReLU(True), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=kernel_size//2))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Encoder(nn.Module):
    def __init__(self, in_channels, n_resblocks, n_feats):
        super(Encoder, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        m_body = [ResBlock(n_feats, 3) for _ in range(n_resblocks)]
        m_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*m_body)
        
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x
    

class TinyEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(TinyEncoder, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1),
                                nn.ReLU())

        self.body = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 512, 3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 3, padding=1),
                                nn.ReLU())
        
    def forward(self, x):
        x = self.head(x)
        o = self.body(x)
        o = torch.cat((x,o), dim=1)
        return o

class TinyDecoder(nn.Module):
    def __init__(self, n_feats, num_heads):
        super(TinyDecoder, self).__init__()
        self.gat = MultiHeadGATLayer(n_feats, n_feats, num_heads, merge='avg')
        self.mlp = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1),
                                nn.ReLU(),
                                nn.Conv2d(n_feats, n_feats, 1),
                                nn.ReLU(),
                                nn.Conv2d(n_feats, n_feats, 1),
                                nn.ReLU(),
                                nn.Conv2d(n_feats, 3, 1),
                                nn.Sigmoid()) #if want RGB out change last output channel to 3
    
    def forward(self, x, p):
        pn = self.add_neighbors(p, 1.0/x.shape[1], 1.0/x.shape[2])
        features = F.grid_sample(x, pn, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features = F.relu(self.gat(features)) #(N, C, num_points, 1)
        out = self.mlp(features)
        return out

    def add_neighbors(self, p, d1, d2):
        displacement = torch.Tensor([[0, 0], [-d1, -d2], [d1, d2], [d1, -d2], [-d1, d2],
                                            [0, -d2], [0, d2], [d1, 0], [-d1, 0]])
        if p.is_cuda: displacement = displacement.cuda()
        return torch.cat([p + d for d in displacement], dim=2) #(N,num_points,5,2)


class Decoder(nn.Module):
    def __init__(self, n_feats, num_heads):
        super(Decoder, self).__init__()
        self.gat = MultiHeadGATLayer(n_feats, n_feats, num_heads, merge='avg')
        self.mlp = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1),
                                nn.ReLU(),
                                nn.Conv2d(n_feats, n_feats, 1),
                                nn.ReLU(),
                                nn.Conv2d(n_feats, n_feats, 1),
                                nn.ReLU(),
                                nn.Conv2d(n_feats, 3, 1),
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        pn = self.add_neighbors(p, 1.5/x.shape[1], 1.5/x.shape[2])
        features = F.grid_sample(x, pn, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features = F.relu(self.gat(features)) #(N, C, num_points, 1)
        out = self.mlp(features)
        return out

    def add_neighbors(self, p, d1, d2):
        displacement = torch.Tensor([[0, 0], [-d1, -d2], [d1, d2], [d1, -d2], [-d1, d2],
                                            [0, -d2], [0, d2], [d1, 0], [-d1, 0]])
        if p.is_cuda: displacement = displacement.cuda()
        return torch.cat([p + d for d in displacement], dim=2) #(N,num_points,5,2)

class Mark_2(nn.Module):
    def __init__(self, in_channels=3, n_resblocks=5, n_feats=256, num_heads=16):
        super(Mark_2, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, n_resblocks=n_resblocks, n_feats=n_feats)
        self.decoder = Decoder(n_feats=n_feats, num_heads=num_heads)

    def forward(self, x, p):
        x = self.encoder(x)
        out = self.decoder(x, p)
        return out

class Mark_3(nn.Module):
    def __init__(self, in_channels=3, num_heads=8):
        super(Mark_3, self).__init__()
        self.encoder = TinyEncoder(in_channels=in_channels)
        self.decoder = TinyDecoder(n_feats=512+64, num_heads=num_heads)

    def forward(self, x, p):
        x = self.encoder(x)
        out = self.decoder(x, p)
        return out

if __name__ == '__main__':
    net = Mark_2()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))