import torch
import torch.nn as nn
import torch.nn.functional as F

class Mark_1(nn.Module):
    def __init__(self, in_channels=3):
        super(Mark_1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 128, 3, padding=1, padding_mode='reflect'),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128*2, 3, padding=1, padding_mode='reflect'),
                                    nn.ReLU(),
                                    nn.Conv2d(128*2, 128*4, 3, padding=1, padding_mode='reflect'),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128*2, 3, padding=1, padding_mode='reflect'),
                                    nn.ReLU(),
                                    nn.Conv2d(128*2, 128*4, 3, padding=1, padding_mode='reflect'),
                                    nn.ReLU())
    
        self.downsample = torch.nn.MaxPool2d(2)
        
        num_features = (128 + 128 + 128) * 2

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 3, 1), 
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        #features_0 = F.grid_sample(x, p, mode='bicubic', align_corners=False) #features_0(N,C0,H',W')

        x = self.conv1(x)
        features_1 = F.grid_sample(x, p, mode='bilinear', align_corners=False) #features_1(N,C1,H',W')
        x_down = self.downsample(x)
        
        x = self.conv2(x)
        x = self.upsample(x)
        features_2 = F.grid_sample(x, p, mode='bilinear', align_corners=False) #features_2(N,C2,H',W')

        x = self.conv3(x)
        x = self.upsample(x)
        features_3 = F.grid_sample(x, p, mode='bilinear', align_corners=False) #features_3(N,C3,H',W')
        
        features_1_down = F.grid_sample(x_down, p, mode='bilinear', align_corners=False)
        x_down = self.downsample(self.conv2_down(x_down))
        features_2_down = F.grid_sample(x_down, p, mode='bilinear', align_corners=False)
        x_down = self.downsample(self.conv3_down(x_down))
        features_3_down = F.grid_sample(x_down, p, mode='bilinear', align_corners=False)
       # x = self.conv4(x)
        #x = self.upsample(x)
        #features_4 = F.grid_sample(x, p, mode='bicubic', align_corners=False) #features_4(N,C4,H',W')

        features = torch.cat(( features_1, features_2, features_3, features_1_down, features_2_down, features_3_down), dim=1)  #features(N,C0+C1+C2+C3,H',W')
        #features = torch.reshape(features, (features.shape[0], features.shape[2]*features.shape[3], -1)) #features(N,H'*W',C0+C1+C2+C3)

        out = self.mlp(features) #features(N,C,H',W')

        return out


class Mark_2(nn.Module):
    def __init__(self, in_channels=3):
        super(Mark_2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1),
                                    nn.ReLU())
                                  
        self.conv2 = nn.Sequential(nn.Conv2d(256 + 3, 256, 3, dilation=2, padding=2),
                                    nn.ReLU())
        
        self.conv3 = nn.Sequential(nn.Conv2d(256 + 256 + 3, 256, 3, dilation=3, padding=3),
                                    nn.ReLU())

        num_features = 256 + 256 + 256 + 3

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 3, 1), 
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        #features_0 = F.grid_sample(x, p, mode='bicubic', align_corners=False) #features_0(N,C0,H',W')
        x = torch.cat((self.conv1(x), x), dim=1) 
        x = torch.cat((self.conv2(x), x), dim=1) 
        x = torch.cat((self.conv3(x), x), dim=1) 

        features = F.grid_sample(x, p, mode='bilinear', align_corners=False)

        out = self.mlp(features)

        return out

class Mark_3(nn.Module):
    def __init__(self, in_channels=3):
        super(Mark_3, self).__init__()
        ''' 
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, 128, 3, padding=1),
                                    nn.ReLU())
        
        self.conv1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1))
                                
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1))
        
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1))
        '''
        self.downsample = nn.AvgPool2d(2)

        num_features = (3) * 3

        self.mlp1 = nn.Sequential(nn.Conv1d(num_features, 1024*3, 1),
                                nn.LeakyReLU(),
                                nn.Conv1d(1024*3, 512, 1),
                                nn.LeakyReLU(),
                                nn.Conv1d(512, 256, 1),
                                nn.LeakyReLU(),
                                nn.Conv1d(256, 256, 1),
                                nn.LeakyReLU(),
                                nn.Conv1d(256, 3, 1)) #if want RGB out change last output channel to 3
    
        d = 0.002
        #self.displacement = torch.Tensor([[0, 0], [-d, 0], [d, 0], [0, -d], [0, d]]).cuda()
        self.displacement = torch.Tensor([[0, 0], [-d, 0], [d, 0]]).cuda()


    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        #features_0 = F.grid_sample(x, p, mode='bicubic', align_corners=False) #features_0(N,C0,H',W')
        p = torch.cat([p + d for d in self.displacement], dim=2) #(N,num_points,5,2)

        features = F.grid_sample(x, p, mode='bilinear', align_corners=False)
        
        #x = self.conv0(x)
        #x = F.relu(self.conv1(x) + x)
        #features_1 = F.grid_sample(x, p, mode='bilinear', align_corners=False)
        #x = self.downsample(x)
        #x = F.relu(self.conv2(x) + x)
        #features_2 = F.grid_sample(x, p, mode='bilinear', align_corners=False)
        #x = self.downsample(x)
        #x = F.relu(self.conv3(x) + x)
        #features_3 = F.grid_sample(x, p, mode='bilinear', align_corners=False)

        

        #features = torch.cat((features_0), dim=1) #features(N,C1+C2+C3,num_points,5)
        features = torch.reshape(features, (features.shape[0], features.shape[1] * features.shape[-1], features.shape[2]))
        out = torch.clamp(self.mlp1(features), min=0.0, max=1.0)
        #out = self.mlp2(torch.reshape(features, (features.shape[0], features.shape[1] * features.shape[-1], features.shape[2])))

        return out.unsqueeze(-1)

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
        
class Mark_4(nn.Module):
    def __init__(self, in_channels=3):
        super(Mark_4, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU())
                                
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1))
        
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1))

        self.gat = MultiHeadGATLayer(256, 256, 3, merge='cat')

        self.mlp = nn.Sequential(nn.Conv2d(256*3, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 3, 1),
                                nn.Sigmoid()) #if want RGB out change last output channel to 3
    
        d = 0.003
        self.displacement = torch.Tensor([[0, 0], [-2*d, 0], [2*d, 0], [0, -2*d], [0, 2*d],
                                                [-d, -d], [d, d], [d, -d], [-d, d]]).cuda()
        #self.displacement = torch.Tensor([[0, 0]]).cuda()


    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        #features_0 = F.grid_sample(x, p, mode='bicubic', align_corners=False) #features_0(N,C0,H',W')
        p = torch.cat([p + d for d in self.displacement], dim=2) #(N,num_points,5,2)

        x = self.conv1(x)
        x = F.relu(self.conv2(x) + x)
        x = F.relu(self.conv3(x) + x)
        features = F.grid_sample(x, p, mode='bilinear', align_corners=False) #(N, C, num_points, num_neighbors+1)
        features = F.relu(self.gat(features)) #(N, C, num_points, 1)
        out = self.mlp(features)
        return out


if __name__ == '__main__':
    net = Mark_2()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))