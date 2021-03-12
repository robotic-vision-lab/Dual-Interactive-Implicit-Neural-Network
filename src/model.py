import torch
import torch.nn as nn
import torch.nn.functional as F

class Mark_1(nn.Module):
    def __init__(self, in_channels=1):
        super(Mark_1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256))
        self.max_pool = nn.MaxPool2d(2)
        
        num_features = in_channels + 32 + 128 + 256

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 1, 1),
                                nn.ReLU()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        features_0 = F.grid_sample(x, p) #features_0(N,C0,H',W')

        x = self.conv1(x)
        features_1 = F.grid_sample(x, p) #features_1(N,C1,H',W')
        x = self.max_pool(x)

        x = self.conv2(x)
        features_2 = F.grid_sample(x, p) #features_2(N,C2,H',W')
        x = self.max_pool(x)

        x = self.conv3(x)
        features_3 = F.grid_sample(x, p) #features_3(N,C3,H',W')

        features = torch.cat((features_0, features_1, features_2, features_3), dim=1)  #features(N,C0+C1+C2+C3,H',W')
        #features = torch.reshape(features, (features.shape[0], features.shape[2]*features.shape[3], -1)) #features(N,H'*W',C0+C1+C2+C3)

        out = self.mlp(features) #features(N,C,H',W')

        return out

class Mark_2(nn.Module):
    def __init__(self, in_channels=1):
        super(Mark_2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 49, padding=24),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 25, padding=12),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 13, padding=6),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 7, padding=3),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256))
        self.max_pool = nn.MaxPool2d(2)

        self.Rconv25 = nn.Sequential(nn.Conv2d(in_channels, 128, 49, padding=24),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 32, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 16, 1),
                                    nn.ReLU())
        self.upsample = nn.PixelShuffle(4)
        num_features = in_channels + 32 + 128 + 256 + 16

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 1, 1),
                                nn.ReLU()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]

        r25 = self.Rconv25(x) #(N,16,H,W)
        hr_out = self.upsample(r25)
        feature_r25 = F.grid_sample(r25, p, mode='bilinear') #(N,16,H',W')

        features_0 = F.grid_sample(x, p, mode='bilinear') #features_0(N,C0,H',W')

        x = self.conv1(x)
        features_1 = F.grid_sample(x, p, mode='bilinear') #features_1(N,C1,H',W')
        x = self.max_pool(x)

        x = self.conv2(x)
        features_2 = F.grid_sample(x, p, mode='bilinear') #features_2(N,C2,H',W')
        x = self.max_pool(x)

        x = self.conv3(x)
        features_3 = F.grid_sample(x, p, mode='bilinear') #features_3(N,C3,H',W')

        features = torch.cat((features_0, features_1, features_2, features_3, feature_r25), dim=1)  #features(N,C0+C1+C2+C3,H',W')
        #features = torch.reshape(features, (features.shape[0], features.shape[2]*features.shape[3], -1)) #features(N,H'*W',C0+C1+C2+C3)

        out = self.mlp(features) #features(N,C,H',W')

        return out, hr_out


class Mark_3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Mark_3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, 5, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64))
        
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128))

        self.upsample1 = nn.Sequential(nn.Conv2d(32, 32*2, 3, padding=1),
                                        nn.ReLU(),
                                        nn.PixelShuffle(2))

        self.upsample2 = nn.Sequential(nn.Conv2d(64, 64*2, 3, padding=1),
                                        nn.ReLU(),
                                        nn.PixelShuffle(2))

        self.upsample3 = nn.Sequential(nn.Conv2d(128, 128*2, 3, padding=1),
                                        nn.ReLU(),
                                        nn.PixelShuffle(2))

        self.max_pool = nn.MaxPool2d(2)

        num_features = in_channels + 32 + 64 + 128

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 256, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, out_channels, 1),
                                nn.ReLU()) #if want RGB out change last output channel to 3

        d = 0.03
        self.displacements = []
        for x in range(2):
            for y in [-1, 1]:
                base = [0, 0]
                base[x] = y * d
                self.displacements.append(base)
        self.displacments = torch.Tensor(self.displacements)

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        features_0 = F.grid_sample(x, p) #features_0(N,C0,H',W')

        x = self.conv1(x)
        x = self.upsample1(x)
        features_1 = F.grid_sample(x, p) #features_1(N,C1,H',W')

        x = self.conv2(x)
        x = self.upsample2(x)
        features_2 = F.grid_sample(x, p) #features_1(N,C2,H',W')

        x = self.conv3(x)
        x = self.upsample3(x)
        features_3 = F.grid_sample(x, p) #features_1(N,C2,H',W')

        features = torch.cat((features_0, features_1, features_2, features_3), dim=1)  #features(N,C0+C1+C2+C3,H',W')
        #features = torch.reshape(features, (features.shape[0], features.shape[2]*features.shape[3], -1)) #features(N,H'*W',C0+C1+C2+C3)

        out = self.mlp(features) #features(N,C,H',W')

        return out, x


class VAE(nn.Module):
    def __init__(self, in_channels=1):
        super(VAE, self).__init__()
        
        

    def forward(self, lr, hr):
        code_lr = self.encode_lr(lr)
        code_lr += torch.rand_like(code_lr)
        code_hr = self.encode_hr(hr)
        return code_lr, code_hr


'''
class Mark_2(nn.Module):
    def __init__(self, in_channels=1, H=120, W=160):
        super(Mark_2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels + 32, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels + 32 + 128, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256))
        
        num_features = in_channels + 32 + 128 + 256

        self.fc = nn.Sequential(nn.Conv2d(num_features, 1, 1),
                                nn.ReLU(),
                                nn.AvgPool2d(2))
        self.mlp = nn.Sequential(nn.Conv2d(2 + H//2 * W//2, 128, 1),
                                nn.ReLU(),
                                nn.Conv2d(128, 64, 1),
                                nn.ReLU(),
                                nn.Conv2d(64, 1, 1),
                                nn.ReLU())

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        N, H_p, W_p, f = p.size()


        x1 = self.conv1(x) #x1(N,C1,H,W)
        x1 = torch.cat((x1, x), dim=1) #(N,C+C1,H,W)

        x2 = self.conv2(x1) #(N,C2,H,W)
        x2 = torch.cat((x2, x1), dim=1) #(N,C+C1+C2,H,W)

        x3 = self.conv3(x2) #(N,C3,H,W)
        x3 = torch.cat((x3, x2), dim=1) #(N,C+C1+C2+C3,H,W)

        global_features = self.fc(x3).reshape(N, -1) #(N,1,H//2,W//2)
        global_features = global_features.unsqueeze(-1).unsqueeze(-1).expand(N, -1, H_p, W_p) #(N,H*W,H,W)
        local_features = p.permute(0, 3, 1, 2) #(N,2,H,W)
        features = torch.cat((local_features, global_features), dim=1) #(N,2+H*W,H,W)
        out = self.mlp(features)
        return out
'''

if __name__ == '__main__':
    from dataset import SRx4Dataset
    dataset = SRx4Dataset()
    img, p, gt = dataset[0]
    net = Mark_1()
    print(net(img.unsqueeze(0),p.unsqueeze(0)).size())
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))