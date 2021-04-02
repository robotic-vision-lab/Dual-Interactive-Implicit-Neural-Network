import torch
import torch.nn as nn
import torch.nn.functional as F

class Mark_1(nn.Module):
    def __init__(self, in_channels=1):
        super(Mark_1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, 3),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(256, 128, 3),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3),
                                    nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(512, 256, 3),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3),
                                    nn.ReLU())
        
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        
        num_features = in_channels + 32 + 64 + 128 + 256

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 3, 1),
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        features_0 = F.grid_sample(x, p, mode='bicubic') #features_0(N,C0,H',W')

        x = self.conv1(x)
        features_1 = F.grid_sample(x, p, mode='bicubic') #features_1(N,C1,H',W')
        x = self.unshuffle(x)

        x = self.conv2(x)
        features_2 = F.grid_sample(x, p, mode='bicubic') #features_2(N,C2,H',W')
        x = self.unshuffle(x)

        x = self.conv3(x)
        features_3 = F.grid_sample(x, p, mode='bicubic') #features_3(N,C3,H',W')
        x = self.unshuffle(x)

        x = self.conv4(x)
        features_4 = F.grid_sample(x, p, mode='bicubic') #features_4(N,C4,H',W')

        features = torch.cat((features_0, features_1, features_2, features_3, features_4), dim=1)  #features(N,C0+C1+C2+C3,H',W')
        #features = torch.reshape(features, (features.shape[0], features.shape[2]*features.shape[3], -1)) #features(N,H'*W',C0+C1+C2+C3)

        out = self.mlp(features) #features(N,C,H',W')

        return out


if __name__ == '__main__':
    from dataset import SRx4Dataset
    dataset = SRx4Dataset()
    img, p, gt = dataset[0]
    net = Mark_1()
    print(net(img.unsqueeze(0),p.unsqueeze(0)).size())
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))