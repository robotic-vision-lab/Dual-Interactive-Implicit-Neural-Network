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
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1),
                                    nn.ReLU())
        
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1))
                                  
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1))
        
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1))

        self.downsample = nn.MaxPool2d(2)

        num_features = 256 + 256 + 256

        self.mlp = nn.Sequential(nn.Conv2d(num_features, 256*3, 1),
                                nn.ReLU(),
                                nn.Conv2d(256*3, 256*3, 1),
                                nn.ReLU(),
                                nn.Conv2d(256*3, 3, 1), 
                                nn.Sigmoid()) #if want RGB out change last output channel to 3

    def forward(self, x, p):
        #images x(N,C,H,W) and sample points p(N,H',W',2)
        #note that p specifies (x,y) corresponding to H[y]W[x]
        #features_0 = F.grid_sample(x, p, mode='bicubic', align_corners=False) #features_0(N,C0,H',W')
        x = self.conv0(x)
        x = self.conv1(x) + x
        features_1 = F.grid_sample(x, p, mode='bilinear', align_corners=False)
        x = self.downsample(x)
        x = self.conv2(x) + x
        features_2 = F.grid_sample(x, p, mode='bilinear', align_corners=False)
        x = self.downsample(x)
        x = self.conv3(x) + x
        features_3 = F.grid_sample(x, p, mode='bilinear', align_corners=False)

        features = torch.cat((features_1, features_2, features_3), dim=1)

        out = self.mlp(features)

        return out

if __name__ == '__main__':
    net = Mark_2()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))