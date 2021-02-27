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
        
        num_features = 1 + 32 + 128 + 256
        self.mlp = nn.Sequential(nn.Linear(num_features, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1)) #if want RGB out, change to nn.Linear(64, 3)
                                 

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
        features = torch.reshape(features, (features.shape[0], features.shape[2]*features.shape[3], -1)) #features(N,H'*W',C0+C1+C2+C3)

        out = self.mlp(features) #features(N,H'*W',1)

        return out

if __name__ == '__main__':
    from dataset import SRx4Dataset
    dataset = SRx4Dataset()
    img, p, gt = dataset[0]
    net = Mark_1()
    print(net(img.unsqueeze(0),p.unsqueeze(0)))