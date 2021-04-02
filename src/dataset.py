import os
import torch
import torchvision
from torchvision import transforms as T
from torch.nn import functional as F
torch.manual_seed(37)
class GaussianNoise(object):
    def __init__(self, mean=0., std=10.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SRx4Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root_dir='data', partition='train', num_points=100, transform=True):
        '''
        Inputs: dir(str) - directory to the data folder
                partition - sub-dir in dataset folder (e.g. 'train', 'test', 'val')
        '''
        self.num_points= num_points
        self.partition = partition
        self.dir = os.path.join(root_dir, partition,'640_flir_hr')
        self.transform = transform
        self.img_paths = []
        for root, dirs, files in os.walk(os.path.join(self.dir)):
            for filename in files:
                if filename.lower().endswith('.jpg'):
                    self.img_paths.append(os.path.join(root,filename))
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.img_paths[idx]
        img = T.Grayscale()(torchvision.io.read_image(img_path)).float()
        lr_img = T.Resize((img.shape[1]//4, img.shape[2]//4))(img)
        #if self.transform:
            #transform = GaussianNoise(0., 10.)
            #lr_img = transform(lr_img)
        if self.partition == 'test':
            h = img.shape[1]
            w = img.shape[2]
            h_idx = torch.arange(-1 + 1/h, 1 + 1/h, 2/h).repeat(w,1).T.unsqueeze(-1)
            w_idx = torch.arange(-1 + 1/w, 1 + 1/w, 2/w).repeat(h,1).unsqueeze(-1)
            points = torch.cat((w_idx, h_idx), -1)
            return lr_img, points, img
        else:
            img = img.unsqueeze(0) #(1,C,H,W)
            points = 1.0 - 2 * torch.rand(1, self.num_points, 1, 2) #(1,num_points,1,2), value range (-1,1)
            gt_intensities = F.grid_sample(img, points, mode='nearest') #(1,C,num_points,1)
            return lr_img, points.squeeze(0), gt_intensities.squeeze(0), img.squeeze(0)

class DIV2K(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root_dir='$WORK/datasets/DIV2K', partition='train', downscale_factor=2, num_points=1000, transform=True):
        '''
        Inputs: dir(str) - directory to the data folder
                partition - sub-dir in dataset folder (e.g. 'train', 'test', 'val')
        '''
        self.num_points= num_points
        self.partition = partition
        self.dir = os.path.join(root_dir, 'DIV2K_'+partition+'_HR')
        print(self.dir)
        self.transform = transform
        self.img_paths = []
        for root, dirs, files in os.walk(os.path.join(self.dir)):
            for filename in files:
                if filename.lower().endswith('.png'):
                    self.img_paths.append(os.path.join(root,filename))
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.img_paths[idx]
        img = torchvision.io.read_image(img_path).float()
        lr_img = T.Resize((img.shape[1]//downscale_factor, img.shape[2]//downscale_factor))(img)
        #if self.transform:
            #transform = GaussianNoise(0., 10.)
            #lr_img = transform(lr_img)
        if self.partition == 'valid' and False:
            h = img.shape[1]
            w = img.shape[2]
            h_idx = torch.arange(-1 + 1/h, 1 + 1/h, 2/h).repeat(w,1).T.unsqueeze(-1)
            w_idx = torch.arange(-1 + 1/w, 1 + 1/w, 2/w).repeat(h,1).unsqueeze(-1)
            points = torch.cat((w_idx, h_idx), -1)
            return lr_img, points, img
        else:
            img = img.unsqueeze(0) #(1,C,H,W)
            points = 1.0 - 2 * torch.rand(1, self.num_points, 1, 2) #(1,num_points,1,2), value range (-1,1)
            gt_intensities = F.grid_sample(img, points, mode='biliear') #(1,C,num_points,1)
            return lr_img, points.squeeze(0), gt_intensities.squeeze(0), img.squeeze(0)

if __name__ == '__main__':
    dataset = DIV2K()
    img, p, gt = dataset[0]
    print(img, p, gt)
    print(img.size(), p.size(), gt.size())