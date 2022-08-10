import os
import torch
from torchvision.io import read_image, ImageReadMode
import torch.utils.data as data
from PIL import Image
import math
import numpy as np
class DemoLR(data.Dataset):
    def __init__(self, demo_dir, scale):
        self.scale = scale
        self.filelist = []
        for f in os.listdir(demo_dir):
            if f.find('.png') >= 0:
                self.filelist.append(os.path.join(demo_dir, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        lr = read_image(self.filelist[idx], ImageReadMode.RGB)
        return lr, self.scale, filename

    def __len__(self):
        return len(self.filelist)

class DemoHR(data.Dataset):
    def __init__(self, demo_dir, scale):
        self.scale = scale
        self.filelist = []
        for f in os.listdir(demo_dir):
            if f.find('.png') >= 0:
                self.filelist.append(os.path.join(demo_dir, f))
        self.filelist.sort()

    def __getitem__(self, idx, temp=True):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        hr = Image.open(self.filelist[idx]).convert('RGB')
        hr_w, hr_h = hr.size
        lr_w = math.floor(hr_w/self.scale + 10e-9)
        lr_h = math.floor(hr_h/self.scale + 10e-9)
        lr = hr.resize((lr_w, lr_h), Image.NEAREST)

        if temp:
            lr = lr.resize((hr_w, hr_h), Image.BICUBIC)
    

        lr = np.ascontiguousarray(np.array(lr).transpose(2, 0, 1))
        hr = np.ascontiguousarray(np.array(hr).transpose(2, 0, 1))
        return torch.from_numpy(lr).float(), torch.from_numpy(hr).float(), self.scale, filename

    def __len__(self):
        return len(self.filelist)