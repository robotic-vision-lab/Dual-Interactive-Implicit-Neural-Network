import os
import torch
import torchvision
import random
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import PIL
torch.manual_seed(37)

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[-2:]
        new_h, new_w = self.output_size
        top = torch.randint(0, h - new_h, ())
        left = torch.randint(0, w - new_w, ())
        image = image[:, top: top + new_h, left: left + new_w]
        return image

class Rotation(object):
    def __init__(self, angles):
        self.angles = angles
    
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, interpolation=T.InterpolationMode.BILINEAR)


class DIV2K(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root_dir='D:\qhn8083\data\DIV2K', partition='train', downscale_factor=4, num_points=1000, transform=True, eval=False):
        '''
        Inputs: dir(str) - directory to the data folder
                partition - sub-dir in dataset folder (e.g. 'train', 'test', 'val')
        '''
        self.num_points= num_points
        self.partition = partition
        self.dir = os.path.join(root_dir, 'DIV2K_'+partition+'_HR')
        self.downscale_factor = downscale_factor
        self.transform = transform
        self.eval = eval
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
        img = torchvision.io.read_image(img_path).float()/255
        if self.partition == 'valid' and self.eval==True:
            lr_img = T.Resize((img.shape[1]//self.downscale_factor, img.shape[2]//self.downscale_factor), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(img)
            h = img.shape[1]
            w = img.shape[2]
            h_idx = torch.arange(-1 + 1/h, 1 + 1/h, 2/h).repeat(w,1).T.unsqueeze(-1)
            w_idx = torch.arange(-1 + 1/w, 1 + 1/w, 2/w).repeat(h,1).unsqueeze(-1)
            p = torch.cat((w_idx, h_idx), -1)
            return lr_img, p, img
        else:
            if self.transform:
                transform = T.Compose([T.RandomApply([Rotation([90])]),
                                        T.RandomHorizontalFlip(),
                                        RandomCrop(400)])
                img = transform(img)
            if self.partition == 'valid':
                s = self.downscale_factor
            else:
                s = torch.randint(1, self.downscale_factor + 1, ())
            lr_img = T.Resize((img.shape[1]//s, img.shape[2]//s), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(img)
            img = img.unsqueeze(0) #(1,C,H,W)
            p = 1.0 - 2 * torch.rand(1, self.num_points, 1, 2) #(1,num_points,1,2), value range (-1,1)
            gt = F.grid_sample(img, p, mode='nearest', align_corners=False) #(1,C,num_points,1)
            return lr_img, p.squeeze(0), gt.squeeze(0)

if __name__ == '__main__':
    dataset = DIV2K()
    img, p, gt = dataset[0]
    print(img, p, gt)
    print(img.size(), p.size(), gt.size())