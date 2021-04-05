import os
import torch
import torchvision
from torchvision import transforms as T
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
        lr_imgs = []
        points = []
        gts = []
        if self.partition == 'valid' and self.eval==True:
            if self.transform:
                transform = T.Compose([T.FiveCrop((128,256))])
                imgs = transform(img)
            for img in imgs:
                lr_imgs.append(T.Resize((img.shape[1]//self.downscale_factor, img.shape[2]//self.downscale_factor), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(img))
                h = img.shape[1]
                w = img.shape[2]
                h_idx = torch.arange(-1 + 1/h, 1 + 1/h, 2/h).repeat(w,1).T.unsqueeze(-1)
                w_idx = torch.arange(-1 + 1/w, 1 + 1/w, 2/w).repeat(h,1).unsqueeze(-1)
                points.append(torch.cat((w_idx, h_idx), -1))
            lr_imgs = torch.stack(lr_imgs)
            points = torch.stack(points)
            imgs = torch.stack(imgs)
            return lr_imgs, points, imgs
        else:
            if self.transform:
                transform = T.Compose([RandomCrop(512), T.TenCrop((256, 256))])
                imgs = transform(img)
            s = torch.randint(2, self.downscale_factor + 1, ())
            for img in imgs:
                lr_imgs.append(T.Resize((img.shape[1]//s, img.shape[2]//s), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(img))
                img = img.unsqueeze(0) #(1,C,H,W)
                p = 1.0 - 2 * torch.rand(1, self.num_points, 1, 2) #(1,num_points,1,2), value range (-1,1)
                gt = F.grid_sample(img, p, mode='bicubic', align_corners=False) #(1,C,num_points,1)
                points.append(p.squeeze(0))
                gts.append(gt.squeeze(0))
            lr_imgs = torch.stack(lr_imgs)
            points = torch.stack(points)
            gts = torch.stack(gts)
            return lr_imgs, points, gts

if __name__ == '__main__':
    dataset = DIV2K()
    img, p, gt = dataset[0]
    print(img, p, gt)
    print(img.size(), p.size(), gt.size())