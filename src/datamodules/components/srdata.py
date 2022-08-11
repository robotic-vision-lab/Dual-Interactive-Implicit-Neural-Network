import os
import pickle
import random
import glob
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from pathlib import Path
from torchvision import transforms
from PIL import Image
import math
DATASET_DIR_STRUCTURE = {
    'DIV2K': {
        'train': {
            'hr_dir': 'DIV2K_train_HR',
            'lr_dir': 'DIV2K_train_LR_bicubic'
        },
        'test': {
            'hr_dir': 'DIV2K_test_HR',
            'lr_dir': 'DIV2K_test_LR_bicubic'
        }
    },
    'benchmark': {
        'B100': {
            'hr_dir': 'B100/HR',
            'lr_dir': 'B100/LR_bicubic'
        },
        'Set5': {
            'hr_dir': 'Set5/HR',
            'lr_dir': 'Set5/LR_bicubic'
        },
        'Set14': {
            'hr_dir': 'Set14/HR',
            'lr_dir': 'Set14/LR_bicubic'
        },
        'Urban100': {
            'hr_dir': 'Urban100/HR',
            'lr_dir': 'Urban100/LR_bicubic'
        }
    }
}

class SRData(Dataset):
    def __init__(self, root="./data/",
                        name='DIV2K',
                        split='train',
                        file_ext='.png',
                        bin=False,
                        reset_bin=False,
                        scales=[2],
                        patch_size=96,
                        augment=True):
        self.file_ext = file_ext
        self.scales = scales
        self.patch_size = patch_size
        self.bin = bin
        self.reset_bin = reset_bin
        self.augment = augment
        self._set_paths(root, name, split)
        self.names_lr, self.names_hr = self._scan(self.lr_dir, self.hr_dir, file_ext)

        if bin:
            self.hr_dir_bin = self.dataset_dir / 'bin' / DATASET_DIR_STRUCTURE[name][split]['hr_dir']
            self.lr_dir_bin = self.dataset_dir / 'bin' / DATASET_DIR_STRUCTURE[name][split]['lr_dir']
            self.hr_dir_bin.mkdir(parents=True, exist_ok=True)
            for scale in self.scales:
                (self.lr_dir_bin / 'X{}'.format(scale)).mkdir(parents=True, exist_ok=True)
            self.names_lr_bin = {scale:[] for scale in self.scales}
            self.names_hr_bin = []
            for img_f in self.names_hr:
                bin_f = str(self.hr_dir_bin / Path(img_f).name.replace(file_ext, '.pt'))
                self._check_and_load(bin_f, img_f)
                self.names_hr_bin.append(bin_f)
            for scale in self.scales:
                for img_f in self.names_lr[scale]:
                    bin_f = str(self.lr_dir_bin / 'X{}'.format(scale) / Path(img_f).name.replace(file_ext, '.pt'))
                    self._check_and_load(bin_f, img_f)
                    self.names_lr_bin[scale].append(bin_f)
            
    def __len__(self):
        return len(self.names_hr)

    def __getitem__(self, idx):
        sample = {}
        for scale in self.scales:
            lr, hr, filename = self._load_file(idx, scale)
            lr_patch, hr_patch = self.get_patch(lr, hr, scale, self.patch_size)
            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5
                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x
                lr_patch = augment(lr_patch)
                hr_patch = augment(hr_patch)
            sample[scale] = (lr_patch.float(), hr_patch.float(), filename)
        return sample

    def _load_file(self, idx, scale):
        if self.bin:
            f_hr = self.names_hr_bin[idx]
            f_lr = self.names_lr_bin[scale][idx]
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
        else:
            f_hr = self.names_hr[idx]
            f_lr = self.names_lr[scale][idx]
            hr = read_image(f_hr, ImageReadMode.RGB)/255.
            lr = read_image(f_lr, ImageReadMode.RGB)/255.
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        return lr, hr, filename

    def _scan(self, lr_dir, hr_dir, file_ext):
        names_hr = sorted(glob.glob(os.path.join(str(hr_dir), '*'+file_ext)))
        names_lr = {scale:[] for scale in self.scales}
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for scale in self.scales:
                names_lr[scale].append(str(lr_dir / 'X{}'.format(scale) / '{}x{}{}'.format(filename, scale, self.file_ext)))
        return names_lr, names_hr
    
    def _set_paths(self, root, name, split):
        self.dataset_dir = Path(root) / name
        self.hr_dir = self.dataset_dir / DATASET_DIR_STRUCTURE[name][split]['hr_dir']
        self.lr_dir = self.dataset_dir / DATASET_DIR_STRUCTURE[name][split]['lr_dir']

    def _check_and_load(self, f, img):
        if not Path(f).exists() or self.reset_bin:
            print('Saving binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(read_image(img, ImageReadMode.RGB)/255., _f)
    

    def get_patch(self, lr, hr, scale, patch_size):
        #lr and hr are of [channels, height, width]
        #scale must be int
        #if patch_size is set to 0, use whole images
        if patch_size == 0:
            lr_h, lr_w = lr.shape[-2:]
            hr = hr[:, 0:lr_h * scale, 0:lr_w * scale]
            return lr, hr
        else:
            lr_h, lr_w = lr.shape[-2:]
            hr_patch_size = patch_size * scale
            lr_patch_size = patch_size

            #get random top-left location in lr
            lr_patch_top = random.randrange(0, lr_h - lr_patch_size + 1)
            lr_patch_left = random.randrange(0, lr_w - lr_patch_size + 1)
            hr_patch_top = lr_patch_top * scale
            hr_patch_left = lr_patch_left * scale
            return [lr[:, lr_patch_top:lr_patch_top+lr_patch_size, lr_patch_left:lr_patch_left+lr_patch_size],
                    hr[:, hr_patch_top:hr_patch_top+hr_patch_size, hr_patch_left:hr_patch_left+hr_patch_size]
            ]

def resize_fn(img, size):
    return transforms.Resize(size=size,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            antialias=True)(img)

class SRDataDownsample(Dataset):
    def __init__(self, root="./data/",
                        name='DIV2K',
                        split='train',
                        file_ext='.png',
                        scales=[2],
                        patch_size=96,
                        augment=True):
        self.file_ext = file_ext
        self.scales = scales
        self.patch_size = patch_size
        self.augment = augment
        self._set_paths(root, name, split)
        self._scan(self.hr_dir, file_ext)
            
    def __len__(self):
        return len(self.names_hr)

    def __getitem__(self, idx):
        sample = {}
        for scale in self.scales:
            hr, filename = self._load_file(idx, scale)
            lr_patch, hr_patch = self.get_patch(hr, scale, self.patch_size)
            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5
                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x
                lr_patch = augment(lr_patch)
                hr_patch = augment(hr_patch)
            sample[scale] = (lr_patch.float()/255., hr_patch.float()/255., filename)
        return sample

    def _load_file(self, idx, scale):
        f_hr = self.names_hr[idx]
        hr = read_image(f_hr, ImageReadMode.RGB)
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        return hr, filename

    def _scan(self, hr_dir, file_ext):
        self.names_hr = sorted(glob.glob(os.path.join(str(hr_dir), '*'+file_ext)))
    
    def _set_paths(self, root, name, split):
        self.dataset_dir = Path(root) / name
        self.hr_dir = self.dataset_dir / DATASET_DIR_STRUCTURE[name][split]['hr_dir']

    def get_patch(self, hr, scale, patch_size):
        #lr and hr are of [channels, height, width]
        #scale must be int
        #if patch_size is set to 0, use whole images
        if patch_size == 0:
            lr_h = round(hr.shape[-2] / scale) 
            lr_w = round(hr.shape[-1] / scale)
            lr = resize_fn(hr, (lr_h, lr_w))
            return lr, hr
        else:
            hr_h, hr_w = hr.shape[-2:]
            hr_patch_size = round(patch_size * scale)
            hr_patch_top = random.randrange(0, hr_h - hr_patch_size + 1)
            hr_patch_left = random.randrange(0, hr_w - hr_patch_size + 1)
            hr = hr[:, hr_patch_top:hr_patch_top+hr_patch_size, hr_patch_left:hr_patch_left+hr_patch_size]
            lr = resize_fn(hr, (patch_size, patch_size))
            return lr, hr